# inkling

**A specialized inference engine for [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on a
single consumer GPU.**

On a laptop RTX 5080 (16 GB), with the same model file and the same drafter as stock llama.cpp:

| | Stock llama.cpp | inkling | Speedup |
|---|---:|---:|---:|
| Decode at 43k context | 26.9 tok/s | **97.4 tok/s** | **3.6x** |
| Decode at 81k context | 19.0 tok/s | **73.1 tok/s** | **3.8x** |
| Prefill, 43k-token prompt | 827 tok/s | **2,292 tok/s** | **2.8x** |

Measured at a 128k context window; 90k and 128k run at the same speed at equal fill.

Built on [Unsloth's llama.cpp](https://github.com/unslothai/llama.cpp) with the method from
[The Art of Moving Bytes](https://0xe2e.github.io/qwen-next-blog/), first applied in
[qwen-next](https://github.com/0xE2E/qwen-next).

---

## Contents

- [Quick start](#quick-start)
- [Results](#results)
- [How it works](#how-it-works)
- [Correctness](#correctness)
- [Open items](#open-items)
- [Acknowledgements and licence](#acknowledgements-and-licence)

## Quick start

**1. Get the model.** Download two files from
[unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF):

- `Inkling-2-Mini-UD-Q4_K_XL.gguf` (16.4 GiB), the model
- `dspark-Inkling-2-Mini-Q8_0.gguf`, the DSpark speculative drafter

**2. Build the engine.** Windows, Visual Studio 2022 and CUDA 13.3. See [docs/BUILD.md](docs/BUILD.md).

**3. Start the server.**

```powershell
.\start-inkling.ps1 -Model <path>\Inkling-2-Mini-UD-Q4_K_XL.gguf -DSpark <path>\dspark-Inkling-2-Mini-Q8_0.gguf
```

The server listens on `http://127.0.0.1:8080` with the llama.cpp / OpenAI-compatible API. By default it uses a
128,000-token context.

| Option | Effect |
|---|---|
| `-Context 90000` | context window |
| `-Port 8081`, `-BindHost 0.0.0.0` | where to listen |
| `-NoDraft` | run without the drafter |
| `-Stock` | no engine switches (the measured llama.cpp floor) |
| `-DryRun` | print the command and environment, do not start |

You can also set the paths once with `INK_MODEL`, `INK_DSPARK` and `INK_BIN`.

## Results

**Setup.** RTX 5080 Laptop GPU (16 GB), Windows 11, one user, greedy decoding, 128k context window. Every row uses the same
model file (`UD-Q4_K_XL`) and the same DSpark drafter. The prompts are replayed coding-agent sessions that include tool
calls and reasoning.

### Same session, side by side

One session, cut at 43k and 81k tokens. The prefill column at 81k is the incremental 43k to 81k step.

| Configuration | Prefill 43k | Prefill 43k→81k | Decode 43k | Decode 81k |
|---|---:|---:|---:|---:|
| Unsloth llama.cpp b12588, Unsloth Studio-style flags¹ | 445 | 313 | 21.1 | 14.3 |
| Unsloth llama.cpp b12588, tuned flags² | 827 | 759 | 26.9 | 19.0 |
| **inkling, default profile** | **2,292** | **1,899** | **97.4** | **73.1** |

¹ `q8_0` KV cache, `--fit-target 2092`, DSpark with 3 draft tokens.
² `--ubatch-size 2048 --fit-target 512 --moe-cache-mib auto`, DSpark with 3 draft tokens.

Against the Studio-style setup, the gain is 5.2x / 6.1x on prefill and 4.6x / 5.1x on decode.

### Five sessions

Averages over five replayed sessions, generating up to 300 tokens each:

| Context | Prefill | Decode, mean | Decode, range |
|---|---:|---:|---:|
| ~43k | 2,331 tok/s (cold prompt) | **88.7 tok/s** | 74 – 97 |
| ~85k | 1,898 tok/s (incremental) | **81.4 tok/s** | 61 – 104 |

At long context, speed depends on how often the model accepts the drafter's guesses, not on the length of the context.
Acceptance ranged from 39% to 100% across these sessions, while one verify round took about 37 ms at 43k and 42 ms at 85k.

### Read with care

- This covers one model, one quantization and one machine. It is not a general benchmark.
- The stock rows are single runs. They started with a warmer GPU (about 60 °C, lower clocks), so they may understate stock
  somewhat.
- `UD-Q4_K_XL` was chosen by measurement. The 21.3 GiB `UD-Q8_K_XL` does not fit 16 GB, misses its expert cache far more
  often, and decoded at 37 against 61 tok/s at the time. Unsloth's model card puts Q4_K_XL at KLD 0.0107 against Q8, with
  98% same top token.

## How it works

Inkling-2-Mini is a looped Mixture-of-Experts model. It has 24 physical blocks, and 12 of them run three times, so every
token makes 48 layer passes. Each layer routes to 8 of 128 experts, and 5 of every 6 layers use sliding-window attention.
At Q4 the file is 16.4 GiB, larger than the GPU's memory, so part of the experts live in system RAM.

The engine was built one measured change at a time: find where a round's time goes, change one thing, and keep it only if
the output stays the same and the time drops.

| Where time was lost | What changed | Effect |
|---|---|---|
| ~7,600 tiny GPU kernels per token | fused the loop controllers, short convolutions, clip, SiTU, depth mix and bias chains | graph halved, decode +50% |
| The expert cache treated each loop pass as a new layer | cache experts per physical block | hit rate 81% → 94% |
| Idle VRAM during decode | lend the compute buffer and the unfilled KV cache to the expert cache | ~1.7 GB more cache |
| Real math was 16% of a prefill batch | a matrix-vector kernel was being fed 2,048 tokens, and sliding-window layers fell to a dense path | fixed both; prefill ~860 → ~1,560 |
| Expert uploads serialized with prefill compute | upload ahead of the layer on a second stream, into ring slots borrowed from the cache | prefill ~1,560 → ~2,300 |
| Drafter overhead | trimmed draft chain, token ids from the graph, precomputed head product, draft length that adapts to acceptance | several ms per round |

A per-kernel accounting of the final profile puts most of the remaining decode time at or near the GPU's memory-bandwidth
floor. With this drafter, the estimated lossless ceiling for this file on this GPU is about 90-95 tok/s at 43k.

## Correctness

Every engine change sits behind an `AMB_*` environment switch that is **off by default**. With no switch set, the engine
behaves like the llama.cpp build it started from. [docs/SWITCHES.md](docs/SWITCHES.md) lists each switch, how it was
checked, and the ideas that were measured and rejected.

- **Exact.** Most accepted switches leave the output unchanged: greedy output tokens are identical with and without them.
- **Rounding.** Four switches change rounding: the fused depth mix in decode and in prefill, GQA head packing, and a float
  matvec kernel that replaces a TF32 one. Each was kept only when its KL divergence against a reference run was no larger
  than the divergence the stock engine shows between two of its own kernel choices.

## Open items

- **256k context.** It runs, but the fit spills 6 of 25 layers and 1 GB of KV cache to RAM and drops the expert cache:
  prefill ~1,300-1,500 tok/s, decode 11-15 tok/s. The plan is to keep all layers on the GPU and lend unfilled KV and
  compute memory to a small expert cache.
- **Turn warm-up.** For the first ~300 ms after a large prompt, decode runs at about half speed while the cache refills.
- **Prefill uploads.** Each looped block is still uploaded three times per batch.

## Acknowledgements and licence

- [Thinking Machines](https://huggingface.co/thinkingmachines/Inkling-2-Mini) for Inkling-2-Mini.
- [Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) for the GGUF quantizations, the DSpark drafter, and the
  [llama.cpp build](https://github.com/unslothai/llama.cpp) this engine starts from.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) and the ggml authors.

MIT licence, see [LICENSE](LICENSE). The `engine/` tree is llama.cpp (MIT) with Unsloth's changes and this project's changes.
