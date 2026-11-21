# inkling

An inference engine specialized for **one model on one machine**: Inkling-2-Mini (GGUF `UD-Q4_K_XL`, 16.4 GiB), a looped
Mixture-of-Experts model (24 physical blocks, 12 of them run three times, so 48 layer passes per token; 128 experts, top 8;
sliding-window attention on 5 of every 6 layers) on a laptop with an RTX 5080 Laptop GPU (16 GB VRAM).

It starts from [Unsloth's llama.cpp](https://github.com/unslothai/llama.cpp) build b12588, which already has the
`inkling2` architecture and the DSpark block drafter, and specializes it with the method of
[qwen-next](https://github.com/0xE2E/qwen-next) ("the art of moving bytes"): measure where each round's time goes, change one
thing, keep it only if the output stays the same and the time drops.

Every engine change is behind an environment switch that is off by default (`docs/SWITCHES.md`). Most accepted switches are
exact: greedy output tokens identical with and without them. Four change rounding (fused depth mix in decode and prefill, GQA
head packing, a float matvec kernel replacing a TF32 one); each was kept only when its KL divergence against a reference run
was no larger than the divergence the stock engine shows between two of its own kernel choices.

## Results

RTX 5080 Laptop, Windows 11, one user, greedy decoding, same model file (UD-Q4_K_XL) and the same DSpark drafter
(`dspark-Inkling-2-Mini-Q8_0.gguf`, 3 draft tokens for stock) in every row, 128k context window.

Same replayed coding-agent session for all three rows, cut at 43k and 81k tokens:

| Configuration | Prefill 43k cold | Prefill 43k->81k | Decode at 43k | Decode at 81k |
|---|---|---|---|---|
| Unsloth llama.cpp b12588, Unsloth Studio-style flags (q8_0 KV cache, `--fit-target 2092`) | 445 | 313 | 21.1 | 14.3 |
| Unsloth llama.cpp b12588, tuned flags (`--ubatch-size 2048 --fit-target 512 --moe-cache-mib auto`) | 827 | 759 | 26.9 | 19.0 |
| **inkling, default profile** | **2,292** | **1,899** | **97.4** | **73.1** |

Against the best stock setup that is **2.8x / 2.5x prefill and 3.6x / 3.8x decode**; against the Studio-style setup
5.2x / 6.1x prefill and 4.6x / 5.1x decode.

Across five replayed sessions (tool calls and reasoning included, up to 300 generated tokens) inkling averages **2,331**
tok/s prefill on a cold 43k prompt, **88.7** tok/s decode at ~43k (range 74-97) and **81.4** at ~85k (range 61-104).
Decode speed at long context follows how much of the drafter's guess the model accepts (39-100% across these sessions),
not the context length: a verify round takes about 37 ms at 43k and 42 ms at 85k.

Read these with care:

- One model, one quantization, one machine. Nothing here is a general benchmark.
- The stock rows are single runs of one session; stock ran with the GPU warmer (about 60 C at start, lower clocks) than
  the inkling rows, so they may understate stock somewhat.
- The quant was chosen by measurement: with the same engine work at the time, Q4_K_XL decoded at 61 tok/s against 37 for
  the 21.3 GiB Q8_K_XL, which does not fit 16 GB and misses its expert cache far more often. Model-card quality of Q4_K_XL
  against Q8: KLD 0.0107, 98% same top token.
- A 128k window runs at the same speed as 90k at equal fill. 256k does not fit 16 GB with this profile yet (see below).

## Where the time went

- **Kernel count, not bandwidth.** The 48 layer passes ran about 7,600 tiny kernels per token. Fusing the loop controllers,
  short convolutions, clip, SiTU, depth mix and bias chains halved the graph and gave +50% decode.
- **The loop.** Caching experts per physical block instead of per layer pass took the hit rate from 81% to 94%.
- **Idle memory.** The compute buffer and the unfilled part of the KV cache are lent to the expert cache during decode.
- **Prefill glue.** Real matmuls were 16% of a prefill batch: a matrix-vector kernel was being fed 2,048 tokens and the
  sliding-window layers fell back to a dense path from the second batch. Fixing both and overlapping expert uploads with
  compute took prefill from ~860 to ~2,300 tok/s.
- **The drafter.** Trimmed chain, ids from the graph, a precomputed head product, and an adaptive draft length.

A per-kernel accounting of the final profile puts most remaining decode time at or near the memory-bandwidth floor; the
lossless ceiling for this file on this GPU is estimated at roughly 90-95 tok/s at 43k with this drafter.

## Use

Build: `docs/BUILD.md`. Start:

    .\start-inkling.ps1 -Model <Inkling-2-Mini-UD-Q4_K_XL.gguf> -DSpark <dspark-Inkling-2-Mini-Q8_0.gguf>

The server speaks the llama.cpp / OpenAI-compatible HTTP API on `127.0.0.1:8080`.

## Open items

- **256k context.** It runs, but the fit spills 6 of 25 layers and 1 GB of KV to RAM and drops the expert cache: prefill
  ~1,300-1,500 tok/s, decode 11-15 tok/s. Planned: keep all layers on the GPU and lend unfilled KV and compute memory to a
  small cache.
- **Turn warm-up.** The first ~300 ms after a large prompt run at about half speed while the cache refills and pinned
  buffers are set up.
- **Prefill** still uploads each looped block three times per batch.

## Licence

MIT (`LICENSE`). The `engine/` tree is llama.cpp (MIT, ggml authors) with Unsloth's changes and this project's changes.
