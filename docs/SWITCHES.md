# Engine switches

Every engine change is behind an environment variable and is off by default, so the engine with no `AMB_*` variable set
behaves like the seed. `start-inkling.ps1` sets the accepted profile. `AMB` stands for "the art of moving bytes", the
method this engine was built with; `AMB_INK_*` switches are specific to the Inkling architecture.

"Exact" means greedy output tokens were identical with and without the switch on replayed sessions. "KLD" means the switch
changes rounding; it was accepted only when its KL divergence against a reference run was no larger than the divergence the
stock engine shows between two of its own kernel choices (about 0.0027-0.0028 on the decode path, 0.0047 on the prefill path,
16k tokens of real text).

## Accepted (set by the starter)

| Switch | What it does | Check |
|---|---|---|
| `AMB_MOE_BIGBATCH_BYPASS=1` | large batches upload the experts they need straight from host memory instead of through the cache | exact |
| `AMB_MOE_BATCH_COPY=1` | batched expert copies | exact |
| `AMB_IDS_ASYNC=1` | routing ids read back asynchronously into pinned memory | exact |
| `AMB_MOE_LOOP_DEDUP=1` | the expert cache is keyed by physical block, so the 3 passes of a looped block share cached experts (hit rate 81% -> 94%) | exact |
| `AMB_INK_GRAPH_LEAN=1`, `AMB_INK_FUSE_CLIP=1`, `AMB_INK_FUSE_CONV=1`, `AMB_INK_FUSE_CTRL=1`, `AMB_INK_FUSE_SITU=1` | fused kernels for the small-batch graph (q/k clip, short convolutions with rollback slots, loop controllers, SiTU) | exact |
| `AMB_INK_FUSE_MIX=1` | fused depth mix for small batches | KLD |
| `AMB_INK_GQA_PACK=1` | flash attention reads each KV head once for its 3 query heads | KLD |
| `AMB_INK_REL_MASK_DEC=1` | relative-position bias and mask built by one kernel in decode | exact |
| `AMB_MMVF_SMALL=1` | float matmuls with at most 128 rows and 2-8 tokens use the matvec kernel (the stock path used a TF32 kernel) | KLD |
| `AMB_ELASTIC_COMPUTE=1`, `AMB_ELASTIC_COMPUTE_SMALL_MAX=8` | the compute buffer above what an 8-token graph needs is lent to the expert cache during decode (~1.7 GB) | exact |
| `AMB_ELASTIC_KV=3` | global KV memory is mapped as the context fills; the unmapped part is lent to the expert cache | exact |
| `AMB_MOE_SCALE_DEV=1` | per-expert scale tensors stay in VRAM (removes two blocking copies per layer) | exact; turns itself off (one warning) when there is no expert cache, e.g. when the fit splits layers onto the CPU |
| `AMB_PF_OVERLAP=1`, `AMB_PF_OVERLAP_SLOTS=3` | prefill uploads whole expert tensors ahead of the layer on a second stream, into ring slots borrowed from the cache | exact |
| `AMB_INK_PF_BIAS=1`, `AMB_INK_PF_CTRL=1` | prefill: the dense sliding-window bias chain and the loop controllers as single kernels | exact |
| `AMB_INK_PF_MIX=1` | prefill: fused depth mix (the stock path fed a 2,048-token batch to a matrix-vector kernel) | KLD |
| `AMB_DSPARK_TRIM=1` | the drafter's recurrent chain stops at the draft length | exact |
| `AMB_DSPARK_IDS=1` | draft token ids come from the graph instead of logits plus host sampling | exact |
| `AMB_DSPARK_HEAD_WH=1` | the drafter's chain reads a precomputed 99 MiB product instead of the 492 MB output head per step | drafts identical |
| `AMB_DSPARK_ADAPT=1` | draft length 3, raised to `--spec-draft-n-max` (5) after 24 rounds at >= 85% acceptance, lowered again at <= 70% | exact (greedy verify) |

## Present but off (measured and rejected, or unfinished)

| Switch | Result |
|---|---|
| `AMB_MOE_GROUPED=1` | one warp computes every token that shares an expert: -10%, the GPU cache already served the repeated reads |
| `AMB_MOE_HIT_FIRST=1` | hit phase of each MoE layer launched before the ids are read: no gain, its check reports mismatches |
| `AMB_NKV_PAD=N` | pad the decode KV length to avoid graph re-captures: not exact, no measured gain |
| `AMB_INK_PF_SWA_REPACK=1` | re-pack the sliding-window ring before each prefill batch: works, +3% over `PF_BIAS`, costs ~0.8 GB of cache |
| `AMB_MOE_LOOKAHEAD`, `AMB_MOE_MISS_OVERLAP` | ported from qwen-next; refused together with loop dedup |
| `AMB_ELASTIC_FLOOR_MIB=N` | meant to let a 256k context fit with a lent cache floor; no effect yet (open item) |

## Debug and checks

`AMB_INK_FUSE_CHECK=1` compares fused against unfused results per site. `AMB_MMVF_SMALL_CHECK=N`, `AMB_DSPARK_IDS=2`,
`AMB_ELASTIC_CHECK=1`, `AMB_ELASTIC_POISON=1` (memory that changes owner is filled with markers; tokens must not change),
`AMB_MOE_STATS=N` (cache statistics every N decode batches), `AMB_INK_NODES=1` (graph node counts).
