// Premise micro-benchmark for docs/decode-math-design.md. Standalone, no engine code.
// Build (x64 Native Tools prompt):  nvcc -O3 -arch=sm_120 membench.cu -o membench.exe
// Run:                              membench.exe            (about 10 s, prints one table per test)
// It measures memory-read patterns only (integer sums, no dot products), so it answers "is this launch geometry
// bandwidth-bound or occupancy/latency-bound", not "how fast is the real kernel".
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

// Test A: nblocks blocks of nthreads threads stream a buffer of n_words uint32, each block one contiguous share.
__global__ void stream_read(const uint32_t * buf, uint64_t n_words, uint32_t * out) {
    const uint64_t per_block = n_words / gridDim.x;
    const uint32_t * p = buf + per_block * blockIdx.x;
    uint32_t acc = 0;
    for (uint64_t i = threadIdx.x; i < per_block; i += blockDim.x) {
        acc += p[i];
    }
    if (acc == 0xFFFFFFFFu) { out[blockIdx.x] = acc; } // keeps the loads alive, practically never taken
}

// Tests B and C: a matvec-shaped read. Matrix of n_rows rows of row_words uint32 per "expert"; grid.x = n_rows/rpb,
// grid.y = channels (routes per token), block = (32, n_tok): one warp per token reads rpb rows of the expert that
// ids[channel + token*channels] selects. Same geometry as mul_mat_vec_q_moe (rpb = 2 there).
__global__ void rows_read(const uint32_t * w, const int * ids, int n_rows, int row_words, int rpb, int channels,
        uint64_t expert_words, uint32_t * out) {
    const int tok = threadIdx.y;
    const int e   = ids[blockIdx.y + tok * channels];
    const int row0 = blockIdx.x * rpb;
    uint32_t acc = 0;
    for (int r = 0; r < rpb; ++r) {
        const int row = row0 + r;
        if (row >= n_rows) { break; }
        const uint32_t * p = w + (uint64_t) e * expert_words + (uint64_t) row * row_words;
        for (int i = threadIdx.x; i < row_words; i += 32) {
            acc += p[i];
        }
    }
    if (acc == 0xFFFFFFFFu) { out[0] = acc; }
}

static double time_us(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, a, b));
    return ms * 1000.0;
}

int main() {
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    uint32_t * out = nullptr;
    CK(cudaMalloc(&out, 1 << 20));
    const int reps = 200;

    // ---- Test A: flash attention premise. 188 MB (K + V + mask of one global layer at 43,776 cells).
    {
        const uint64_t bytes = 188ull << 20;
        uint32_t * buf = nullptr;
        CK(cudaMalloc(&buf, bytes));
        CK(cudaMemset(buf, 1, bytes));
        printf("Test A: streaming read of %llu MiB, 128 threads per block\n", (unsigned long long) (bytes >> 20));
        printf("  blocks   us/pass   GB/s\n");
        const int grids[] = { 120, 240, 480, 960, 1920, 7680 };
        for (int g : grids) {
            for (int i = 0; i < 20; ++i) { stream_read<<<g, 128>>>(buf, bytes / 4, out); }
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(e0));
            for (int i = 0; i < reps; ++i) { stream_read<<<g, 128>>>(buf, bytes / 4, out); }
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            const double us = time_us(e0, e1) / reps;
            printf("  %6d  %8.1f  %6.0f\n", g, us, bytes / us / 1000.0);
        }
        CK(cudaFree(buf));
    }

    // ---- Tests B and C share the row reader.
    struct shape { const char * name; int n_expert; int n_rows; int row_bytes; int channels; int n_tok; int distinct; };
    const shape shapes[] = {
        // NVFP4 down projection: 2304 rows x 1152 weights x 4.5 bit = 648 B per row, 8 routes x 4 tokens, 20 distinct
        { "B  expert down   (NVFP4, 2304 x 648 B, 32 routes, 20 distinct)", 128, 2304,  648, 8, 4, 20 },
        // NVFP4 gate_up: 2304 rows x 2304 weights = 1296 B per row
        { "B' expert gate_up (NVFP4, 2304 x 1296 B, 32 routes, 20 distinct)", 128, 2304, 1296, 8, 4, 20 },
        // Q8_0 output head, one column: 201,024 rows x 2304 x 8.5 bit = 2448 B per row (one pass today: ~619 us)
        { "C0 output head Q8_0 (201024 x 2448 B, 1 column)", 1, 201024, 2448, 1, 1, 1 },
        // proposed draft head: 201,024 rows x 256 F16 = 512 B per row
        { "C1 draft head W*H F16 (201024 x 512 B, 1 column)", 1, 201024,  512, 1, 1, 1 },
    };
    for (const shape & s : shapes) {
        const int row_words = s.row_bytes / 4;
        const uint64_t expert_words = (uint64_t) s.n_rows * row_words;
        const uint64_t bytes = expert_words * 4 * s.n_expert;
        uint32_t * w = nullptr;
        CK(cudaMalloc(&w, bytes));
        CK(cudaMemset(w, 1, bytes));
        std::vector<int> ids_h((size_t) s.channels * s.n_tok);
        for (size_t i = 0; i < ids_h.size(); ++i) {
            // `distinct` different experts, spread over the bank
            ids_h[i] = (int) ((i % (size_t) s.distinct) * (size_t) (s.n_expert / s.distinct > 0 ? s.n_expert / s.distinct : 1)) % s.n_expert;
        }
        int * ids = nullptr;
        CK(cudaMalloc(&ids, ids_h.size() * sizeof(int)));
        CK(cudaMemcpy(ids, ids_h.data(), ids_h.size() * sizeof(int), cudaMemcpyHostToDevice));
        const double distinct_mb = (double) expert_words * 4 * s.distinct / 1e6;
        printf("\nTest %s: %.1f MB distinct per pass\n", s.name, distinct_mb);
        printf("  rows/block   blocks   us/pass   GB/s (distinct bytes)\n");
        const int rpbs[] = { 1, 2, 4, 8, 16, 32 };
        for (int rpb : rpbs) {
            const dim3 grid((s.n_rows + rpb - 1) / rpb, s.channels);
            const dim3 block(32, s.n_tok);
            for (int i = 0; i < 20; ++i) { rows_read<<<grid, block>>>(w, ids, s.n_rows, row_words, rpb, s.channels, expert_words, out); }
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(e0));
            for (int i = 0; i < reps; ++i) { rows_read<<<grid, block>>>(w, ids, s.n_rows, row_words, rpb, s.channels, expert_words, out); }
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            const double us = time_us(e0, e1) / reps;
            printf("  %10d  %7d  %8.1f  %6.0f\n", rpb, (int) (grid.x * grid.y), us, distinct_mb * 1000.0 / us);
        }
        CK(cudaFree(w));
        CK(cudaFree(ids));
    }
    printf("\nReference: VRAM streaming probe on this laptop 858 GB/s.\n");
    return 0;
}
