#include "amb-publish.cuh"

#include "ggml-cuda.h"
#include "ggml-backend-impl.h" // ggml_backend::context

#include <cstring>

// AMB_MOE_MISS_OVERLAP (C23): GGML_OP_AMB_PUBLISH
//
// The kernel copies the selected (or predicted) expert ids of a layer into a slot of mapped pinned host memory and then
// publishes them with a flag, so that the host can read them while the stream keeps running the rest of the split graph
// (the "tail": predictor of the next layer, shared expert). The host polls the flag (llama-moe-cache.cpp, mo_wait).
//
// Why this is sound (and why not an event or a memcpy):
// - A kernel node is the only one of the three that behaves the same in eager execution and in a captured CUDA graph: its
//   parameters (the device addresses of the slot and of the counter) are fixed at capture and every replay executes it
//   once. A D2H memcpy plus an event record placed between two parts of one split could not be expressed as a ggml node,
//   and an event recorded after the graph launch would complete only at the end of the split.
// - Ordering against earlier work: the kernel is launched with a plain <<<>>> launch, without the programmatic stream
//   serialization attribute, so it starts only after the preceding node of the stream has completed. A PDL-launched
//   kernel may start before its predecessor completes; it cannot complete before its predecessor completes only if at
//   least one of its threads reaches cudaGridDependencySynchronize (ggml_cuda_pdl_sync), which waits for the completion
//   (and memory flush) of the predecessor. Under that condition completion is transitive along the stream: when this
//   kernel runs, every kernel enqueued before it (in particular the MUL_MAT_ID of the earlier layers, which read the
//   expert slots) has completed. The condition was audited for every PDL-launched kernel family of this model's decode
//   graphs (C24 report): each has a thread that reaches the sync for every non-empty launch (early returns are bounds
//   checks that block 0 / thread 0 passes). A future PDL kernel with a path that skips the sync in every thread would
//   break this argument; the MoE kernels of earlier layers are also upstream of the published ids by data dependency
//   (residual -> mixer -> router -> top-k), which a correct ggml graph orders anyway.
// - Visibility to the host: one thread writes the ids with volatile stores to the mapped allocation (cudaHostAllocMapped,
//   device address from cudaHostGetDevicePointer), then executes __threadfence_system(), which orders its earlier writes
//   before its later ones as observed by all threads including host threads, then stores the new counter value into the
//   flag word. A host thread that reads the new flag value (volatile load, then an acquire fence) therefore reads the new
//   ids. One thread on purpose: the ordering argument needs no inter-thread step (at most GGML_AMB_PUBLISH_MAX stores).
//   Measured on this machine (S0, tools/s0/amb_s0.cu, measurements/s0/s0.json, WDDM): a kernel write to such memory was
//   seen by the host in 3 us median, also while the stream stayed blocked by a later spinning kernel (round-trip test, 100 of
//   100 trials).
// - Sequence: the device counter (device memory, one per slot) makes every execution publish a new value (1, 2, 3, ...).
//   The host counts the launched publish nodes of each slot and waits for exactly that value, so a flag left over from an
//   earlier, unconsumed execution can never be taken for the current one.

static __global__ void k_amb_publish(const int32_t * __restrict__ src, const int n, int32_t * slot, uint32_t * ctr) {
    volatile int32_t * data = slot + GGML_AMB_PUBLISH_DATA;
    for (int i = 0; i < n; ++i) {
        data[i] = src[i];
    }
    const uint32_t v = *ctr + 1;
    *ctr = v;
    __threadfence_system(); // the ids before the flag, for every observer
    *((volatile uint32_t *) slot) = v;
    __threadfence_system();
}

bool ggml_cuda_amb_publish_supported(const ggml_tensor * op) {
    const ggml_tensor * a = op->src[0];
    return a != nullptr && a->type == GGML_TYPE_I32 && ggml_is_contiguous(a) && ggml_nelements(a) <= GGML_AMB_PUBLISH_MAX;
}

void ggml_cuda_op_amb_publish(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0];
    GGML_ASSERT(ggml_cuda_amb_publish_supported(dst));

    uint64_t s = 0, c = 0;
    memcpy(&s, &dst->op_params[0], sizeof(s));
    memcpy(&c, &dst->op_params[2], sizeof(c));
    int32_t * slot = (int32_t *) (uintptr_t) s;
    uint32_t * ctr = (uint32_t *) (uintptr_t) c;
    GGML_ASSERT(slot != nullptr && ctr != nullptr && "GGML_OP_AMB_PUBLISH built without its buffers (no_alloc graph executed?)");

    const int n = (int) ggml_nelements(a);
    // one thread, a plain launch (no PDL attribute), see above
    k_amb_publish<<<1, 1, 0, ctx.stream()>>>((const int32_t *) a->data, n, slot, ctr);
}

// AMB_MOE_MISS_OVERLAP=2 (C24): GGML_OP_AMB_GATE
//
// The miss phase of a layer must not read its slots before their upload on the upload stream has completed. The gate
// makes the compute stream wait (GPU side) on the layer's event, which the host records on the upload stream after the
// layer's miss uploads (llama-moe-cache.cpp, copy()), and then copies the miss-phase ids; the miss-phase MUL_MAT_IDs read
// that copy, so they run after the wait. Under stream capture, cudaStreamWaitEvent with cudaEventWaitExternal becomes an
// external event-wait node with this event (the event object is fixed per layer; every replay waits on it again).
// Which record a replayed wait node waits for: the host records the event before it launches the graph of the layer,
// and records it again for the same layer only in a later round, after it has seen a later publish of that layer, i.e.
// after this graph has executed; so the most recent record is the right one whether the node binds the record at
// graph launch or at node execution. A layer without misses records nothing: the wait finds the old, completed record.
// The copy kernel is a plain launch (no PDL attribute): it starts after the wait and after the hit phase before it.
static __global__ void k_amb_gate_copy(const int32_t * __restrict__ src, int32_t * __restrict__ dst, const int n,
        volatile unsigned long long * stamp) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = src[i];
    }
    if (stamp != nullptr && threadIdx.x == 0) {
        // the time from the stamp kernel (before the wait) to here, summed in [1], counted in [2]
        unsigned long long t1;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        stamp[1] = stamp[1] + (t1 - stamp[0]);
        stamp[2] = stamp[2] + 1;
        __threadfence_system();
    }
}

static __global__ void k_amb_gate_stamp(volatile unsigned long long * stamp) {
    unsigned long long t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    stamp[0] = t0;
}

void ggml_cuda_op_amb_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0];
    GGML_ASSERT(ggml_cuda_amb_publish_supported(dst) && ggml_is_contiguous(dst) && ggml_nelements(dst) == ggml_nelements(a));

    uint64_t e = 0;
    memcpy(&e, &dst->op_params[0], sizeof(e));
    ggml_backend_event_t ev = (ggml_backend_event_t) (uintptr_t) e;
    GGML_ASSERT(ev != nullptr && ev->context != nullptr && "GGML_OP_AMB_GATE built without its event (no_alloc graph executed?)");

    uint64_t t = 0;
    memcpy(&t, &dst->op_params[3], sizeof(t));
    volatile unsigned long long * stamp = (volatile unsigned long long *) (uintptr_t) t;
    if (stamp != nullptr) {
        k_amb_gate_stamp<<<1, 1, 0, ctx.stream()>>>(stamp);
    }
    // C24 fix: cudaEventWaitExternal is only legal while the stream is being captured (it makes the wait an external
    // event-wait node); in eager execution (warm-up, the first evaluations before ggml captures a graph, or CUDA graphs
    // off) the same flag returns cudaErrorIllegalState, so the plain wait is used there. Both wait for the event's most
    // recent record (eager: at the call; captured: see above)
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaError_t err = cudaStreamIsCapturing(ctx.stream(), &cap);
    const unsigned int flags = err == cudaSuccess && cap == cudaStreamCaptureStatusActive ? cudaEventWaitExternal : 0u;
    if (err == cudaSuccess) {
        err = cudaStreamWaitEvent(ctx.stream(), (cudaEvent_t) ev->context, flags);
    }
    if (err != cudaSuccess) {
        // printed directly (and flushed) so the reason survives an abort that the asynchronous logger may not flush
        fprintf(stderr, "amb_miss_overlap: GGML_OP_AMB_GATE layer %d (mode 2): %s(stream, event, flags=0x%x) failed: %s (%s); "
            "capture status %d\n", (int) dst->op_params[2], cap == cudaStreamCaptureStatusNone && flags == 0 && err != cudaSuccess ?
            "cudaStreamIsCapturing/cudaStreamWaitEvent" : "cudaStreamWaitEvent", flags, cudaGetErrorName(err), cudaGetErrorString(err),
            (int) cap);
        fflush(stderr);
        CUDA_CHECK(err);
    }
    const int n = (int) ggml_nelements(a);
    k_amb_gate_copy<<<1, 128, 0, ctx.stream()>>>((const int32_t *) a->data, (int32_t *) dst->data, n, stamp);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "amb_miss_overlap: GGML_OP_AMB_GATE layer %d (mode 2): the copy kernel launch failed: %s (%s)\n",
            (int) dst->op_params[2], cudaGetErrorName(err), cudaGetErrorString(err));
        fflush(stderr);
        CUDA_CHECK(err);
    }
}

bool ggml_backend_cuda_amb_publish_alloc(ggml_backend_t backend, size_t n_slots, void ** host, void ** slots_dev, void ** ctrs_dev) {
    static_assert(GGML_AMB_PUBLISH_SLOT >= (GGML_AMB_PUBLISH_DATA + GGML_AMB_PUBLISH_MAX)*sizeof(int32_t), "slot too small");
    if (backend == nullptr || !ggml_backend_is_cuda(backend) || n_slots == 0 || host == nullptr || slots_dev == nullptr || ctrs_dev == nullptr) {
        return false;
    }
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(cuda_ctx->device);

    void * h = nullptr;
    void * d = nullptr;
    void * c = nullptr;
    const size_t bytes = n_slots*GGML_AMB_PUBLISH_SLOT;
    bool ok = cudaHostAlloc(&h, bytes, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess;
    if (ok) {
        memset(h, 0, bytes);
        ok = cudaHostGetDevicePointer(&d, h, 0) == cudaSuccess && d != nullptr;
    }
    if (ok) {
        ok = cudaMalloc(&c, n_slots*sizeof(uint32_t)) == cudaSuccess;
    }
    if (ok) {
        ok = cudaMemset(c, 0, n_slots*sizeof(uint32_t)) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
    }
    if (!ok) {
        (void) cudaGetLastError();
        if (c != nullptr) {
            (void) cudaFree(c);
        }
        if (h != nullptr) {
            (void) cudaFreeHost(h);
        }
        (void) cudaGetLastError();
        return false;
    }
    *host      = h;
    *slots_dev = d;
    *ctrs_dev  = c;
    return true;
}

void ggml_backend_cuda_amb_publish_free(ggml_backend_t backend, void * host, void * ctrs_dev) {
    if (backend != nullptr && ggml_backend_is_cuda(backend)) {
        ggml_cuda_set_device(((ggml_backend_cuda_context *) backend->context)->device);
    }
    if (ctrs_dev != nullptr) {
        (void) cudaFree(ctrs_dev);
    }
    if (host != nullptr) {
        (void) cudaFreeHost(host);
    }
    (void) cudaGetLastError();
}
