#pragma once

#include "common.cuh"

// AMB_MOE_MISS_OVERLAP (C23): GGML_OP_AMB_PUBLISH, see ggml.h and amb-publish.cu
void ggml_cuda_op_amb_publish(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_amb_publish_supported(const ggml_tensor * op);

// AMB_MOE_MISS_OVERLAP=2 (C24): GGML_OP_AMB_GATE, see ggml.h and amb-publish.cu (same support rule as the publish)
void ggml_cuda_op_amb_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// proc addresses "ggml_backend_cuda_amb_publish_alloc" / "ggml_backend_cuda_amb_publish_free" of the CUDA registry:
// n_slots slots of GGML_AMB_PUBLISH_SLOT bytes (ggml.h) in mapped pinned host memory (zeroed; *host and *slots_dev are the host
// and device addresses of slot 0) and n_slots uint32 counters in device memory (zeroed), on the device of backend.
// false on any failure (nothing allocated, error cleared)
bool ggml_backend_cuda_amb_publish_alloc(ggml_backend_t backend, size_t n_slots, void ** host, void ** slots_dev, void ** ctrs_dev);
void ggml_backend_cuda_amb_publish_free(ggml_backend_t backend, void * host, void * ctrs_dev);
