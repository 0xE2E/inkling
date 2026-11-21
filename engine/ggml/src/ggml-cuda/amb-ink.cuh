#pragma once

#include "common.cuh"

// AMB_INK (C3): fused Inkling-2 decode ops (GGML_OP_AMB_INK), see ggml.h enum ggml_amb_ink_kind
bool ggml_cuda_amb_ink_supported(const ggml_tensor * op);
void ggml_cuda_op_amb_ink(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
