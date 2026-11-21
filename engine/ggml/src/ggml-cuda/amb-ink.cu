// AMB_INK (C3): fused Inkling-2 decode ops, one ggml op (GGML_OP_AMB_INK) with a kind in op param 0.
//
// Exactness notes (the build uses -use_fast_math, so sqrtf, '/' and tanhf are the same approximate instructions as in
// the unfused kernels; every product or sum that the unfused graph stores to memory is written here with __fmul_rn,
// __fadd_rn or __fmaf_rn so that nvcc cannot contract it into an FMA the unfused graph does not have):
//   CLIP      bit-identical: same reduction as sum_rows (reduce_row_f32, same block size rule), then the same scalar ops
//   CONV      bit-identical: the tap loop of ssm_conv_f32, then the dynamic terms, the add, the scale and the residual add
//   MIX       NOT bit-identical: it replaces two mul_mats and a soft_max by one reduction per row (float sums reordered)
//   PAIR_RMS  bit-identical: the rms_norm kernel itself (norm.cu), with the second row read through the row stride
//   CTRL_TAIL bit-identical: tanh, scale_bias(-1, 1), mul, tanh, mul, add in the unfused order
//   SITU      bit-identical: scale, tanh, scale, sigmoid*mul (the unary-gated kernel), scale, tanh, scale, mul

#include "amb-ink.cuh"
#include "convert.cuh"
#include "norm.cuh"

// ---------------------------------------------------------------------------------------------------------------------
// CLIP: dst = y*clamp(t/(sqrt(sum_rows(y*y))*1 + 1e-6), 0, 1)[*tau], y = x[*scale]

static __global__ void amb_ink_clip_f32(const float * x, const float * scale, const float * tau, float * dst,
        const int ncols, const int rows_per_token, const float threshold) {
    const int64_t row = blockIdx.x;
    const int     col = threadIdx.x;

    const float * GGML_CUDA_RESTRICT xr = x + row*ncols;
    const float * GGML_CUDA_RESTRICT sr = scale ? scale + (row % rows_per_token)*ncols : nullptr;

    // reduce_row_f32 (reduce_rows.cuh) over the elements of sqr(mul(x, scale))
    float     sum        = 0.0f;
    const int num_unroll = 8;
    float     temp[num_unroll];
    float     sum_temp[num_unroll] = { 0.0f };

    for (int i = col; i < ncols;) {
        for (int j = 0; j < num_unroll; ++j) {
            if (i < ncols) {
                const float y = sr ? __fmul_rn(xr[i], sr[i]) : xr[i];
                temp[j] = __fmul_rn(y, y);
            } else {
                temp[j] = 0;
            }
            i += blockDim.x;
        }
        for (int j = 0; j < num_unroll; ++j) {
            sum_temp[j] += temp[j];
        }
    }
    for (int j = 0; j < num_unroll; ++j) {
        sum += sum_temp[j];
    }

    __shared__ float shared_vals[32];
    sum = block_reduce<block_reduce_method::SUM>(sum, shared_vals);

    // sqrt, scale_bias(1, 1e-6), fill(t), div, clamp(0, 1)
    float norm = sqrtf(sum);
    norm = __fmaf_rn(1.0f, norm, 1e-6f);
    float f = threshold / norm;
    f = fminf(fmaxf(f, 0.0f), 1.0f);

    const float tv = tau ? tau[row / rows_per_token] : 1.0f;

    float * GGML_CUDA_RESTRICT y = dst + row*ncols;
    for (int i = col; i < ncols; i += blockDim.x) {
        const float yi = sr ? __fmul_rn(xr[i], sr[i]) : xr[i];
        float v = __fmul_rn(yi, f);
        if (tau) {
            v = __fmul_rn(v, tv);
        }
        y[i] = v;
    }
}

static void amb_ink_clip(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x     = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * tau   = dst->src[2];

    const float threshold = ggml_get_op_params_f32(dst, 1);
    const int   ncols     = (int) x->ne[0];
    const int   nrows     = (int) ggml_nrows(x);

    // the block size rule of sum_rows_f32_cuda, the reduction tree depends on it
    const int  id  = ggml_cuda_get_device();
    const int  nsm = ggml_cuda_info().devices[id].nsm;
    const dim3 block_nums(nrows, 1, 1);
    const dim3 block_dims((nrows / nsm) < 2 ? 512 : (ncols < 1024 ? 32 : 128), 1, 1);

    amb_ink_clip_f32<<<block_nums, block_dims, 0, ctx.stream()>>>(
        (const float *) x->data, scale ? (const float *) scale->data : nullptr, tau ? (const float *) tau->data : nullptr,
        (float *) dst->data, ncols, (int) x->ne[1], threshold);
}

// ---------------------------------------------------------------------------------------------------------------------
// CONV: one thread per channel; reads the past inputs from state row ids[0], writes the newest to row cell_dst

#define AMB_INK_CONV_T_MAX 8

template <int K>
static __global__ void amb_ink_conv_f32(
        const float * cur, const int64_t cur_s1,
        const float * w,
        const float * dyn, const int64_t dyn_s1, const int dyn_hd,
        float * state, const int64_t state_s1, const int32_t * ids, const int64_t cell_dst,
        const float * res, const int64_t res_s1,
        float * dst, const int64_t dst_s1,
        const int C, const int T, const int64_t offset, const float scale,
        const int n_slots, const int64_t slot_stride) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= C) {
        return;
    }

    const float * s_src = state + (int64_t) ids[0]*state_s1 + (offset + c)*(K - 1);
    float       * s_dst = state + cell_dst*state_s1         + (offset + c)*(K - 1);

    // x = concat(past, cur^T): the window of channel c
    float xw[K - 1 + AMB_INK_CONV_T_MAX];
#pragma unroll
    for (int k = 0; k < K - 1; ++k) {
        xw[k] = s_src[k];
    }
    for (int t = 0; t < T; ++t) {
        xw[K - 1 + t] = cur[t*cur_s1 + c];
    }

    float wr[K];
#pragma unroll
    for (int j = 0; j < K; ++j) {
        wr[j] = w[(int64_t) c*K + j];
    }

    for (int t = 0; t < T; ++t) {
        // ssm_conv_f32: sumf += x*w per tap (an FMA chain), then the zero bias
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < K; ++j) {
            sumf = __fmaf_rn(xw[t + j], wr[j], sumf);
        }
        sumf = __fadd_rn(sumf, 0.0f);

        if (dyn) {
            // out = add(out, mul(window(lag), tap(lag))) for lag 0..K-1
            const float * tap = dyn + t*dyn_s1 + (int64_t) (c / dyn_hd)*K;
#pragma unroll
            for (int lag = 0; lag < K; ++lag) {
                sumf = __fadd_rn(sumf, __fmul_rn(xw[K - 1 - lag + t], tap[lag]));
            }
        }

        // scale(add(cur, out)): the scale kernel computes scale*x + bias with bias 0
        float v = __fmaf_rn(scale, __fadd_rn(xw[K - 1 + t], sumf), 0.0f);
        if (res) {
            v = __fadd_rn(res[t*res_s1 + c], v);
        }
        dst[t*dst_s1 + c] = v;
    }

    // the state keeps the last K - 1 columns of x; with rollback slots (build_conv_state_store, set_rows), slot s
    // (row cell_dst + s*slot_stride) keeps the window that ends s tokens before the last one
    for (int sl = 0; sl < n_slots; ++sl) {
        float * d = s_dst + sl*slot_stride*state_s1;
#pragma unroll
        for (int k = 0; k < K - 1; ++k) {
            d[k] = xw[T - sl + k];
        }
    }
}

static void amb_ink_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * cur      = dst->src[0];
    const ggml_tensor * weight   = dst->src[1];
    const ggml_tensor * dynamic  = dst->src[2];
    const ggml_tensor * state    = dst->src[3];
    const ggml_tensor * ids      = dst->src[4];
    const ggml_tensor * residual = dst->src[5];

    const int64_t offset   = ggml_get_op_params_i32(dst, 1);
    const int64_t cell_dst = ggml_get_op_params_i32(dst, 2);
    const float   scale    = ggml_get_op_params_f32(dst, 3);
    const int     n_slots  = ggml_get_op_params_i32(dst, 4);
    const int64_t s_stride = ggml_get_op_params_i32(dst, 5);

    const int C = (int) cur->ne[0];
    const int T = (int) cur->ne[1];
    const int K = (int) weight->ne[0];
    GGML_ASSERT(T <= AMB_INK_CONV_T_MAX);

    const int dyn_hd = dynamic ? (int) (C / (dynamic->ne[0] / K)) : 1;

    const int  threads = 128;
    const dim3 blocks((C + threads - 1) / threads, 1, 1);

#define AMB_INK_CONV_LAUNCH(KK) \
    amb_ink_conv_f32<KK><<<blocks, threads, 0, ctx.stream()>>>( \
        (const float *) cur->data, cur->nb[1]/sizeof(float), \
        (const float *) weight->data, \
        dynamic ? (const float *) dynamic->data : nullptr, dynamic ? dynamic->nb[1]/sizeof(float) : 0, dyn_hd, \
        (float *) state->data, state->nb[1]/sizeof(float), (const int32_t *) ids->data, cell_dst, \
        residual ? (const float *) residual->data : nullptr, residual ? residual->nb[1]/sizeof(float) : 0, \
        (float *) dst->data, dst->nb[1]/sizeof(float), C, T, offset, scale, n_slots, s_stride)

    switch (K) {
        case 2: AMB_INK_CONV_LAUNCH(2); break;
        case 3: AMB_INK_CONV_LAUNCH(3); break;
        case 4: AMB_INK_CONV_LAUNCH(4); break;
        case 5: AMB_INK_CONV_LAUNCH(5); break;
        case 6: AMB_INK_CONV_LAUNCH(6); break;
        case 7: AMB_INK_CONV_LAUNCH(7); break;
        case 8: AMB_INK_CONV_LAUNCH(8); break;
        default: GGML_ABORT("amb_ink_conv: unsupported kernel size %d", K);
    }
#undef AMB_INK_CONV_LAUNCH
}

// ---------------------------------------------------------------------------------------------------------------------
// MIX: one block per (query, token)

#define AMB_INK_MIX_ROWS_MAX GGML_MAX_SRC

struct amb_ink_mix_args {
    const float * rows[AMB_INK_MIX_ROWS_MAX]; // row pointers at token 0, in the concat order
    int64_t       rows_s1[AMB_INK_MIX_ROWS_MAX];
    const float * query;
    int64_t       query_s1;
    float       * dst;
    int64_t       dst_s1;
    int64_t       dst_s2;
    int           n;
    int           nr;
    float         eps;
    float         kq_scale;
};

template <int block_size>
static __global__ void amb_ink_mix_f32(const amb_ink_mix_args a) {
    const int q   = blockIdx.x;
    const int t   = blockIdx.y;
    const int tid = threadIdx.x;

    const float * GGML_CUDA_RESTRICT qr = a.query + q*a.query_s1;

    __shared__ float sh[2*AMB_INK_MIX_ROWS_MAX][32];
    __shared__ float wsh[AMB_INK_MIX_ROWS_MAX];

    float ss[AMB_INK_MIX_ROWS_MAX];
    float dot[AMB_INK_MIX_ROWS_MAX];

#pragma unroll
    for (int j = 0; j < AMB_INK_MIX_ROWS_MAX; ++j) {
        ss[j]  = 0.0f;
        dot[j] = 0.0f;
        if (j < a.nr) {
            const float * GGML_CUDA_RESTRICT xr = a.rows[j] + t*a.rows_s1[j];
            for (int i = tid; i < a.n; i += block_size) {
                const float xi = xr[i];
                ss[j]  += xi*xi;
                dot[j] += xi*qr[i];
            }
        }
    }

#pragma unroll
    for (int j = 0; j < AMB_INK_MIX_ROWS_MAX; ++j) {
        if (j < a.nr) {
            ss[j]  = block_reduce<block_reduce_method::SUM, block_size>(ss[j],  sh[2*j + 0]);
            dot[j] = block_reduce<block_reduce_method::SUM, block_size>(dot[j], sh[2*j + 1]);
        }
    }

    if (tid == 0) {
        float s[AMB_INK_MIX_ROWS_MAX];
        float p[AMB_INK_MIX_ROWS_MAX];
        float mx = -INFINITY;
        for (int j = 0; j < a.nr; ++j) {
            s[j] = rsqrtf(ss[j] / a.n + a.eps);
            p[j] = dot[j]*s[j]*a.kq_scale;
            mx   = fmaxf(mx, p[j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < a.nr; ++j) {
            p[j] = expf(p[j] - mx);
            sum += p[j];
        }
        for (int j = 0; j < a.nr; ++j) {
            wsh[j] = s[j]*(p[j]/sum);
        }
    }
    __syncthreads();

    float * GGML_CUDA_RESTRICT y = a.dst + q*a.dst_s1 + t*a.dst_s2;
    for (int i = tid; i < a.n; i += block_size) {
        float acc = 0.0f;
        for (int j = 0; j < a.nr; ++j) {
            acc += a.rows[j][t*a.rows_s1[j] + i]*wsh[j];
        }
        y[i] = acc;
    }
}

static void amb_ink_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * cur   = dst->src[0];
    const ggml_tensor * query = dst->src[1];
    const int n_blocks = ggml_get_op_params_i32(dst, 1);

    amb_ink_mix_args a = {};
    a.nr = n_blocks + 1;
    GGML_ASSERT(a.nr <= AMB_INK_MIX_ROWS_MAX);
    for (int j = 0; j < a.nr; ++j) {
        const ggml_tensor * r = j < n_blocks ? dst->src[2 + n_blocks - 1 - j] : cur;
        a.rows[j]    = (const float *) r->data;
        a.rows_s1[j] = r->nb[1]/sizeof(float);
    }
    a.query    = (const float *) query->data;
    a.query_s1 = query->nb[1]/sizeof(float);
    a.dst      = (float *) dst->data;
    // [n, nq, T], or [n, T, nq] (tq): the kernel indexes y = dst + q*dst_s1 + t*dst_s2
    const bool tq = ggml_get_op_params_i32(dst, 3) != 0;
    a.dst_s1   = (tq ? dst->nb[2] : dst->nb[1])/sizeof(float);
    a.dst_s2   = (tq ? dst->nb[1] : dst->nb[2])/sizeof(float);
    a.n        = (int) cur->ne[0];
    a.eps      = ggml_get_op_params_f32(dst, 2);
    a.kq_scale = 1.0f/sqrtf((float) a.n);

    const dim3 blocks(query->ne[1], cur->ne[1], 1);
    amb_ink_mix_f32<256><<<blocks, 256, 0, ctx.stream()>>>(a);
}

// ---------------------------------------------------------------------------------------------------------------------
// CTRL_TAIL: (-tanh(f) + 1)*cur + tanh(g)*msg, each step rounded as in its own kernel

static __global__ void amb_ink_ctrl_tail_f32(
        const float * f, const int64_t f_s1, const float * g, const int64_t g_s1,
        const float * cur, const int64_t cur_s1, const float * msg, const int64_t msg_s1,
        float * dst, const int64_t dst_s1, const int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= n) {
        return;
    }
    const float forget = tanhf(f[t*f_s1 + i]);
    const float gain   = tanhf(g[t*g_s1 + i]);
    const float keep   = __fmaf_rn(-1.0f, forget, 1.0f); // scale_bias: scale*x + bias
    const float a      = __fmul_rn(keep, cur[t*cur_s1 + i]);
    const float b      = __fmul_rn(gain, msg[t*msg_s1 + i]);
    dst[t*dst_s1 + i]  = __fadd_rn(a, b);
}

static void amb_ink_ctrl_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * f   = dst->src[0];
    const ggml_tensor * g   = dst->src[1];
    const ggml_tensor * cur = dst->src[2];
    const ggml_tensor * msg = dst->src[3];
    const int n = (int) dst->ne[0];
    const int T = (int) dst->ne[1];
    const dim3 blocks((n + 255)/256, T, 1);
    amb_ink_ctrl_tail_f32<<<blocks, 256, 0, ctx.stream()>>>(
        (const float *) f->data,   f->nb[1]/sizeof(float),   (const float *) g->data,   g->nb[1]/sizeof(float),
        (const float *) cur->data, cur->nb[1]/sizeof(float), (const float *) msg->data, msg->nb[1]/sizeof(float),
        (float *) dst->data, dst->nb[1]/sizeof(float), n);
}

// ---------------------------------------------------------------------------------------------------------------------
// SITU: beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb), every step rounded as in the unfused kernels

static __global__ void amb_ink_situ_f32(
        const float * gate, const int64_t g_s1, const int64_t g_s2, const float * up, const int64_t u_s1, const int64_t u_s2,
        float * dst, const int n, const int n1, const int64_t k,
        const float inv_beta, const float beta, const float inv_lb, const float lb, const bool use_lb) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    const int64_t i0 = i % n;
    const int64_t r  = i / n;
    const int64_t i1 = r % n1;
    const int64_t i2 = r / n1;

    const float g = gate[i2*g_s2 + i1*g_s1 + i0];
    const float u = up  [i2*u_s2 + i1*u_s1 + i0];

    // scale(tanh(scale(g, 1/beta)), beta): the scale kernel is scale*x + bias with bias 0
    const float act = __fmaf_rn(beta, tanhf(__fmaf_rn(inv_beta, g, 0.0f)), 0.0f);
    // mul(act, sigmoid(g)): the fused unary-gated kernel, op_sigmoid(x)*other
    const float sig = 1.0f / (1.0f + expf(-g));
    const float a   = __fmul_rn(sig, act);
    const float uu  = use_lb ? __fmaf_rn(lb, tanhf(__fmaf_rn(inv_lb, u, 0.0f)), 0.0f) : u;
    dst[i] = __fmul_rn(a, uu);
}

static void amb_ink_situ(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * gate = dst->src[0];
    const ggml_tensor * up   = dst->src[1];
    const float beta = ggml_get_op_params_f32(dst, 1);
    const float lb   = ggml_get_op_params_f32(dst, 2);
    const int64_t k = ggml_nelements(dst);
    const int64_t nb = (k + 255)/256;
    amb_ink_situ_f32<<<(unsigned) nb, 256, 0, ctx.stream()>>>(
        (const float *) gate->data, gate->nb[1]/sizeof(float), gate->nb[2]/sizeof(float),
        (const float *) up->data,   up->nb[1]/sizeof(float),   up->nb[2]/sizeof(float),
        (float *) dst->data, (int) dst->ne[0], (int) dst->ne[1], k,
        1.0f/beta, beta, lb > 0.0f ? 1.0f/lb : 0.0f, lb, lb > 0.0f);
}

// ---------------------------------------------------------------------------------------------------------------------
// REL_MASK (AMB_INK_PF_BIAS, P1): dst[kv, t, h] = f16((table[rel[kv, t], h, t] - sinks[h, t]) + mask[kv, t])
// bit-identical to sub (bin_bcast) -> cont(permute) -> get_rows -> cont(permute) -> cast(mask, F32) -> add -> cast F16:
// the same two float ops per element, each rounded on its own, and the same float -> half conversion as cpy.cu.
// One thread per (kv, t), the heads in the loop: rel and mask are read once, the writes of a warp are 64-byte runs.

template <typename mask_t>
static __global__ void amb_ink_rel_mask_f16(
        const float * table, const float * sinks,
        const int32_t * rel, const mask_t * mask, half * dst,
        const int64_t n_kv, const int64_t T, const int H, const int64_t E) {
    const int64_t i = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= n_kv*T) {
        return;
    }
    const int64_t t  = i / n_kv;
    const int64_t kv = i - t*n_kv;

    const float   m = ggml_cuda_cast<float>(mask[i]);
    const int64_t r = rel[i];

    const float * GGML_CUDA_RESTRICT tb = table + t*H*E + r;
    const float * GGML_CUDA_RESTRICT sk = sinks ? sinks + t*H : nullptr;
    half        * GGML_CUDA_RESTRICT y  = dst + t*n_kv + kv;
    const int64_t y_s = T*n_kv;

    for (int h = 0; h < H; ++h) {
        float v = tb[h*E];
        if (sk) {
            v = __fsub_rn(v, sk[h]);
        }
        y[h*y_s] = ggml_cuda_cast<half>(__fadd_rn(v, m));
    }
}

static void amb_ink_rel_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * table = dst->src[0];
    const ggml_tensor * sinks = dst->src[1];
    const ggml_tensor * rel   = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int64_t E    = table->ne[0];
    const int     H    = (int) table->ne[1];
    const int64_t T    = table->ne[2];
    const int64_t n_kv = rel->ne[0];

    const int64_t nb = (n_kv*T + 255)/256;
    GGML_ASSERT(nb <= INT32_MAX);
    if (mask->type == GGML_TYPE_F16) {
        amb_ink_rel_mask_f16<half><<<(unsigned) nb, 256, 0, ctx.stream()>>>(
            (const float *) table->data, sinks ? (const float *) sinks->data : nullptr, (const int32_t *) rel->data,
            (const half *) mask->data, (half *) dst->data, n_kv, T, H, E);
    } else {
        amb_ink_rel_mask_f16<float><<<(unsigned) nb, 256, 0, ctx.stream()>>>(
            (const float *) table->data, sinks ? (const float *) sinks->data : nullptr, (const int32_t *) rel->data,
            (const float *) mask->data, (half *) dst->data, n_kv, T, H, E);
    }
}

// ---------------------------------------------------------------------------------------------------------------------

bool ggml_cuda_amb_ink_supported(const ggml_tensor * op) {
    if ((ggml_amb_ink_kind) ggml_get_op_params_i32(op, 0) == GGML_AMB_INK_REL_MASK) {
        // the mask may be F16
        return op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F32 && op->src[2]->type == GGML_TYPE_I32 &&
               (!op->src[1] || op->src[1]->type == GGML_TYPE_F32) &&
               (op->src[3]->type == GGML_TYPE_F16 || op->src[3]->type == GGML_TYPE_F32) &&
               (op->src[2]->ne[0]*op->src[2]->ne[1] + 255)/256 <= INT32_MAX;
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const ggml_tensor * s = op->src[i];
        if (s && s->type != GGML_TYPE_F32 && s->type != GGML_TYPE_I32) {
            return false;
        }
    }
    switch ((ggml_amb_ink_kind) ggml_get_op_params_i32(op, 0)) {
        case GGML_AMB_INK_CLIP:
            return ggml_is_contiguous(op->src[0]) && ggml_nrows(op->src[0]) <= INT32_MAX;
        case GGML_AMB_INK_CONV:
            return op->src[1]->ne[0] >= 2 && op->src[1]->ne[0] <= 8 && op->src[0]->ne[1] <= AMB_INK_CONV_T_MAX &&
                   op->src[3]->buffer && !ggml_backend_buffer_is_host(op->src[3]->buffer);
        case GGML_AMB_INK_MIX:
            return ggml_get_op_params_i32(op, 1) + 1 <= AMB_INK_MIX_ROWS_MAX;
        case GGML_AMB_INK_PAIR_RMS:
        case GGML_AMB_INK_CTRL_TAIL:
            return true;
        case GGML_AMB_INK_SITU:
            return ggml_nelements(op) <= (int64_t) UINT32_MAX*256;
        default:
            return false;
    }
}

void ggml_cuda_op_amb_ink(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch ((ggml_amb_ink_kind) ggml_get_op_params_i32(dst, 0)) {
        case GGML_AMB_INK_CLIP:      amb_ink_clip(ctx, dst);                    break;
        case GGML_AMB_INK_CONV:      amb_ink_conv(ctx, dst);                    break;
        case GGML_AMB_INK_MIX:       amb_ink_mix(ctx, dst);                     break;
        case GGML_AMB_INK_PAIR_RMS:  ggml_cuda_op_amb_ink_pair_rms(ctx, dst);   break;
        case GGML_AMB_INK_CTRL_TAIL: amb_ink_ctrl_tail(ctx, dst);               break;
        case GGML_AMB_INK_SITU:      amb_ink_situ(ctx, dst);                    break;
        case GGML_AMB_INK_REL_MASK:  amb_ink_rel_mask(ctx, dst);                break;
        default: GGML_ABORT("unknown amb_ink kind");
    }
}
