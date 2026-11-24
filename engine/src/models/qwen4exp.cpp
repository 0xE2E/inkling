#include "models.h"
#include "llama-impl.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"
#include "llama-moe-cache.h"
#include "llama-ext.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstring>
#include <mutex>
#include <string>

// AMB_MTP_SHARE_HEAD (C26, default off): a separate MTP draft file (-md, MTP-only GGUF) carries its own copies of
// token_embd.weight and output.weight, which are the target's matrices at another precision. With the switch, the draft
// model does not create them (the loader counts them as unused, nothing is allocated or uploaded) and its MTP graph reads
// the target's tensors through cparams.model_tgt (graph_mtp below, llama_context: the existing path for a draft head
// without token_embd/output). 1 = share output.weight and token_embd.weight, 2 = share output.weight only; else off.
// The draft file cannot see its target at load time, so the target (the last trunk qwen4exp file loaded in this process,
// recorded below) is what the shapes and the placement are checked against; the draft context checks the real target
// again (llama_context, cparams.model_tgt) and refuses to start on a mismatch.
int amb_mtp_share_head_env() {
    static const int v = [] {
        const char * s = getenv("AMB_MTP_SHARE_HEAD");
        if (s == nullptr) {
            return 0;
        }
        if (strcmp(s, "1") == 0) {
            return 1;
        }
        if (strcmp(s, "2") == 0) {
            return 2;
        }
        return 0;
    }();
    return v;
}

// C26b: the --fit pass turns sharing off for the rest of the process when the sharing draft cannot be measured
static std::atomic<bool> g_amb_share_head_suppressed{false};

int amb_mtp_share_head_active() {
    return g_amb_share_head_suppressed.load() ? 0 : amb_mtp_share_head_env();
}

int32_t llama_amb_mtp_share_head_active(void) {
    return amb_mtp_share_head_active();
}

void llama_amb_mtp_share_head_suppress(void) {
    g_amb_share_head_suppressed.store(true);
}

namespace {
// C26b: the record names the trunk model it describes (owner); it is dropped when that model is freed
struct amb_head_record {
    bool                valid = false;
    const llama_model * owner = nullptr;
    int64_t     out_ne0 = 0, out_ne1 = 0, emb_ne0 = 0, emb_ne1 = 0;
    int64_t     n_vocab = 0;
    ggml_type   out_type = GGML_TYPE_COUNT, emb_type = GGML_TYPE_COUNT;
    bool        out_host = true;
    std::string out_buft, emb_buft;
};
std::mutex        g_amb_head_mutex;
amb_head_record g_amb_head;
} // namespace

void amb_mtp_share_head_forget(const llama_model * model) {
    std::lock_guard<std::mutex> lock(g_amb_head_mutex);
    if (g_amb_head.valid && g_amb_head.owner == model) {
        g_amb_head = amb_head_record();
    }
}

// C26b: device memory only: a buffer type that is host memory, or whose device is not a GPU (CPU, CPU repack/extra
// types, which can report no is_host), counts as host
static bool amb_buft_is_host_like(ggml_backend_buffer_type_t buft) {
    if (buft == nullptr || ggml_backend_buft_is_host(buft)) {
        return true;
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr) {
        return true;
    }
    const enum ggml_backend_dev_type t = ggml_backend_dev_type(dev);
    return t != GGML_BACKEND_DEVICE_TYPE_GPU && t != GGML_BACKEND_DEVICE_TYPE_IGPU;
}

// the buffer type a created tensor will be allocated in (the loader context it was created in), nullptr if not found
static ggml_backend_buffer_type_t amb_created_buft(const llama_model_loader & ml, const ggml_tensor * t) {
    if (t == nullptr) {
        return nullptr;
    }
    for (const auto & [key, ctx] : ml.ctx_map) {
        if (ggml_get_tensor(ctx.get(), ggml_get_name(t)) == t) {
            return key.buft;
        }
    }
    return nullptr;
}

// bad metadata must be catchable: GGML_ASSERT aborts the whole process
static void qwen4exp_require_nonzero(const llama_model_loader & ml, llm_kv kid, uint32_t value) {
    if (value == 0) {
        throw std::runtime_error(format("%s must be greater than zero, got %u", ml.llm_kv(kid).c_str(), value));
    }
}

// get_arr() copies a short array as-is, leaving a zero tail the n-gram hash silently drops
static void qwen4exp_require_arr_len(llama_model_loader & ml, llm_kv kid, uint32_t n_min) {
    uint32_t n_arr = 0;
    ml.get_arr_n(kid, n_arr, true);
    if (n_arr < n_min) {
        throw std::runtime_error(format("%s has %u entries, but at least %u are required",
                                        ml.llm_kv(kid).c_str(), n_arr, n_min));
    }
}

void llama_model_qwen4exp::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // HC; low_rank is qwen4exp-specific, DeepSeek-V4 leaves it absent (full rank)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,    hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    // a count of 1 has nothing to mix: transformers configuration_qwen4_exp.py:196, vLLM
    // config.py:49 and SGLang configs/qwen4_exp.py:38 all raise on hc_count <= 1
    if (hparams.dsv4_hc_mult <= 1) {
        throw std::runtime_error(format("%s must be greater than one, got %u",
                                        ml.llm_kv(LLM_KV_HYPER_CONNECTION_COUNT).c_str(), hparams.dsv4_hc_mult));
    }
    qwen4exp_require_nonzero(ml, LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios, hparams.n_layer_all, false);

    // QSA pools the indexer keys of blocks of compress_ratio cells, one block size for the whole model
    hparams.indexer_kpool = 0;
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        const uint32_t r = hparams.dsv4_compress_ratios[il];
        if (r == 0) {
            continue;
        }
        if (hparams.indexer_kpool != 0 && r != hparams.indexer_kpool) {
            throw std::runtime_error(format("QSA layers must share one compress ratio, got %u and %u", hparams.indexer_kpool, r));
        }
        hparams.indexer_kpool = r;
    }
    if (hparams.indexer_kpool == 1 || (hparams.indexer_kpool > 0 && hparams.indexer_top_k % hparams.indexer_kpool != 0)) {
        throw std::runtime_error(format("QSA needs a compress ratio above 1 that divides the budget, got %u and %u",
                                        hparams.indexer_kpool, hparams.indexer_top_k));
    }
    // the reference groups the visible tokens in cache order and always keeps the tail
    hparams.indexer_kpool_row         = 2; // raw key | pooled key
    hparams.indexer_kpool_by_order    = true;
    hparams.indexer_kpool_select_tail = true;

    // PLE n-gram hash embeddings; if the key group is absent every field stays zero
    hparams.is_ple_impl.reset();
    hparams.ple_n_heads = 0;

    uint32_t n_ple = 0;
    ml.get_arr_n(LLM_KV_PLE_LAYERS, n_ple, false);
    if (n_ple > 0) {
        std::vector<uint32_t> ple_layers;
        ml.get_arr(LLM_KV_PLE_LAYERS, ple_layers);
        if (n_ple != 1) {
            // hparams holds one set of hash constants, so several PLE modules cannot be represented
            throw std::runtime_error(format("%s lists %u layers, but only one PLE layer is supported",
                                            ml.llm_kv(LLM_KV_PLE_LAYERS).c_str(), n_ple));
        }
        for (uint32_t il : ple_layers) {
            if (il >= hparams.n_layer_all) {
                throw std::runtime_error(format("PLE layer %u is out of range", il));
            }
            hparams.is_ple_impl.set(il);
        }

        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,      hparams.ple_ngram_size);
        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM, hparams.ple_heads_per_ngram);
        ml.get_key(LLM_KV_PLE_CONV_KERNEL,     hparams.ple_conv_kernel);
        ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID,    hparams.ple_eos_token_id);
        // optional: files written before this key fall back to the EOS token
        ml.get_key(LLM_KV_PLE_IMAGE_TOKEN_ID,  hparams.ple_image_token_id, false);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.n_embd_per_layer);
        qwen4exp_require_nonzero(ml, LLM_KV_PLE_CONV_KERNEL,             hparams.ple_conv_kernel);
        qwen4exp_require_nonzero(ml, LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);

        hparams.ple_n_heads  = (hparams.ple_ngram_size - 1) * hparams.ple_heads_per_ngram;
        hparams.ple_head_dim = hparams.n_embd_per_layer;
        hparams.ple_embd_token = hparams.ple_image_token_id != 0
            ? (llama_token) hparams.ple_image_token_id
            : (llama_token) hparams.ple_eos_token_id;
        if (hparams.ple_ngram_size < 2 || hparams.ple_ngram_size > LLAMA_MAX_PLE_NGRAM) {
            throw std::runtime_error(format("PLE n-gram size %u is out of range", hparams.ple_ngram_size));
        }
        if (hparams.ple_n_heads == 0 || hparams.ple_n_heads > LLAMA_MAX_PLE_HEADS) {
            throw std::runtime_error(format("PLE head count %u is out of range", hparams.ple_n_heads));
        }

        qwen4exp_require_arr_len(ml, LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_ngram_size);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_OFFSETS,      hparams.ple_n_heads);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_VOCAB_SIZES,  hparams.ple_n_heads);

        ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_layer_multipliers);

        // the file stores the head ranges as uint64, so read at that width and narrow to the int32 the gather uses
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_offsets     = {};
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_vocab_sizes = {};
        ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,     head_offsets);
        ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES, head_vocab_sizes);
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            if (head_vocab_sizes[h] == 0 ||
                head_offsets[h]     > INT32_MAX ||
                head_vocab_sizes[h] > INT32_MAX ||
                head_offsets[h] + head_vocab_sizes[h] > INT32_MAX) {
                throw std::runtime_error(format("PLE head %u range does not fit the int32 row index", h));
            }
            hparams.ple_head_offsets[h]     = (uint32_t) head_offsets[h];
            hparams.ple_head_vocab_sizes[h] = (uint32_t) head_vocab_sizes[h];
        }
    }

    // linear attention everywhere except every full_attention_interval-th layer
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        qwen4exp_require_nonzero(ml, LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    // the PLE conv history is a row of the recurrent cache, which linear layers alone have
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        if (hparams.is_ple(i) && !hparams.is_recr(i)) {
            throw std::runtime_error(format("PLE layer %u is not a linear attention layer", i));
        }
    }

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t hc_lr  = hparams.hc_low_rank;

    const auto nf = nextn_flags(ml, LLM_TENSOR_HC_ATTN_NORM);
    const int trunk_flags = nf.trunk;
    const int mtp_flags   = nf.mtp;

    // AMB_MTP_SHARE_HEAD (C26): decide for an MTP-only file whether its token_embd/output are left to the target
    // C26b: 0 also when the --fit pass suppressed sharing for this process (llama_amb_mtp_share_head_suppress)
    const int  share_head = amb_mtp_share_head_active();
    const bool mtp_only  = (trunk_flags & TENSOR_NOT_REQUIRED) != 0 && (mtp_flags & (TENSOR_NOT_REQUIRED | TENSOR_SKIP)) == 0;
    bool skip_out = false;
    bool skip_emb = false;
    if (share_head > 0 && mtp_only) {
        const std::string out_name = tn(LLM_TENSOR_OUTPUT,     "weight").str();
        const std::string emb_name = tn(LLM_TENSOR_TOKEN_EMBD, "weight").str();
        const auto * w_out = ml.get_weight(out_name.c_str());
        const auto * w_emb = ml.get_weight(emb_name.c_str());

        amb_head_record rec;
        {
            std::lock_guard<std::mutex> lock(g_amb_head_mutex);
            rec = g_amb_head;
        }

        const char * why = nullptr;
        if (!rec.valid || rec.owner == nullptr) {
            // C26b: also after the recorded target was freed (amb_mtp_share_head_forget)
            why = "no live qwen4exp target with token_embd/output was loaded before the draft";
        } else if (w_out == nullptr) {
            why = "the draft file has no output.weight (nothing to skip; a draft without one reads the target's anyway)";
        } else if (w_out->tensor->ne[0] != rec.out_ne0 || w_out->tensor->ne[1] != rec.out_ne1 ||
                   w_out->tensor->ne[2] != 1 || w_out->tensor->ne[3] != 1 || rec.out_ne0 != n_embd) {
            why = "output.weight shapes differ between draft and target";
        } else if (rec.n_vocab != n_vocab) {
            why = "the draft and target vocabularies differ in size";
        } else if (rec.out_host) {
            why = "the target's output.weight is not in device memory";
        } else if (share_head == 1 && w_emb != nullptr &&
                   (w_emb->tensor->ne[0] != rec.emb_ne0 || w_emb->tensor->ne[1] != rec.emb_ne1 ||
                    w_emb->tensor->ne[2] != 1 || w_emb->tensor->ne[3] != 1)) {
            why = "token_embd.weight shapes differ between draft and target";
        }

        if (why != nullptr) {
            LLAMA_LOG_WARN("%s: amb_mtp_share_head: AMB_MTP_SHARE_HEAD=%d refused: %s; the draft loads its own "
                    "token_embd/output as without the switch\n", __func__, share_head, why);
        } else {
            skip_out = true;
            skip_emb = share_head == 1 && w_emb != nullptr;

            // C26b: what the draft context must find in its ctx_other's model (llama_context checks it)
            amb_share_head.tgt         = rec.owner;
            amb_share_head.skipped_out = skip_out;
            amb_share_head.skipped_emb = skip_emb;
            amb_share_head.out_ne0     = rec.out_ne0;
            amb_share_head.out_ne1     = rec.out_ne1;
            amb_share_head.out_type    = rec.out_type;
            amb_share_head.emb_ne0     = rec.emb_ne0;
            amb_share_head.emb_ne1     = rec.emb_ne1;
            amb_share_head.emb_type    = rec.emb_type;

            const ggml_backend_dev_t dev_out = dev_output();
            const bool out_on_dev = dev_out != nullptr && ggml_backend_dev_type(dev_out) != GGML_BACKEND_DEVICE_TYPE_CPU;
            const size_t out_bytes = ggml_nbytes(w_out->tensor);
            const size_t emb_bytes = skip_emb ? ggml_nbytes(w_emb->tensor) : 0;
            // the input layer (token_embd) is always assigned to the CPU, see llama_model::load_tensors
            const size_t vram_bytes = out_on_dev ? out_bytes : 0;

            std::string emb_part = "token_embd.weight kept (AMB_MTP_SHARE_HEAD=2)";
            if (skip_emb) {
                emb_part = format("token_embd.weight %s %.2f MiB (host) -> target's %s on %s", ggml_type_name(w_emb->tensor->type),
                        emb_bytes/1024.0/1024.0, ggml_type_name(rec.emb_type), rec.emb_buft.c_str());
            } else if (share_head == 1) {
                emb_part = "token_embd.weight not in the draft file";
            }
            LLAMA_LOG_INFO("%s: amb_mtp_share_head: AMB_MTP_SHARE_HEAD=%d: [draft model] not loaded: output.weight %s "
                    "%.2f MiB (%s) -> target's %s on %s; %s; device memory not uploaded: %.2f MiB (%zu bytes), host memory "
                    "not loaded: %.2f MiB\n", __func__, share_head, ggml_type_name(w_out->tensor->type), out_bytes/1024.0/1024.0,
                    out_on_dev ? ggml_backend_dev_name(dev_out) : "host", ggml_type_name(rec.out_type), rec.out_buft.c_str(),
                    emb_part.c_str(), vram_bytes/1024.0/1024.0, vram_bytes,
                    ((out_on_dev ? 0 : out_bytes) + emb_bytes)/1024.0/1024.0);
        }
    }

    if (skip_emb) {
        // counted as an unused tensor of the file: nothing is allocated, mapped or read
        create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED | TENSOR_SKIP);
        tok_embd = nullptr;
    } else {
        tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, trunk_flags);
    }

    // there is no output_norm: the final hyper-connection mixer carries it
    // the gammas load as [n_embd, hc] so the grouped norm multiplies them without a graph reshape
    hc_head_norm = create_tensor(tn(LLM_TENSOR_HC_HEAD_NORM, "weight"), { n_embd, hc }, trunk_flags | TENSOR_ALLOW_RESHAPE);
    hc_head_down = create_tensor(tn(LLM_TENSOR_HC_HEAD_DOWN, "weight"), { hc_dim, hc_lr }, trunk_flags);
    hc_head_up   = create_tensor(tn(LLM_TENSOR_HC_HEAD_UP,   "weight"), { hc_lr, hc_dim }, trunk_flags);

    if (skip_out) {
        create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED | TENSOR_SKIP);
        output = nullptr; // no tie to token_embd: the target's output is read (graph_mtp, cparams.model_tgt)
    } else {
        output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
        if (output == NULL && tok_embd != NULL) {
            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
        }
    }

    // AMB_MTP_SHARE_HEAD (C26): a trunk file records its token_embd/output for a draft loaded after it
    if (amb_mtp_share_head_env() > 0 && (trunk_flags & TENSOR_NOT_REQUIRED) == 0 && tok_embd != nullptr && output != nullptr) {
        amb_head_record rec;
        rec.valid    = true;
        rec.owner    = this; // C26b: identity of the target; dropped in ~llama_model
        rec.out_ne0  = output->ne[0];
        rec.out_ne1  = output->ne[1];
        rec.emb_ne0  = tok_embd->ne[0];
        rec.emb_ne1  = tok_embd->ne[1];
        rec.n_vocab  = n_vocab;
        rec.out_type = output->type;
        rec.emb_type = tok_embd->type;
        ggml_backend_buffer_type_t out_buft = amb_created_buft(ml, output);
        ggml_backend_buffer_type_t emb_buft = amb_created_buft(ml, tok_embd);
        rec.out_host = amb_buft_is_host_like(out_buft); // C26b: by device type, not only is_host
        rec.out_buft = out_buft ? ggml_backend_buft_name(out_buft) : "?";
        rec.emb_buft = emb_buft ? ggml_backend_buft_name(emb_buft) : "?";
        std::lock_guard<std::mutex> lock(g_amb_head_mutex);
        g_amb_head = rec;
    }

    // flat [ple_head_dim, n_rows] gather target
    if (hparams.ple_n_heads > 0) {
        // the head ranges are what the gather indexes, so they set the minimum row count
        int64_t ple_rows = 0;
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            ple_rows = std::max(ple_rows, (int64_t) hparams.ple_head_offsets[h] + hparams.ple_head_vocab_sizes[h]);
        }

        // the converter pads the table; a model synthesised from metadata has no tensor to ask
        const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        if (const auto * ple_w = ml.get_weight(ple_name.c_str())) {
            if (ple_w->tensor->ne[1] < ple_rows) {
                throw std::runtime_error(format("%s has %" PRId64 " rows, too few for the PLE head ranges (%" PRId64 ")",
                                                ple_name.c_str(), ple_w->tensor->ne[1], ple_rows));
            }
            ple_rows = ple_w->tensor->ne[1];
        }

        per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                           { hparams.ple_head_dim, ple_rows }, TENSOR_READ_LAZY);
        // an FP8 table keeps its checkpoint bytes, so its per-tensor scale is applied after the gather
        per_layer_tok_embd_s = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "scale"), { 1 }, TENSOR_NOT_REQUIRED);
        if (per_layer_tok_embd_s && per_layer_tok_embd_s->type != GGML_TYPE_F32) {
            throw std::runtime_error(format("%s must be F32", per_layer_tok_embd_s->name));
        }
    }

    auto load_block = [&](int il, int flags) {
        auto & layer = layers[il];

        const int64_t n_ff_exp   = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        // matmuls that read the same input: try the merged [a | b] weight the converter writes, else the separate ones
        auto create_merged = [&](llm_tensor t_merged, llm_tensor t_a, llm_tensor t_b, int64_t n_in, int64_t n_a, int64_t n_b,
                ggml_tensor *& merged, ggml_tensor *& a, ggml_tensor *& b) {
            if (flags & TENSOR_SKIP) {
                const int skip = TENSOR_NOT_REQUIRED | TENSOR_SKIP;
                create_tensor(tn(t_merged, "weight", il), { n_in, n_a + n_b }, skip | TENSOR_SKIP_IF_VIRTUAL);
                create_tensor(tn(t_a,      "weight", il), { n_in, n_a },       skip);
                create_tensor(tn(t_b,      "weight", il), { n_in, n_b },       skip);
                return;
            }
            merged = create_tensor(tn(t_merged, "weight", il), { n_in, n_a + n_b }, flags | TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
            if (!merged) {
                a = create_tensor(tn(t_a, "weight", il), { n_in, n_a }, flags);
                b = create_tensor(tn(t_b, "weight", il), { n_in, n_b }, flags);
            }
        };

        // two HC modules per layer: before the token mixer, before the MoE
        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { hc_lr, hc_dim }, flags);
        create_merged(LLM_TENSOR_HC_ATTN_DOWN_INJECT, LLM_TENSOR_HC_ATTN_DOWN, LLM_TENSOR_HC_ATTN_INJECT, hc_dim, hc_lr, hc,
                layer.hc_attn_down_inject, layer.hc_attn_down, layer.hc_attn_inject);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { hc_lr, hc_dim }, flags);
        create_merged(LLM_TENSOR_HC_FFN_DOWN_INJECT, LLM_TENSOR_HC_FFN_DOWN, LLM_TENSOR_HC_FFN_INJECT, hc_dim, hc_lr, hc,
                layer.hc_ffn_down_inject, layer.hc_ffn_down, layer.hc_ffn_inject);

        if (!hparams.is_recr(il)) {
            // full attention: wq holds [q|gate] interleaved per head
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);

            const int64_t idx_dim = hparams.indexer_head_size;
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, hparams.indexer_n_head * idx_dim }, flags);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, idx_dim }, flags);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { idx_dim }, flags);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { idx_dim }, flags);
        } else {
            create_merged(LLM_TENSOR_ATTN_QKV_GATE, LLM_TENSOR_ATTN_QKV, LLM_TENSOR_ATTN_GATE, n_embd, key_dim * 2 + value_dim, value_dim,
                    layer.wqkv_z, layer.wqkv, layer.wqkv_gate);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, flags);
            create_merged(LLM_TENSOR_SSM_BETA_ALPHA_CAT, LLM_TENSOR_SSM_BETA, LLM_TENSOR_SSM_ALPHA, n_embd, n_v_heads, n_v_heads,
                    layer.ssm_beta_alpha_cat, layer.ssm_beta, layer.ssm_alpha);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, flags);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, flags);
        }

        if (hparams.is_ple(il)) {
            layer.ple_key        = create_tensor(tn(LLM_TENSOR_PLE_KEY,        "weight", il), { n_embd, hc_dim }, flags);
            layer.ple_value      = create_tensor(tn(LLM_TENSOR_PLE_VALUE,      "weight", il), { n_embd, n_embd }, flags);
            layer.ple_norm_key   = create_tensor(tn(LLM_TENSOR_PLE_NORM_KEY,   "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_norm_query = create_tensor(tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_norm_conv  = create_tensor(tn(LLM_TENSOR_PLE_NORM_CONV,  "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_conv1d     = create_tensor(tn(LLM_TENSOR_PLE_CONV1D,     "weight", il), { hparams.ple_conv_kernel, hc_dim }, flags);
        }

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, flags);
    };

    for (int il = 0; il < n_layer; ++il) {
        load_block(il, trunk_flags);
    }

    // the MTP block: one full-attention QSA layer fed by [enorm(e) ; hnorm(h)_s] -> eh_proj per hc stream
    for (int il = n_layer; il < n_layer_all; ++il) {
        load_block(il, mtp_flags);

        auto & nextn = layers[il].nextn;
        nextn.eh_proj      = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,      "weight", il), { 2*n_embd, n_embd }, mtp_flags);
        nextn.enorm        = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,        "weight", il), { n_embd }, mtp_flags);
        // RMS per hc stream of the trunk residual, so the gammas load as [n_embd, hc] like the mixer norms
        nextn.hnorm        = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,        "weight", il), { n_embd, hc }, mtp_flags | TENSOR_ALLOW_RESHAPE);
        nextn.hc_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_NORM, "weight", il), { n_embd, hc }, mtp_flags | TENSOR_ALLOW_RESHAPE);
        nextn.hc_head_down = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_DOWN, "weight", il), { hc_dim, hc_lr }, mtp_flags);
        nextn.hc_head_up   = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_UP,   "weight", il), { hc_lr, hc_dim }, mtp_flags);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen4exp::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(params);
    }
    return std::make_unique<graph>(params);
}

// DFlash drafters read the layer inputs as [n_embd, T] rows, so collapse the streams
static ggml_tensor * hc_stream_mean(ggml_context * ctx, ggml_tensor * x) {
    const int64_t n_embd = x->ne[0];
    const int64_t hc     = x->ne[1];
    const int64_t nt     = x->ne[2];

    ggml_tensor * acc = ggml_view_2d(ctx, x, n_embd, nt, x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx, acc, ggml_view_2d(ctx, x, n_embd, nt, x->nb[2], s*x->nb[1]));
    }

    return ggml_scale(ctx, acc, 1.0f / (float) hc);
}

// Hyper-connections keep hc parallel residual streams [n_embd, hc, T] in place of layer norms.
// build_hc_mix returns the mixed [n_embd, T] stream and the [hc, T] scatter weights.
llama_model_qwen4exp::graph::hc_mix
llama_model_qwen4exp::graph::build_hc_mix(
        ggml_tensor *  x,
        ggml_tensor *  w_norm,
        ggml_tensor *  w_down,
        ggml_tensor *  w_up,
        ggml_tensor *  w_inject,
        ggml_tensor *  w_down_inject,
        int            il) {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    // grouped RMSNorm: reduce over one stream, then scale all streams with the [n_embd, hc] gamma
    // the converter folded each gamma to (1 + w)
    ggml_tensor * xn = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps), w_norm);
    xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    cb(xn, "hc_norm", il);

    ggml_tensor * lo = nullptr;
    ggml_tensor * inject = nullptr;
    if (w_down_inject) {
        // one matmul for both: the hc_down rows, then the hc_inject rows
        const int64_t n_lo = w_down_inject->ne[1] - hc;
        ggml_tensor * lo_inject = build_lora_mm(w_down_inject, xn);
        // scale the combined rows before the split; both lo and inject use 1/hc
        lo_inject = ggml_scale(ctx0, lo_inject, 1.0f / (float) hc);
        lo     = ggml_view_2d(ctx0, lo_inject, n_lo, nt, lo_inject->nb[1], 0);
        inject = ggml_view_2d(ctx0, lo_inject, hc, nt, lo_inject->nb[1], ggml_row_size(lo_inject->type, n_lo));
        cb(inject, "hc_inject", il);
    } else {
        lo = ggml_scale(ctx0, build_lora_mm(w_down, xn), 1.0f / (float) hc);
    }
    lo = ggml_silu(ctx0, lo);
    ggml_tensor * gate = build_lora_mm(w_up, lo);
    cb(gate, "hc_gate", il);

    ggml_tensor * mixed = nullptr;
    if (cparams.fused_dsv4_hc_pre && il >= 0) {
        // sigmoid gate and mean over the streams in one op
        mixed = ggml_dsv4_hc_pre_gated(ctx0,
                ggml_reshape_3d(ctx0, xn,   n_embd, hc, nt),
                ggml_reshape_3d(ctx0, gate, n_embd, hc, nt), 1.0f / (float) hc);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, mixed, il});
    } else {
        ggml_tensor * gated = ggml_mul(ctx0, xn, ggml_sigmoid(ctx0, gate));
        gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

        // collapse the streams by their mean
        mixed = ggml_view_2d(ctx0, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd) * hc, 0);
        mixed = ggml_cont(ctx0, mixed);
        for (int64_t c = 1; c < hc; ++c) {
            ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                    ggml_row_size(gated->type, n_embd) * hc,
                    ggml_row_size(gated->type, n_embd) * c);
            mixed = ggml_add(ctx0, mixed, s);
        }
        mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
    }
    cb(mixed, "hc_mixed", il);

    if (!w_down_inject && w_inject) {
        inject = ggml_scale(ctx0, build_lora_mm(w_inject, xn), 1.0f / (float) hc);
        cb(inject, "hc_inject", il);
    }

    return { mixed, inject };
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_combine(
        ggml_tensor * residual,
        ggml_tensor * block_out,
        ggml_tensor * inject,
        int           il) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = residual->ne[2];

    // 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add
    ggml_tensor * w = ggml_sigmoid(ctx0, inject);
    w = ggml_scale(ctx0, w, 2.0f);

    ggml_tensor * cur = nullptr;
    if (cparams.fused_dsv4_hc_post && il >= 0) {
        // identity comb: every stream adds the same block output, scaled by its own weight
        cur = ggml_dsv4_hc_post(ctx0, block_out, residual, w, nullptr);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, cur, il});
    } else {
        w = ggml_reshape_3d(ctx0, w, 1, hc, nt);

        ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
        b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

        cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
    }
    cb(cur, "hc_combine", il);

    return cur;
}

llama_model_qwen4exp::graph::graph(const llm_graph_params & params) :
    llm_build_delta_net_base(params) {
    const int64_t hc = hparams.dsv4_hc_mult;

    GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());

    auto * inp = build_inp_mem_hybrid();

    // qwen4exp always builds llama_memory_hybrid_idx, so this downcast is safe
    // the indexer cache inside it is absent when the GGUF has no indexer tensors
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp->mctx);

    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();
    if (mctx_idx) {
        GGML_ASSERT(mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv() &&
                "the indexer cache must track the attention cache cell for cell");
    }

    ggml_tensor * ple_emb = nullptr;
    if (hparams.ple_n_heads > 0) {
        ple_emb = build_inp_ple(mctx_hyb);
        // make sure ple_emb and build_inp_embd are in the same graph split
        ggml_build_forward_expand(gf, ple_emb);
    }

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);
    ggml_build_forward_expand(gf, inpL);

    // the QSA layers share one set of k-pool inputs
    // the CUDA lightning indexer takes 32 or 64 heads, QSA has a few, so it scores with plain ops
    llm_graph_input_kpool * inp_kpool = nullptr;
    if (mctx_idx && hparams.indexer_kpool > 0) {
        inp_kpool = build_inp_kpool(mctx_hyb);
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // the wide residual starts as hc identical copies of the embedding
    ggml_tensor * res_hc = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(res_hc, "hc_init", -1);
    // make sure hc_init is in the same graph split as the first layer (-sm tensor)
    ggml_build_forward_expand(gf, res_hc);

    // AMB_MOE_PREDICT (measurement only, see llama-moe-cache.cpp): early router probes for single-token ubatches
    // the router of layer il (ffn_gate_inp, softmax, descending argsort, as build_moe_ffn) applied to earlier hidden states;
    // the probes are only built here, llm_graph_result::set_outputs appends them after the real graph
    // not with expert groups: build_moe_ffn masks the groups before the top-k, the probes do not
    const bool moe_predict = moe_cache != nullptr && n_tokens == 1 && hparams.n_expert_groups <= 1;
    std::vector<ggml_tensor *> pred_ffn_in;  // [n_layer] router input of the layer (output of its FFN mixer)
    std::vector<ggml_tensor *> pred_res_ffn; // [n_layer] residual streams read by the FFN mixer of the layer
    if (moe_predict) {
        pred_ffn_in.assign(n_layer, nullptr);
        pred_res_ffn.assign(n_layer, nullptr);
    }

    for (int il = 0; il < n_layer; ++il) {
        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = hc_stream_mean(ctx0, res_hc);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        if (hparams.is_ple(il)) {
            res_hc = build_ple(inp->get_recr(), ple_emb, res_hc, il);
        }

        // AMB_MOE_PREDICT: the residual streams read by the attention mixer of the layer
        ggml_tensor * pred_res_attn = res_hc;

        hc_mix attn_mix = build_hc_mix(res_hc,
                model.layers[il].hc_attn_norm,
                model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,
                model.layers[il].hc_attn_inject,
                model.layers[il].hc_attn_down_inject,
                il);
        ggml_tensor * cur = attn_mix.cur;
        ggml_tensor * pred_attn_in = attn_mix.cur; // AMB_MOE_PREDICT: the input of GDN/QSA

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), mctx_hyb, inp_kpool, cur, inp_pos, il);
        }

        if (il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {
            // everything below is per token, so drop the rows that produce no output
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            attn_mix.inject = ggml_get_rows(ctx0, attn_mix.inject, inp_out_ids);

            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }

        res_hc = build_hc_combine(res_hc, cur, attn_mix.inject, il);

        hc_mix ffn_mix = build_hc_mix(res_hc,
                model.layers[il].hc_ffn_norm,
                model.layers[il].hc_ffn_down,
                model.layers[il].hc_ffn_up,
                model.layers[il].hc_ffn_inject,
                model.layers[il].hc_ffn_down_inject,
                il);
        cur = ffn_mix.cur;

        if (moe_predict) {
            pred_res_ffn[il] = res_hc;
            pred_ffn_in[il]  = ffn_mix.cur;
        }

        // AMB_MOE_PREFETCH (prototype, see llama-moe-cache.cpp): the predicted experts of layer il + 1, from the state of layer il
        // expanded here, before the router of layer il: the nodes precede the slot lookup of layer il, which starts a new split
        // (it reads the host slot map), so they are computed in an earlier split on the same stream and are complete when the
        // copy callback of that lookup synchronizes the stream to read the real ids; that readback also fetches these ids
        // only nodes are added: the real router reads the same tensors; no expert groups (the real top-k masks the groups)
        if (moe_cache != nullptr && n_tokens == 1 && hparams.n_expert_groups <= 1 && il + 1 < n_layer) {
            const int pf_mode = moe_cache->prefetch_mode(il + 1, n_tokens);
            if (pf_mode != 0) {
                const auto & ly = model.layers[il + 1];
                ggml_tensor * x = nullptr;
                if (pf_mode == 1) {
                    // P1: the router input of layer il (output of its FFN mixer)
                    x = ffn_mix.cur;
                } else if (ly.hc_ffn_norm != nullptr) {
                    // P4: the FFN mixer of layer il + 1 on the residual streams the FFN mixer of layer il reads, as the probe;
                    // il = -1 keeps it out of the fused HC ops
                    x = build_hc_mix(res_hc, ly.hc_ffn_norm, ly.hc_ffn_down, ly.hc_ffn_up, nullptr, ly.hc_ffn_down_inject, -1).cur;
                }
                if (x != nullptr && ly.ffn_gate_inp != nullptr && x->ne[0] == ly.ffn_gate_inp->ne[0]) {
                    // softmax gating without a selection bias: the top-k of the logits is the top-k of the probabilities
                    ggml_tensor * logits = build_lora_mm(ly.ffn_gate_inp, x); // [n_expert, 1]
                    ggml_tensor * ids    = ggml_top_k(ctx0, logits, moe_cache->prefetch_k()); // I32 [k, 1], unordered
                    ggml_format_name(ids, "amb_moe_pf%d-%d", pf_mode, il + 1);
                    // an output: the allocator keeps its memory until the copy callback of layer il has read it
                    ggml_set_output(ids);
                    ggml_build_forward_expand(gf, ids);
                    res->t_moe_pf.push_back({ il + 1, ids });
                }
            }
        }

        // AMB_MOE_LOOKAHEAD (C18, see llama-moe-cache.cpp): the predicted experts of layer il + 1 for every token of a small
        // ubatch, at the same place as the C6 nodes above (the two switches exclude each other): before the router of layer il,
        // so the nodes are in the split that precedes the slot lookup of layer il and are read back by AMB_IDS_ASYNC together
        // with the real ids of layer il (same event, no extra sync)
        // exactness: only new nodes are added; they read existing tensors (ffn_mix.cur or res_hc) and device weights of layer
        // il + 1. Inserted after the last node of the mixer chain (P1) or after the HC combine (P4): no CUDA fusion window
        // starts at DSV4_HC_PRE / SCALE (end of the mixer) or DSV4_HC_POST, and the router mul_mat stays next to its
        // softmax; the allocator decrements child counts in node order, so the in-place decisions of the real nodes (all
        // after these nodes) do not change. Not proven bit-identical: the extra allocations can move later tensors, which
        // only matters for the address-dependent fusion checks; the tester compares tokens with the switch off and in DRY mode
        // (the block is a lambda only so that AMB_MOE_MISS_OVERLAP can expand it later; with the switch unset it is called
        // right here, so the nodes and their order are exactly as before)
        auto la_predict = [&](const bool mo_publish) {
            if (moe_cache != nullptr && hparams.n_expert_groups <= 1 && il + 1 < n_layer) {
                const int la_mode = moe_cache->lookahead_mode(il + 1, n_tokens);
                const auto & ly = model.layers[il + 1];
                // the scheduler must not see a host weight here (a new split would change the split layout)
                auto on_device = [](const ggml_tensor * w) {
                    return w == nullptr || (w->buffer != nullptr && !ggml_backend_buffer_is_host(w->buffer));
                };
                if (la_mode != 0 && ly.ffn_gate_inp != nullptr && on_device(ly.ffn_gate_inp)) {
                    ggml_tensor * x = nullptr;
                    if (la_mode == 1) {
                        // P1: the router of layer il + 1 on the router input of layer il (output of its FFN mixer)
                        x = ffn_mix.cur;
                    } else if (la_mode == 4 && ly.hc_ffn_norm != nullptr && on_device(ly.hc_ffn_norm) && on_device(ly.hc_ffn_down) &&
                               on_device(ly.hc_ffn_up) && on_device(ly.hc_ffn_down_inject)) {
                        if (moe_cache->lookahead_p4_fast() && cparams.fused_dsv4_hc_pre && ly.hc_ffn_up != nullptr &&
                                (ly.hc_ffn_down_inject != nullptr || ly.hc_ffn_down != nullptr)) {
                            // C18b AMB_MOE_LOOKAHEAD_P4_FAST=1: the same mixer with fewer kernels, predictor nodes only: the
                            // inject rows of the merged down matrix are not computed (a row view of the weight), and the sigmoid
                            // gate + stream mean is the fused DSV4_HC_PRE op the real mixers use (not registered with
                            // add_fused_node: the real mixers already register the op; no cb(), as for the other predictor nodes)
                            const int64_t hc_n = hparams.dsv4_hc_mult;
                            const int64_t nt   = res_hc->ne[2];
                            ggml_tensor * xn = ggml_mul(ctx0, ggml_rms_norm(ctx0, res_hc, hparams.f_norm_rms_eps), ly.hc_ffn_norm);
                            xn = ggml_reshape_2d(ctx0, xn, hc_n*n_embd, nt);
                            ggml_tensor * w_lo = ly.hc_ffn_down;
                            if (ly.hc_ffn_down_inject != nullptr) {
                                const int64_t n_lo = ly.hc_ffn_down_inject->ne[1] - hc_n;
                                w_lo = ggml_view_2d(ctx0, ly.hc_ffn_down_inject, ly.hc_ffn_down_inject->ne[0], n_lo,
                                        ly.hc_ffn_down_inject->nb[1], 0);
                            }
                            ggml_tensor * lo = ggml_silu(ctx0, ggml_scale(ctx0, ggml_mul_mat(ctx0, w_lo, xn), 1.0f / (float) hc_n));
                            ggml_tensor * gate = ggml_mul_mat(ctx0, ly.hc_ffn_up, lo);
                            x = ggml_dsv4_hc_pre_gated(ctx0,
                                    ggml_reshape_3d(ctx0, xn,   n_embd, hc_n, nt),
                                    ggml_reshape_3d(ctx0, gate, n_embd, hc_n, nt), 1.0f / (float) hc_n);
                        } else {
                            // P4: the FFN mixer of layer il + 1 on the residual streams the FFN mixer of layer il reads; il = -1
                            // keeps it out of the fused HC ops (and out of res->add_fused_node)
                            x = build_hc_mix(res_hc, ly.hc_ffn_norm, ly.hc_ffn_down, ly.hc_ffn_up, nullptr, ly.hc_ffn_down_inject, -1).cur;
                        }
                    }
                    // C18b (review m5): the last layer runs its FFN on the output rows only (crop_before_nextn above): predict
                    // those rows only (the readback then has fewer rows than the ubatch, see ids_post_split)
                    if (x != nullptr && il + 1 == n_layer - 1 && crop_before_nextn(inp_out_ids)) {
                        x = ggml_get_rows(ctx0, x, inp_out_ids);
                    }
                    if (x != nullptr && x->ne[0] == ly.ffn_gate_inp->ne[0]) {
                        // softmax gating without a selection bias: the top-k of the logits is the top-k of the probabilities
                        ggml_tensor * logits = ggml_mul_mat(ctx0, ly.ffn_gate_inp, x); // [n_expert, n_rows]
                        ggml_tensor * ids = nullptr;
                        if (moe_cache->lookahead_rank()) {
                            // C18b AMB_MOE_LOOKAHEAD_RANK=1: in descending logit order (argsort + view), contiguous for the readback
                            ids = ggml_cont(ctx0, ggml_argsort_top_k(ctx0, logits, moe_cache->lookahead_k())); // I32 [k, n_rows]
                        } else {
                            ids = ggml_top_k(ctx0, logits, moe_cache->lookahead_k()); // I32 [k, n_rows], unordered
                        }
                        ggml_format_name(ids, "amb_moe_la%d-%d", la_mode, il + 1);
                        // an output: the allocator keeps its memory for the readback enqueued after the split
                        ggml_set_output(ids);
                        ggml_build_forward_expand(gf, ids);
                        res->t_moe_pf.push_back({ il + 1, ids });
                        if (mo_publish) {
                            // AMB_MOE_MISS_OVERLAP (C23): published right after the predictor, in the tail that runs
                            // after the publish of the real ids of layer il (see the hook below)
                            ggml_tensor * pub = moe_cache->miss_overlap_publish(ctx0, ids, il + 1, 1);
                            ggml_backend_sched_set_tensor_backend(sched, pub, moe_cache->backend(il + 1));
                            ggml_build_forward_expand(gf, pub);
                        }
                    }
                }
            }
        };

        // AMB_MOE_MISS_OVERLAP=1 (C23, default off; see llama-moe-cache.cpp): build_moe_cache_slots publishes the selected
        // ids of layer il to the host right after the router, then calls moe_pre_lookup, which expands the predictor of
        // layer il + 1 (with a publish of its ids) and the shared expert of layer il: work that does not depend on the
        // experts of layer il, so it runs at the end of the split that precedes the slot lookup, while the host plans layer
        // il and the missed experts upload on the second stream. Exactness: only the order of independent nodes changes
        // (each node reads the same tensors); the shared expert's output is added to the routed output as before
        if (moe_cache == nullptr || !moe_cache->miss_overlap(il, cur->ne[1])) {
            la_predict(false);
            cur = build_layer_ffn(cur, il);
        } else {
            ggml_tensor * ffn_in   = cur;
            ggml_tensor * mo_shexp = nullptr;
            bool          mo_ran   = false;
            moe_pre_lookup = [&]() {
                if (mo_ran) {
                    return;
                }
                mo_ran = true;
                la_predict(true);
                mo_shexp = build_shexp(ffn_in, il);
                if (mo_shexp != nullptr) {
                    ggml_build_forward_expand(gf, mo_shexp);
                }
            };
            cur = build_layer_ffn(ffn_in, il, &mo_shexp);
            moe_pre_lookup = nullptr;
            if (!mo_ran) {
                // build_moe_cache_slots refused the lookup after all (a backend support check): no publish, the predictor
                // (if any) follows the FFN; nothing reads its ids early in that case
                la_predict(false);
            }
        }
        cb(cur, "ffn_out", il);

        res_hc = build_hc_combine(res_hc, cur, ffn_mix.inject, il);

        // "l_last" is the layer output name that build_cvec and imatrix look for
        cb(res_hc, "l_last", il);

        if (moe_predict && moe_cache->predict_layer(il, n_tokens)) {
            const auto & ly = model.layers[il];
            // P0..P5, see llama_moe_cache::impl::predict_name
            auto probe = [&](int p, ggml_tensor * x) {
                if (x == nullptr || x->ne[0] != ly.ffn_gate_inp->ne[0]) {
                    return;
                }
                ggml_tensor * logits = build_lora_mm(ly.ffn_gate_inp, x); // [n_expert, n_tokens]
                ggml_tensor * probs  = ggml_soft_max(ctx0, logits);
                ggml_tensor * ids    = ggml_argsort(ctx0, probs, GGML_SORT_ORDER_DESC);
                ggml_format_name(ids, "amb_moe_pred%d-%d", p, il);
                res->t_moe_pred.push_back({ il, p, ids });
            };
            // the FFN mixer of this layer on other residual streams; il = -1 keeps it out of the fused HC ops
            auto ffn_mixer = [&](ggml_tensor * r) -> ggml_tensor * {
                if (r == nullptr) {
                    return nullptr;
                }
                return build_hc_mix(r, ly.hc_ffn_norm, ly.hc_ffn_down, ly.hc_ffn_up, nullptr, ly.hc_ffn_down_inject, -1).cur;
            };
            probe(0, pred_ffn_in[il]);
            probe(1, il >= 1 ? pred_ffn_in[il - 1] : nullptr);
            probe(2, pred_attn_in);
            probe(3, il >= 2 ? pred_ffn_in[il - 2] : nullptr);
            probe(4, il >= 1 ? ffn_mixer(pred_res_ffn[il - 1]) : nullptr);
            probe(5, ffn_mixer(pred_res_attn));
        }
    }

    // the MTP head reads the hc-wide residual, before the final mixer
    res->t_h_nextn = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
    cb(res->t_h_nextn, "h_nextn", -1);
    ggml_build_forward_expand(gf, res->t_h_nextn);

    if (crop_after_nextn(inp_out_ids)) {
        res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
        res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
        res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
    }

    // the final mixer is the output norm: there is no separate one
    hc_mix head_mix = build_hc_mix(res_hc,
            model.hc_head_norm, model.hc_head_down, model.hc_head_up,
            nullptr, nullptr, -1);
    ggml_tensor * cur = head_mix.cur;

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

llama_model_qwen4exp::graph_mtp::graph_mtp(const llm_graph_params & params) :
    graph(params, no_build{}) {
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "qwen4exp MTP has a single block");
    ASSERT_EMBD_OR_TOKEN(ubatch);

    const int64_t hc = hparams.dsv4_hc_mult;
    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) (n_embd*hc) && "qwen4exp MTP hidden width mismatch");

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && layer.nextn.enorm && layer.nextn.hnorm && layer.nextn.hc_head_norm &&
            "MTP block missing, load the model with MTP enabled");

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_inp(), hparams.n_embd_out());

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    inp->h = build_inp_embd_state(inp.get(), hparams.n_embd_out());

    const llama_model & emb  = model.tok_embd ? model : *cparams.model_tgt;
    const llama_model & head = model.output   ? model : *cparams.model_tgt;

    ggml_tensor * tok_embd;
    if (ubatch.token) {
        tok_embd = ggml_get_rows(ctx0, emb.tok_embd, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h = inp->h;

    res->add_input(std::move(inp));

    auto * inp_hyb = build_inp_mem_hybrid();
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp_hyb->mctx);

    // the draft memory has no recurrent layer, but its input still has to be allocated
    ggml_build_forward_expand(gf, inp_hyb->get_recr()->s_copy);

    llm_graph_input_kpool * inp_kpool = nullptr;
    if (mctx_hyb->get_idx() && hparams.indexer_kpool > 0 && hparams.dsv4_compress_ratios[il] > 0) { // ratio 0: dense MTP block
        GGML_ASSERT(mctx_hyb->get_idx()->get_n_kv() == mctx_hyb->get_attn()->get_n_kv() &&
                "the indexer cache must track the attention cache cell for cell");
        inp_kpool = build_inp_kpool(mctx_hyb);
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * h_norm = build_norm(ggml_reshape_3d(ctx0, h, n_embd, hc, n_tokens), layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e_norm = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens), n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * res_hc = build_lora_mm(layer.nextn.eh_proj, ggml_concat(ctx0, e_norm, h_norm, 0)); // [n_embd, hc, n_tokens]
    cb(res_hc, "mtp_eh_proj", il);

    hc_mix attn_mix = build_hc_mix(res_hc, layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject, layer.hc_attn_down_inject, il);
    ggml_tensor * cur    = attn_mix.cur;
    cur    = build_layer_attn(inp_hyb->get_attn(), mctx_hyb, inp_kpool, cur, inp_pos, il);

    // AMB_MTP_CROP=1 (C16, default off): keep only the output rows from here on, as the target graph does before its
    // last layer's combine (graph(), crop_before_nextn). Every state write of the draft (K/V rows, raw and pooled indexer
    // keys) is inside build_layer_attn and was expanded into gf above from the full rows, so it is unchanged; everything
    // below is per token. The tensors read below this point are exactly cur (attention output), attn_mix.inject (scatter
    // weights of the attention combine) and res_hc (the wide residual, read by the attention combine, the FFN mixer, the
    // FFN combine, h_nextn and the head); each is cropped once here. The FFN/MoE then sees n_outputs rows, so with
    // --spec-draft-cpu-moe its expert matmuls stay on the CPU (below the offload batch) and the scheduler plans no device
    // copy of a host expert tensor, also in the reserve graph (n_tokens = n_ubatch, n_outputs = n_outputs_max)
    bool crop = false;
    if (amb_mtp_crop_wanted()) {
        const char * why = amb_mtp_crop_blocker(inp_out_ids,
                { layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps, layer.ffn_gate_up_exps });
        crop = why == nullptr;
        res->amb_crop_asked  = true;
        res->amb_cropped     = crop;
        res->amb_crop_reason = why;
        res->amb_crop_rows   = (int32_t) n_outputs;
        res->amb_crop_total  = (int32_t) n_tokens;
        // C17b: an uncropped graph whose MoE is offloaded needs a device copy of a host expert tensor that the cropped
        // reservation does not include (process_ubatch logs it once before the allocation grows the buffer)
        if (!crop && why != nullptr && strncmp(why, "below the offload", 17) != 0 && strncmp(why, "op offload", 10) != 0 &&
                strncmp(why, "no outputs", 10) != 0) {
            size_t bytes = 0;
            res->amb_crop_offload_uncropped = amb_mtp_moe_offloaded(
                { layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps, layer.ffn_gate_up_exps }, &bytes);
            res->amb_crop_copy_bytes = bytes;
        }
    }
    if (crop) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        cb(cur, "mtp_crop_attn_out", il);
        if (attn_mix.inject != nullptr) {
            attn_mix.inject = ggml_get_rows(ctx0, attn_mix.inject, inp_out_ids);
            cb(attn_mix.inject, "mtp_crop_inject", il);
        }
        res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
        res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
        res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        cb(res_hc, "mtp_crop_res", il);

        static bool logged = false;
        if (!logged) {
            logged = true;
            LLAMA_LOG_INFO("%s: amb_mtp_crop: first cropped draft graph: FFN/MoE and head on %d of %d rows (n_outputs of "
                    "n_tokens; reserve graphs count)\n", __func__, (int) n_outputs, (int) n_tokens);
        }
    }

    res_hc = build_hc_combine(res_hc, cur, attn_mix.inject, il);

    hc_mix ffn_mix = build_hc_mix(res_hc, layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject, layer.hc_ffn_down_inject, il);
    cur    = ffn_mix.cur;
    cur    = build_layer_ffn(cur, il);
    res_hc = build_hc_combine(res_hc, cur, ffn_mix.inject, il);

    // the next draft step reads this residual as its h
    // cropped: res_hc already holds only the output rows (in out_ids order), and the blocker guarantees masked extraction
    ggml_tensor * flat     = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
    ggml_tensor * flat_out = crop ? flat : (inp_out_ids ? ggml_get_rows(ctx0, flat, inp_out_ids) : flat);
    res->t_h_nextn = cparams.embeddings_nextn_masked ? flat_out : flat;
    cb(res->t_h_nextn, "h_nextn", il);
    ggml_build_forward_expand(gf, res->t_h_nextn);

    hc_mix head_mix = build_hc_mix(ggml_reshape_3d(ctx0, flat_out, n_embd, hc, flat_out->ne[1]),
            layer.nextn.hc_head_norm, layer.nextn.hc_head_down, layer.nextn.hc_head_up,
            nullptr, nullptr, il);
    cur = head_mix.cur;
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(head.output, cur, head.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    // AMB_MTP_PRUNE=1 (C15): a ubatch with no outputs (prompt pass, verify catch-up) only has to leave the draft's state
    // behind: its K/V and indexer rows (raw keys and the pooled keys it completes). Everything after those writes
    // (attention, FFN/MoE, h_nextn, head) is dropped, see llm_graph_context::amb_prune_to_state_writes. C14b: only
    // requested here; llama_model::build_graph prunes at its end (after the sampling chains and set_outputs)
    amb_prune_requested = amb_mtp_prune_wanted();
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen4exp::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    if (model.layers[il].wqkv_z) {
        // qkv and z read the same input: one matmul, then the qkv rows and the z rows
        ggml_tensor * qkvz = build_lora_mm(model.layers[il].wqkv_z, input);
        cb(qkvz, "linear_attn_qkvz", il);

        const int64_t n_z   = hparams.ssm_d_state * hparams.ssm_dt_rank;
        const int64_t n_qkv = qkvz->ne[0] - n_z;

        ggml_tensor * qkv_mixed = ggml_view_3d(ctx0, qkvz, n_qkv, n_seq_tokens, n_seqs, qkvz->nb[1], qkvz->nb[1] * n_seq_tokens, 0);
        cb(qkv_mixed, "linear_attn_qkv_mixed", il);

        ggml_tensor * z = ggml_view_2d(ctx0, qkvz, n_z, n_seq_tokens * n_seqs, qkvz->nb[1], ggml_row_size(qkvz->type, n_qkv));
        cb(z, "z", il);

        return { qkv_mixed, z };
    }

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen4exp::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    // the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not silu
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated = ggml_sigmoid(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated);
}

// QSA k-pool inputs, shared by the QSA layers: blocks of compress_ratio cells in sequence order, see llama_memory_hybrid_idx
class llama_model_qwen4exp::llm_graph_input_kpool : public llm_graph_input_i {
public:
    llm_graph_input_kpool(const llama_memory_hybrid_idx_context * mctx, uint32_t kpool) : mctx(mctx), kpool(kpool) {}
    virtual ~llm_graph_input_kpool() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        mctx->set_input_kpool(pool_cells, pool_idxs, pool_mask, tail_idxs, nullptr, new_pool_idxs, new_pool_rep,
                              ubatch, new_pool_pos);
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        bool res = true;

        res &= k_idxs->ne[0]     == params.ubatch.n_tokens;
        res &= pool_cells->ne[0] == mctx->get_n_kpool();
        res &= pool_mask->ne[1]  == params.ubatch.n_tokens;
        res &= tail_idxs->ne[1]  == params.ubatch.n_tokens;
        // the scatter mask shape follows n_kv
        res &= n_kv              == idx->get_n_kv();
        res &= n_new             == mctx->get_n_kpool_new();

        return res;
    }

    ggml_tensor * k_idxs        = nullptr; // I64 [n_tokens]
    ggml_tensor * pool_cells    = nullptr; // I32 [n_pool]         cell caching each block's pooled key
    ggml_tensor * pool_idxs     = nullptr; // I32 [kpool, n_pool]  member cells per block, n_kv sentinel for the padded blocks
    ggml_tensor * pool_mask     = nullptr; // F16 [n_pool, n_tokens]
    ggml_tensor * tail_idxs     = nullptr; // I32 [kpool - 1, n_tokens]
    ggml_tensor * new_pool_idxs = nullptr; // I32 [kpool, n_new]   members of the blocks to re-pool this ubatch
    ggml_tensor * new_pool_flat = nullptr; // I32 [kpool*n_new]    view of new_pool_idxs, shared by the QSA layers
    ggml_tensor * new_pool_rep  = nullptr; // I64 [n_new]          cell to write each new pooled key into
    ggml_tensor * new_pool_pos  = nullptr; // I32 [4*n_new]        M-RoPE position of each new block's first member

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t kpool;
    uint32_t n_new = 0; // padded to a stable bound, never below 1
    uint32_t n_sel = 0;
    uint32_t n_kv  = 0;
};

llama_model_qwen4exp::llm_graph_input_kpool * llama_model_qwen4exp::graph::build_inp_kpool(const llama_memory_hybrid_idx_context * mctx_hyb) {
    const auto * mctx_idx = mctx_hyb->get_idx();
    GGML_ASSERT(mctx_idx != nullptr);

    const uint32_t kpool  = hparams.indexer_kpool;
    const uint32_t n_pool = mctx_hyb->get_n_kpool();

    auto inp = std::make_unique<llm_graph_input_kpool>(mctx_hyb, kpool);

    inp->k_idxs     = mctx_idx->build_input_k_idxs(ctx0, ubatch);
    inp->pool_cells = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pool);
    inp->pool_idxs  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool, n_pool);
    inp->pool_mask  = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_pool, n_tokens);
    inp->tail_idxs  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool - 1, n_tokens);
    ggml_set_input(inp->pool_cells);
    ggml_set_input(inp->pool_idxs);
    ggml_set_input(inp->pool_mask);
    ggml_set_input(inp->tail_idxs);

    // set_input fills them all, so keep them allocated even when no op reads them
    ggml_build_forward_expand(gf, inp->pool_cells);
    ggml_build_forward_expand(gf, inp->pool_idxs);
    ggml_build_forward_expand(gf, inp->pool_mask);
    ggml_build_forward_expand(gf, inp->tail_idxs);

    inp->n_kv  = mctx_idx->get_n_kv();
    inp->n_new = mctx_hyb->get_n_kpool_new();
    // the top blocks plus the tail
    inp->n_sel = kpool*std::min<uint32_t>(n_pool, hparams.indexer_top_k / kpool) + kpool - 1;

    inp->new_pool_idxs = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool, inp->n_new);
    ggml_set_input(inp->new_pool_idxs);
    // one view for all layers: each view of a host input is a separate upload
    inp->new_pool_flat = ggml_reshape_1d(ctx0, inp->new_pool_idxs, kpool*inp->n_new);
    // one scatter row per new pool, each a distinct rep row (see kpool_build_state)
    inp->new_pool_rep = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, inp->n_new);
    ggml_set_input(inp->new_pool_rep);
    inp->new_pool_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*inp->n_new);
    ggml_set_input(inp->new_pool_pos);

    return (llm_graph_input_kpool *) res->add_input(std::move(inp));
}

// QSA attends to the top blocks of compress_ratio cells plus the incomplete tail, like the glm5-next k-pool indexer
// a block is scored by one pooled key: the mean of its raw indexer keys, normed and rotated to its first member
ggml_tensor * llama_model_qwen4exp::graph::build_qsa_sel(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        llm_graph_input_kpool *                 inp_kpool,
        ggml_tensor *                           cur,
        ggml_tensor *                           inp_pos,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim = hparams.indexer_head_size;
    const int64_t n_idx_h = hparams.indexer_n_head;
    const int64_t kpool   = inp_kpool->kpool;
    const int64_t n_pool  = inp_kpool->pool_cells->ne[0];
    const int64_t n_new   = inp_kpool->n_new;

    GGML_ASSERT(hparams.dsv4_compress_ratios[il] == kpool);

    // cache rows store raw key | pooled key: pooling precedes norm and rotation, so the raw key gets neither
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    cb(k_raw, "indexer_k_raw", il);

    ggml_tensor * pzero  = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, idx_dim, n_tokens), 0.0f);
    ggml_tensor * packed = ggml_reshape_3d(ctx0, ggml_concat(ctx0, k_raw, pzero, 0), 2*idx_dim, 1, n_tokens);
    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, packed, inp_kpool->k_idxs, il));

    // the raw keys and the persistent pooled slots, see llama_memory_hybrid_idx::mem_idx_stale
    auto kpool_cache = mctx_hyb->get_kpool_access(ctx0, il, idx_dim);

    // pool only the blocks this ubatch completes or regroups
    ggml_tensor * rows = kpool_cache.gather_key_gate(inp_kpool->new_pool_flat);
    rows = ggml_reshape_3d(ctx0, rows, idx_dim, kpool, n_new);

    // mean over the members; kpool is small, so summing slices beats a transpose plus sum_rows
    ggml_tensor * pooled_new = nullptr;
    for (int64_t i = 0; i < kpool; ++i) {
        ggml_tensor * slice = ggml_view_2d(ctx0, rows, idx_dim, n_new, rows->nb[2], i*rows->nb[1]);
        pooled_new = pooled_new ? ggml_add(ctx0, pooled_new, slice) : ggml_cont(ctx0, slice);
    }
    pooled_new = ggml_scale(ctx0, pooled_new, 1.0f/(float) kpool);
    pooled_new = build_norm(pooled_new, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);

    pooled_new = ggml_reshape_3d(ctx0, pooled_new, idx_dim, 1, n_new);
    pooled_new = build_rope(pooled_new, inp_kpool->new_pool_pos, il);
    pooled_new = ggml_reshape_2d(ctx0, pooled_new, idx_dim, n_new);
    cb(pooled_new, "indexer_pool_k_new", il);

    // scatter the fresh pooled keys into their rep rows, then gather all n_pool of them by cell:
    // the older pools come from the rows earlier ubatches wrote
    ggml_build_forward_expand(gf, kpool_cache.scatter_pooled(pooled_new, inp_kpool->new_pool_rep));
    ggml_tensor * pooled = kpool_cache.gather_pooled(inp_kpool->pool_cells);
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, n_pool);
    cb(pooled, "indexer_k", il);

    ggml_tensor * q = build_lora_mm(model.layers[il].index_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = build_norm(q, model.layers[il].index_q_norm, nullptr, LLM_NORM_RMS, il);
    q = build_rope(q, inp_pos, il);
    cb(q, "indexer_q", il);

    // the reference sums the rectified head scores unweighted, scaled by 1/sqrt(head_dim),
    // which is the lightning indexer with every head weight set to that scale
    ggml_tensor * weights = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_idx_h, n_tokens), 1.0f/sqrtf((float) idx_dim));
    ggml_tensor * score = ggml_lightning_indexer(ctx0, q, pooled, weights, inp_kpool->pool_mask); // [n_pool, n_tokens]
    cb(score, "indexer_score", il);

    const int64_t n_top_pool = std::min<int64_t>(n_pool, hparams.indexer_top_k / kpool);
    ggml_tensor * top_k = ggml_top_k(ctx0, score, n_top_pool); // [n_top_pool, n_tokens], unordered
    cb(top_k, "indexer_top_k", il);

    // flash attention requires an f16 mask
    const ggml_type type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    // keep the cells of the live picked pools and the tail, -inf elsewhere
    // all retained cells are in the token's sequence, so the kq_mask is not needed
    if (cparams.fused_top_k_mask) {
        GGML_ASSERT(kpool*n_top_pool + inp_kpool->tail_idxs->ne[0] == inp_kpool->n_sel);

        ggml_tensor * keep = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, type_mask, inp_kpool->n_kv), 0.0f);
        ggml_tensor * sel  = ggml_top_k_mask(ctx0, keep, top_k, score, inp_kpool->pool_idxs, inp_kpool->tail_idxs); // [n_kv, n_tokens]
        res->add_fused_node({LLM_FUSED_OP_TOP_K_MASK, sel, il});
        cb(sel, "indexer_sel", il);

        return sel;
    }

    // the top blocks, then the incomplete tail with n_kv for missing cells
    ggml_tensor * sel_idx = ggml_get_rows(ctx0, inp_kpool->pool_idxs,
            ggml_reshape_1d(ctx0, top_k, n_top_pool*n_tokens)); // [kpool, n_top_pool*n_tokens]
    sel_idx = ggml_reshape_2d(ctx0, sel_idx, kpool*n_top_pool, n_tokens);
    sel_idx = ggml_concat(ctx0, sel_idx, inp_kpool->tail_idxs, 0);
    const int64_t n_sel = sel_idx->ne[0];
    GGML_ASSERT(n_sel == inp_kpool->n_sel);

    ggml_build_forward_expand(gf, sel_idx);

    // TODO: figure out to reduce the large copmute buffer that this creates

    // scatter zeros for the selected cells into an all -inf row, each dead slot into its own dump row n_kv + slot
    // seeding from sel_idx ties the scatter storage lifetime to this layer
    const int64_t n_kv = inp_kpool->n_kv;

    ggml_tensor * mask_all = ggml_new_tensor_4d(ctx0, type_mask, n_kv + n_sel, 1, 1, 1);
    mask_all = ggml_fill(ctx0, mask_all, -INFINITY);
    mask_all = ggml_repeat_4d(ctx0, mask_all, n_kv + n_sel, n_tokens, 1, 1);
    mask_all = ggml_reshape_3d(ctx0, mask_all, 1, n_kv + n_sel, n_tokens);

    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, type_mask, n_sel, 1, 1, 1);
    zeros = ggml_fill(ctx0, zeros, 0.0f);
    zeros = ggml_repeat_4d(ctx0, zeros, n_sel, n_tokens, 1, 1);
    zeros = ggml_reshape_3d(ctx0, zeros, 1, n_sel, n_tokens);

    // live slots address disjoint cells, but padded pools and missing tail cells share the n_kv sentinel, and
    // top_k fills a short selection with invisible pools that can overlap the tail, so the scatter would write
    // some cells from several threads: map every dead slot to its own dump row, idx = dump + live*(idx - dump)
    // a picked pool is live when visible: a visible score is a rectified sum >= 0, an invisible one is -inf
    ggml_tensor * top_score = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, score, 1, n_pool, n_tokens), top_k); // [1, n_top_pool, n_tokens]
    ggml_tensor * live_pool = ggml_clamp(ctx0, ggml_scale_bias(ctx0, top_score, 1.0f, 1.0f), 0.0f, 1.0f);
    live_pool = ggml_reshape_2d(ctx0, ggml_repeat_4d(ctx0, live_pool, kpool, n_top_pool, n_tokens, 1), kpool*n_top_pool, n_tokens);
    // a tail cell is live unless it is the n_kv sentinel
    ggml_tensor * live_tail = ggml_cast(ctx0, inp_kpool->tail_idxs, GGML_TYPE_F32);
    live_tail = ggml_clamp(ctx0, ggml_scale_bias(ctx0, live_tail, -1.0f, (float) n_kv), 0.0f, 1.0f);
    ggml_tensor * live = ggml_concat(ctx0, live_pool, live_tail, 0); // [n_sel, n_tokens]

    // dump rows n_kv + slot as a cumulative sum: the meta backend cannot split an arange, which has no source
    ggml_tensor * dump  = ggml_scale_bias(ctx0, ggml_cumsum(ctx0, ggml_fill(ctx0, live, 1.0f)), 1.0f, (float) (n_kv - 1));
    ggml_tensor * idx_f = ggml_cast(ctx0, sel_idx, GGML_TYPE_F32);
    idx_f   = ggml_add(ctx0, ggml_mul(ctx0, ggml_sub(ctx0, idx_f, dump), live), dump);
    sel_idx = ggml_cast(ctx0, idx_f, GGML_TYPE_I32);

    ggml_tensor * sel = ggml_set_rows(ctx0, mask_all, zeros, ggml_reshape_3d(ctx0, sel_idx, n_sel, n_tokens, 1));

    sel = ggml_view_2d(ctx0, sel, n_kv, n_tokens, sel->nb[2], 0);
    sel = ggml_cont(ctx0, sel);
    cb(sel, "indexer_sel", il);

    return sel;
}

// Dense GQA self-attention over the cells that the QSA mask keeps.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        ggml_tensor *             sel,
        int64_t                   n_sel,
        float                     kq_scale,
        int                       il) {
    // rotate q/k/v before they reach a quantized cache, as the dense path does. the indexer
    // has already scored with its own query in build_qsa_sel, so the selection is unaffected.
    if (inp->self_k_rot[il]) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot[il]);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot[il]);
    }

    if (inp->self_v_rot[il]) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot[il]);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    // the selection replaces the kq_mask: split it into the streams of the K view
    const int64_t n_stream = k->ne[3];
    GGML_ASSERT(sel->ne[0] == k->ne[2] && sel->ne[1] % n_stream == 0);
    sel = ggml_reshape_4d(ctx0, sel, sel->ne[0], sel->ne[1]/n_stream, 1, n_stream);
    cb(sel, "kq_mask_qsa", il);

    ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, sel, nullptr, nullptr, nullptr, n_sel, kq_scale, il);
    cb(cur, "kqv_out", il);

    // the rotation is its own inverse, so undo it on the value side of the output
    if (inp->self_v_rot[il]) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot[il]);
    }

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        llm_graph_input_kpool *   inp_kpool,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // indexer reads the same block input as q/k/v; no cache or no ratio means dense
    const bool qsa = inp_kpool != nullptr && hparams.dsv4_compress_ratios[il] > 0;

    ggml_tensor * sel = qsa ? build_qsa_sel(mctx_hyb, inp_kpool, cur, inp_pos, il) : nullptr;

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = build_rope(Qcur, inp_pos, il);
    Kcur = build_rope(Kcur, inp_pos, il);

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (sel) {
        cur = build_attn_qsa(inp, Qcur, Kcur, Vcur, sel, inp_kpool->n_sel, kq_scale, il);
    } else {
        cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = hparams.ssm_d_state;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
    GGML_ASSERT(head_v_dim * num_v_heads == d_inner);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta  = nullptr;
    ggml_tensor * alpha = nullptr;
    if (model.layers[il].ssm_beta_alpha_cat) {
        // beta and alpha read the same input: one matmul, then the beta rows and the alpha rows
        ggml_tensor * ba = build_lora_mm(model.layers[il].ssm_beta_alpha_cat, cur);
        cb(ba, "beta_alpha", il);

        beta  = ggml_view_4d(ctx0, ba, 1, num_v_heads, n_seq_tokens, n_seqs,
                ggml_element_size(ba), ba->nb[1], ba->nb[1] * n_seq_tokens, 0);
        alpha = ggml_view_3d(ctx0, ba, num_v_heads, n_seq_tokens, n_seqs,
                ba->nb[1], ba->nb[1] * n_seq_tokens, ggml_row_size(ba->type, num_v_heads));
    } else {
        beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
        beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);

        alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
        alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    }
    cb(beta, "beta", il);
    cb(alpha, "alpha", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];

    // the channels must match how load_arch_tensors sizes wqkv, not ssm_d_inner
    const int64_t conv_channels    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    ggml_tensor * conv_input = build_conv_state_at(inp, conv_states_all, qkv_mixed,
            conv_kernel_size - 1, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, conv_channels);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    ggml_tensor * z_2d = ggml_view_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(z->type, head_v_dim), z->nb[1], z->nb[1] * n_seq_tokens, 0);

    // gated normalization, as self.norm(core_attn_out, z) in the reference
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    ggml_tensor * final_output = ggml_reshape_2d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens * n_seqs);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_shexp(ggml_tensor * cur, const int il) {
    if (model.layers[il].ffn_up_shexp == nullptr) {
        return nullptr;
    }
    ggml_tensor * ffn_shexp =
        build_ffn(cur,
            model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
            model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
            model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
            NULL,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(ffn_shexp, "ffn_shexp", il);

    // shared expert has its own sigmoided gate (ffn_gate_inp_shexp, one value per token)
    ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
    cb(shared_gate, "shared_expert_gate", il);

    shared_gate = ggml_sigmoid(ctx0, shared_gate);
    cb(shared_gate, "shared_expert_gate_sigmoid", il);

    ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
    cb(ffn_shexp, "ffn_shexp_gated", il);

    return ffn_shexp;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(ggml_tensor * cur, const int il, ggml_tensor * const * shexp_pre) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps);
    cb(moe_out, "ffn_moe_out", il);

    // shared experts, as in the Qwen3Next reference (AMB_MOE_MISS_OVERLAP: built before the slot lookup, see build_shexp)
    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp = shexp_pre != nullptr && *shexp_pre != nullptr ? *shexp_pre : build_shexp(cur, il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}

// PLE n-gram hash embedding: each token gathers ple_n_heads rows of a shared table.
//   mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]);  row = mixed_n % vocab[h] + offset[h]
// The hash runs host-side because ggml has no int64 and no xor. EOS resets the window.

class llm_graph_input_qwen4exp_ple : public llm_graph_input_i {
public:
    llm_graph_input_qwen4exp_ple(const llama_model & model,
                        const llama_kv_cache_context * mctx) : model(model), mctx(mctx) {}
    virtual ~llm_graph_input_qwen4exp_ple() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx)->get_attn();
        return rows->ne[0] == (int64_t) model.hparams.ple_n_heads * params.ubatch.n_tokens;
    }

    ggml_tensor * rows = nullptr;   // I32 [ple_n_heads * n_tokens]

    const llama_model & model;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_context * mctx;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
};

void llm_graph_input_qwen4exp_ple::set_input(const llama_ubatch * ubatch) {
    const auto & hparams = model.hparams;

    // an image arrives as an embd batch, so ubatch->token is null, but every position still needs a row for ggml_get_rows
    // stand in the image token id that the reference hashes, or EOS if the file has no such key
    // gemma3n and gemma4 do the same with a hardcoded row 0 of per_layer_token_embd.
    const llama_token img_tok = hparams.ple_image_token_id != 0
        ? (llama_token) hparams.ple_image_token_id
        : (llama_token) hparams.ple_eos_token_id;
    auto tok_of = [&](int64_t k) -> llama_token {
        const bool is_embd = !ubatch->token || (ubatch->is_mixed() && ubatch->type[k]);
        return is_embd ? img_tok : ubatch->token[k];
    };

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hparams.ple_ngram_size;
    const int64_t n_heads  = hparams.ple_n_heads;
    const int64_t per_gram = hparams.ple_heads_per_ngram;
    const int64_t eos      = hparams.ple_eos_token_id;
    const int64_t n_prev   = n_gram - 1;

    std::vector<int32_t> idx(n_heads * n_tokens);

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "PLE n-gram embeddings do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() already stored this ubatch, so its own tokens count too
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // an EOS in the window resets everything at or before it
        // a missing predecessor (before the sequence start, or no cached cell) reads as EOS
        // the EOS of the token itself does not cut its own context, as in the reference
        std::vector<int64_t> ctx(n_gram);
        ctx[0] = tok_of(i);
        bool cut = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            const llama_token t = cut ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            cut = cut || t < 0 || t == eos;
            ctx[s] = cut ? eos : t;
        }

        for (int64_t n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * hparams.ple_layer_multipliers[0];
            for (int64_t j = 1; j < n; ++j) {
                mixed ^= (uint64_t) ctx[j] * hparams.ple_layer_multipliers[j];
            }
            const int64_t base = (n - 2) * per_gram;
            for (int64_t g = 0; g < per_gram; ++g) {
                const int64_t h_i = base + g;
                idx[i * n_heads + h_i] =
                    (int32_t) (mixed % hparams.ple_head_vocab_sizes[h_i] + hparams.ple_head_offsets[h_i]);
            }
        }
    }

    {
        ggml_tensor * ple = model.per_layer_tok_embd;

        const bool prefetch = model.can_prefetch.count(ple);
        if (prefetch) {
            llama_prefetch_rows(ple, idx.data(), idx.size());
        }
    }

    ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
}

// Read a conv history out of its own recurrent row and write the new tail back.
// The shared build_conv_state cannot do this: qwen4exp has two such rows per layer.
ggml_tensor * llama_model_qwen4exp::graph::build_conv_state_at(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        x,
        int64_t              state_cols,
        int64_t              channels,
        int                  il) {
    const int64_t n_seqs    = ubatch.n_seqs;
    const int64_t row_total = conv_states_all->ne[0];

    // the row is exactly this convolution's state, so the gather is reused as a whole
    GGML_ASSERT(state_cols * channels == row_total);

    auto it = rs_rows.find(conv_states_all);
    if (it == rs_rows.end()) {
        it = rs_rows.emplace(conv_states_all, build_rs(inp, conv_states_all, row_total, n_seqs)).first;
    }
    ggml_tensor * rows = it->second;

    ggml_tensor * state = ggml_reshape_3d(ctx0, rows, state_cols, channels, n_seqs);
    cb(state, "conv_state_at", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, state, ggml_transpose(ctx0, x), 0);

    build_conv_state_store(inp, conv_states_all, conv_input, state_cols, 0);

    return conv_input;
}

ggml_tensor * llama_model_qwen4exp::graph::build_inp_ple(
        const llama_memory_hybrid_idx_context * mctx_hyb) {
    const int64_t n_heads = hparams.ple_n_heads;

    // the attention cells see every ubatch regardless of the layer types
    auto ple_inp = std::make_unique<llm_graph_input_qwen4exp_ple>(model, mctx_hyb->get_attn());

    ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
    ggml_set_input(ple_inp->rows);
    ggml_tensor * rows = ple_inp->rows;
    res->add_input(std::move(ple_inp));

    // gather then flatten the heads: get_rows lays the head dimension out slowest, as the reference does
    ggml_tensor * emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, rows);
    if (model.per_layer_tok_embd_s) {
        emb = ggml_scale(ctx0, emb, model.per_layer_tok_embd_scale);
    }
    emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
    cb(emb, "ple_embd", -1);

    return emb;
}

ggml_tensor * llama_model_qwen4exp::graph::build_ple(
        llm_graph_input_rs * inp,
        ggml_tensor *        emb,
        ggml_tensor *        hidden,
        int                  il) {
    const int64_t hc      = hparams.dsv4_hc_mult;
    const int64_t hc_dim  = hc * n_embd;

    ggml_tensor * key   = build_lora_mm(model.layers[il].ple_key,   emb);
    ggml_tensor * value = build_lora_mm(model.layers[il].ple_value, emb);

    // both norms group over one hc stream, with a [n_embd, hc] weight
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens);
        return ggml_mul(ctx0, ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps), w);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);

    // per-stream dot product, then a signed square root before the sigmoid
    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    s = ggml_scale(ctx0, s, 1.0f / sqrtf((float) n_embd));

    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "ple_gate", il);

    // [n_embd, 1, T] value broadcast across the hc streams, scaled by the gate
    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);

    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);

    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens),
            model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);

    // depthwise causal conv, dilated by the n-gram size, as a sum of shifted copies
    // ggml_conv_1d_dw is documented as unreliable:
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    // The history of the earlier ubatches is prepended, so a chunked prefill matches a single-shot one.
    const int64_t kern = hparams.ple_conv_kernel;
    const int64_t dil  = hparams.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    // the conv history is per sequence, so the input carries the sequence axis too
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    // [hist + n_seq_tokens, hc_dim, n_seqs], tokens on ne[0]
    ggml_tensor * padded = build_conv_state_at(inp, inp->mctx->get_p_l(il),
            ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs),
            hist, hc_dim, il);

    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        // tap k reads (kern-1-k)*dilation positions back
        const int64_t start = hist - (kern - 1 - k) * dil;

        ggml_tensor * shifted = ggml_cont(ctx0,
                ggml_transpose(ctx0,
                        ggml_view_3d(ctx0, padded, n_seq_tokens, hc_dim, n_seqs,
                                padded->nb[1], padded->nb[2],
                                ggml_row_size(padded->type, start))));

        // column k of the [kern, hc_dim] kernel is one weight per channel
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1],
                        k * model.layers[il].ple_conv1d->nb[0]));
        // this kernel keeps the file type, so cast it before it multiplies an f32 activation
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }

        ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }

    conv_out = ggml_silu(ctx0, conv_out);
    conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, n_tokens);
    cb(conv_out, "ple_conv_out", il);

    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
}
