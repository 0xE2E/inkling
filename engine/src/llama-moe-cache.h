#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;
struct llm_graph_context;

// the experts of a MoE layer, as llm_graph_context::build_moe_ffn passes them to the MoE cache
struct llama_moe_cache_layer {
    int32_t       il;
    ggml_tensor * up_exps;
    ggml_tensor * gate_exps;
    ggml_tensor * down_exps;
    ggml_tensor * gate_up_exps;
    bool          weight_before_ffn; // the router weights scale the expert input
};

// the lookup of the selected experts of a layer in the MoE cache, see llama_moe_cache::build_lookup
struct llama_moe_cache_lookup {
    int32_t       il      = -1;
    ggml_tensor * slots   = nullptr; // [n_expert_used, n_tokens] slots of the selected experts in the MoE cache, nullptr if the layer does not read the cache
    ggml_tensor * cpu_ids = nullptr; // [n_expert_used, n_tokens] misses computed from the host experts, -1 for the other experts, or nullptr
    // AMB_MOE_MISS_OVERLAP=2 (C24): with it, slots are the hit-phase slots and slots_miss the miss-phase slots behind
    // the layer's gate, both masked (-1 where the phase does not compute the pair); nullptr otherwise
    ggml_tensor * slots_miss = nullptr;
};

// builds the experts selected by mm_ids up to the down projection, from the MoE cache if cached: [n_embd, n_expert_used, n_tokens]
// cur is the expert input and ids are the selected experts, for the biases and scales (see llm_graph_context::build_moe_ffn)
using llama_moe_cache_build_experts_t = ggml_tensor * (*)(ggml_tensor * cur, ggml_tensor * ids, ggml_tensor * mm_ids, bool cached, void * user_data);
class llama_elastic_lender;

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // the largest ubatch (tokens) that reads its experts through the cache slot maps; larger ubatches use the scheduler's
    // expert copies (and AMB_PF_OVERLAP's ring for exactly these ubatches): the one definition of "large batch"
    static constexpr int64_t max_batch = 32;

    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // cpu_share is the share of the misses computed on the CPU instead of uploaded
    // elastic_layout (AMB_ELASTIC_KV=3, the same in the --fit pass): banks on elastic memory; elastic_lend_bytes: the KV chunks
    // the banks may borrow on top of `size`, see llama-elastic.h
    llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share, bool elastic_layout = false, size_t elastic_lend_bytes = 0);
    ~llama_moe_cache();

    // AMB_ELASTIC_KV=3: the banks are elastic and can take lent KV chunks, nullptr otherwise
    llama_elastic_lender * elastic_lender();
    // the banks are on elastic memory (AMB_ELASTIC_KV=3), the same in the --fit pass
    bool elastic_layout() const;
    // C4b (AMB_ELASTIC_FLOOR_MIB): true if a group has fewer active slots than the graphs count on (the compute floor is
    // not lent); the floor in bytes (0 = off or no elastic banks)
    bool   elastic_floor_short() const;
    size_t elastic_floor_bytes() const;

    // AMB_PF_OVERLAP with elastic banks (C13): the overlap ring lives in a fixed range of the own bank slots, lent only
    // for the duration of a large-batch graph. ring_plan: up to n_want ring slots of slot_size bytes, returns how many;
    // ring_ptr(k): their fixed device addresses in ring_buffer(); ring_borrow parks the range (returns the experts
    // dropped), ring_return clears the head of every ring-range slot on the compute stream (C13c, MMQ over-reads into the
    // next slot) and makes the slots active and empty again, returns the slots unparked; no KV growth or shrink in between
    int32_t ring_plan(size_t slot_size, int32_t n_want);
    ggml_backend_buffer_t ring_buffer() const;
    void *  ring_ptr(int32_t k) const;
    size_t  ring_borrow();
    int64_t ring_return();
    int64_t ring_slots_parked() const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // the device that caches layer il (kept public for the AMB graph hooks of the models, e.g. the lookahead predictor
    // publish of qwen4exp, which must run on the backend of the cache; upstream b12588 made it private)
    ggml_backend_t backend(int32_t il) const;

    // the graph of a MoE layer, built by llm_graph_context::build_moe_ffn in two steps:

    // 1. the lookup of the selected experts in the cache, built after the expert input and before the expert matmuls
    //    returns no slots if the layer does not read its experts from the cache for this ubatch
    llama_moe_cache_lookup build_lookup(const llm_graph_context & g, const llama_moe_cache_layer & layer, ggml_tensor * selected_experts) const;

    // 2. the experts of a layer with slots, from the cache and from the host experts: [n_embd, n_expert_used, n_tokens]
    // calls build synchronously; does not retain user_data
    ggml_tensor * build_experts(const llm_graph_context & g, const llama_moe_cache_lookup & lookup,
            ggml_tensor * cur, ggml_tensor * selected_experts, llama_moe_cache_build_experts_t build, void * user_data) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // AMB_IDS_ASYNC=1 (C14, default off), see llama-moe-cache.cpp: enabled (all preconditions met at init)
    bool ids_async_enabled() const;
    // from the scheduler's split callback, after a split was launched: enqueue the ids readback of the next split's lookup
    void ids_post_split(const ggml_backend_sched_split_info * info);
    // before a graph evaluation: forget pending readbacks
    void ids_async_reset();

    // AMB_STEP_EVENTS (C14): called by copy() right before (which = 0) and after (which = 1) the miss uploads are enqueued
    // on the compute stream; with AMB_MOE_MISS_OVERLAP (C23) which = 2 / 3 and backend is the upload stream's backend
    void set_copy_mark_hook(void (*fn)(void * ud, ggml_backend_t backend, int which), void * ud);

    // AMB_MOE_MISS_OVERLAP=1 (C23, default off), see llama-moe-cache.cpp
    // the graph publishes the selected ids of layer il (n_tokens rows) right after its router and expands the work that
    // does not depend on its experts before its slot lookup; a pure function of the configuration, the cache layout and
    // n_tokens (false with the switch unset)
    bool miss_overlap(int32_t il, int64_t n_tokens) const;
    // the GGML_OP_AMB_PUBLISH node of layer il (kind 0: the selected ids, 1: the predicted ids of layer il)
    ggml_tensor * miss_overlap_publish(ggml_context * ctx, ggml_tensor * ids, int32_t il, int kind) const;
    // the runtime part is active (llama_context then installs the scheduler's no-sync callback)
    bool miss_overlap_enabled() const;
    // AMB_MOE_HIT_FIRST (D2): the runtime part is active (the scheduler's no-sync callback is needed for it too)
    bool hit_first_enabled() const;
    // for the scheduler: the copy callback of this slot map orders itself (a publish of its ids is pending)
    bool copy_nosync(const ggml_tensor * src) const;
    // C24: turn the switch off before any graph is built (reason logged), e.g. when an eval callback is set
    void miss_overlap_refuse(const char * reason);
    // AMB_MOE_MISS_OVERLAP=2 (C24): the routed experts of layer il run in two phases (resident experts, then a gate on
    // the layer's miss uploads, then the missed ones); the I32 [2, n_expert] slot map the graph looks up instead of the
    // one of get_slot_map (row 0: hit-phase slot or -1, row 1: miss-phase slot or -1), nullptr if not split
    bool miss_overlap_split(int32_t il, int64_t n_tokens) const;
    ggml_tensor * get_slot_map2(int32_t il) const;
    // the GGML_OP_AMB_GATE node of layer il on the miss-phase ids
    ggml_tensor * miss_overlap_gate(ggml_context * ctx, ggml_tensor * ids, int32_t il) const;

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // AMB_MOE_TIMING (0 = off, 1, 2), only active together with AMB_MOE_STATS=N, see llama-moe-cache.cpp
    int  timing_mode() const;
    void timing_graph_begin();         // before a graph evaluation
    bool timing_graph_used() const;    // the graph just evaluated went through the cache with a slot map (a small batch)
    void timing_graph_end(int64_t ns); // host wall time of that graph evaluation, including the GPU sync added in timing mode

    // AMB_MOE_PREDICT=1 (measurement only), see llama-moe-cache.cpp: accuracy of early router evaluations
    static constexpr int predict_n_pred = 6; // predictors P0..P5
    static constexpr int predict_n_ids  = 24; // ids read per predictor, the widest predicted set
    // the graph adds the early router probes of layer il for a ubatch of n_tokens tokens
    bool predict_layer(int32_t il, int64_t n_tokens) const;
    void predict_begin(); // before a graph evaluation
    // after it: the predict_n_ids highest-probability ids of predictor p for layer il, in descending order
    void predict_add(int32_t il, int32_t p, const int32_t * ids);

    // AMB_MOE_PREFETCH=1|4 (prototype, default off), see llama-moe-cache.cpp: predictive expert prefetch for single-token ubatches
    // the predictor of layer il (1 = P1, 4 = P4) the graph builds during layer il - 1, 0 if none
    int  prefetch_mode(int32_t il, int64_t n_tokens) const;
    int  prefetch_k() const; // the predicted set size, the ne[0] of the I32 ids of a predictor
    // before a graph evaluation: the I32 [prefetch_k(), 1] predicted ids of layer il, a graph output computed before the slot
    // lookup of layer il - 1; copy() of layer il - 1 reads them back together with the real ids of layer il - 1
    // AMB_MOE_LOOKAHEAD (C18) also stores its predicted ids here (I32 [lookahead_k(), n_tokens] of layer il)
    void prefetch_set_ids(int32_t il, ggml_tensor * ids);
    void prefetch_clear(); // after the evaluation (the copy callbacks have run), forget the ids of the graph
    bool prefetch_ids_wanted() const; // AMB_MOE_PREFETCH or AMB_MOE_LOOKAHEAD keeps predicted ids (prefetch_set_ids)

    // AMB_MOE_LOOKAHEAD=1|4 (C18, default off), see llama-moe-cache.cpp: cross-layer predicted lookahead prefetch on a
    // second stream for ubatches of up to AMB_MOE_LOOKAHEAD_MAX_TOK (<= 8) tokens
    // the predictor (1 = P1, 4 = P4) the graph builds during layer il - 1 for layer il, 0 if none; a pure function of the
    // configuration, the cache layout and n_tokens (reserve, --fit and keep-region probe graphs get the same nodes)
    int  lookahead_mode(int32_t il, int64_t n_tokens) const;
    int  lookahead_k() const; // predicted experts per token: the ne[0] of the I32 ids of a predictor
    bool lookahead_rank() const;    // C18b AMB_MOE_LOOKAHEAD_RANK=1: the predictor's ids are ordered by router score
    bool lookahead_p4_fast() const; // C18b AMB_MOE_LOOKAHEAD_P4_FAST=1: the cheaper P4 mixer (predictor nodes only)
    // C18b: before a ubatch of n_tokens tokens: drain if it may take the MMQ path (more tokens than the smallest MMVQ batch
    // limit of the cached expert types, at most 8), see lookahead_drain
    void lookahead_drain_ubatch(int64_t n_tokens);
    // finish all lookahead uploads (synchronize the upload stream) and clear the in-flight and prefetched-unused state;
    // reason is counted in the amb_moe_lookahead line; a no-op (not counted) if nothing is in flight or unused
    void lookahead_drain(const char * reason);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
