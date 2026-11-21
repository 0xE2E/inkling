#pragma once

#include "ggml-backend.h"
#include "llama-arch.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// AMB_ELASTIC_KV (C10, default off): the KV and indexer caches on CUDA virtual memory ("elastic" buffers)
//   unset/0: the plain CUDA buffers, as before
//   1: the KV/indexer buffers are elastic buffers, fully mapped at load; nothing else changes
//   2: only the cells in use plus a margin are mapped in the KV/indexer buffers; the physical chunks of the other cells
//      stay mapped in a private parking ("spare") range of the same cache, so the VRAM use is unchanged, and move back
//      (zero filled) before any cell beyond the mapped ones is read or written
//   3: as 2, and the parked chunks are lent to the MoE expert cache at load (extra slots at the tails of its banks, see
//      llama-moe-cache.cpp); a KV growth takes chunks back from the bank tails (their slots are parked, the experts in
//      them dropped), a KV shrink at decode entry (hysteresis: at least 2 units) gives them back. Refused (level 2 with a
//      warning) with a cache CPU share, AMB_MOE_PREFETCH, LLAMA_MOE_CACHE_TRACE, no MoE cache, or more than one GPU
// AMB_ELASTIC_UNIT=N: the growth unit in cells (default 4096, rounded up to a multiple of 256, at least 256)
// with AMB_MOE_STATS=N: one "amb_elastic:" line per cache every N applied ubatches, besides the per-event lines
// supported only for the qwen4exp and inkling2 (C4: its non-SWA KV only) target contexts (not the MTP draft), with flash attention (V not transposed), one KV
// stream, the KV offloaded to CUDA devices with VMM; otherwise a warning and the plain buffers

// the level asked by the environment (0..3), read once
int llama_elastic_env_level();

// C4: the target architectures whose KV readers were audited for elastic KV memory (qwen4exp: hybrid_idx KV + indexer;
// inkling2: the non-SWA KV of llama_memory_hybrid_iswa, see docs/port-report.md "C4"). Every other context keeps the
// plain KV buffers, and its MoE cache (if any) has no elastic banks
bool llama_elastic_arch_ok(llm_arch arch);

// C4 verification modes (debug, default off, read once):
// AMB_ELASTIC_CHECK=1: host-side invariant checks: every applied ubatch, the cells a graph can read (n_kv, the cells in
//   use, the cells written) lie inside the chunks mapped in every KV tensor; after every MoE cache plan and every slot
//   activation/parking, every slot-map entry of the group points to an active (mapped, unparked) slot and no parked slot
//   holds an expert. Aborts on a violation; the counts are logged at teardown
// AMB_ELASTIC_POISON=1: memory that changes owner is filled with a marker instead of zeros: KV cells that become mapped
//   (load, growth, restore) get byte 0x41 (finite in f16/bf16/f32/q8_0: reads of them under the mask contribute exactly 0,
//   as stale cells do today, an unmasked read changes the output); MoE cache slots that become active through lending get
//   0xFF (NaN scales/values) except the first 512 bytes of each slot (the MMQ over-read zone, kept zero). Output must stay
//   identical to the run without poison; a difference or NaN means a reader touched memory it does not own
int llama_elastic_check_env();

// C4b AMB_ELASTIC_FLOOR_MIB=N (default 0 = off; needs AMB_ELASTIC_KV=3 and AMB_ELASTIC_COMPUTE=1): N MiB of the
// compute buffer lent at decode count as MoE cache slots the graphs may rely on (the cache's own memory can then be below
// its minimum, so --fit can keep a long context's KV on the GPU). The context lends the compute buffer before every small
// ubatch and refuses to load if less than N MiB is lendable; --fit counts the N MiB toward the cache minimum
size_t llama_elastic_floor_bytes(); // 0 if off or the two switches are not both on
size_t llama_elastic_span_bytes(size_t gran); // AMB_ELASTIC_COMPUTE_CHUNK_MIB rounded to whole chunks
int llama_elastic_poison_env();

// the request is a property of the memory being created: llama_context sets it around create_memory for the contexts
// it supports, and llama_kv_cache reads it in its constructor (thread local, so concurrent loads do not interfere)
struct llama_elastic_request_scope {
    explicit llama_elastic_request_scope(int level);
    ~llama_elastic_request_scope();
private:
    int prev;
};

int llama_elastic_requested();

// the elastic buffer type of the device of a plain CUDA device buffer type, nullptr if not available
ggml_backend_buffer_type_t llama_elastic_buft_of(ggml_backend_buffer_type_t buft);

// the elastic helpers of the CUDA backend (proc addresses of its registry); ok() is false if not available
struct llama_elastic_fns {
    ggml_backend_buffer_type_t (*buft_of)(ggml_backend_buffer_type_t) = nullptr;
    ggml_backend_buffer_t (*reserve)(ggml_backend_buffer_type_t, size_t) = nullptr;
    bool   (*is_elastic)(ggml_backend_buffer_t) = nullptr;
    size_t (*granularity)(ggml_backend_buffer_t) = nullptr;
    size_t (*buft_granularity)(ggml_backend_buffer_type_t) = nullptr;
    size_t (*n_chunks)(ggml_backend_buffer_t) = nullptr;
    size_t (*n_mapped)(ggml_backend_buffer_t) = nullptr;
    bool   (*chunk_mapped)(ggml_backend_buffer_t, size_t) = nullptr;
    bool   (*map_new)(ggml_backend_buffer_t, const size_t *, size_t) = nullptr;
    void   (*sync)(ggml_backend_buffer_t) = nullptr;
    bool   (*move)(ggml_backend_buffer_t, const size_t *, ggml_backend_buffer_t, const size_t *, size_t, bool) = nullptr;
    // C17c (optional, see spans_ok): whole-mapping moves for AMB_ELASTIC_COMPUTE
    ggml_backend_buffer_type_t (*span_buft)(ggml_backend_buffer_type_t, size_t) = nullptr;
    size_t (*span_of)(ggml_backend_buffer_t, size_t) = nullptr;
    bool   (*move_spans)(ggml_backend_buffer_t, const size_t *, ggml_backend_buffer_t, const size_t *, size_t, bool, size_t *) = nullptr;
    bool resolved = false;

    bool ok() const;
    bool spans_ok() const { return resolved && span_buft && span_of && move_spans; }
};

const llama_elastic_fns & llama_elastic_fns_of(ggml_backend_dev_t dev);

class llama_kv_elastic;

// level 3: the MoE expert cache takes the parked KV chunks into its bank tails
class llama_elastic_lender {
public:
    virtual ~llama_elastic_lender() = default;
    // take up to n chunks of src (an elastic buffer of the cache's device) into the bank tails; returns how many were
    // taken: chunks[0, result) were moved, the others stay in src. The caller guarantees no GPU work uses them.
    virtual size_t give(ggml_backend_buffer_t src, const size_t * chunks, size_t n) = 0;
    // move n lent chunks back to the unmapped chunks `chunks` of dst, zero filled (synchronizes the device first,
    // parks the bank slots on them); aborts if it cannot
    virtual void take(ggml_backend_buffer_t dst, const size_t * chunks, size_t n) = 0;
    // for the stats line
    virtual std::string describe() const = 0;
    // a KV cache that lent to it; told when the lender goes away (lender_gone)
    virtual void attach(llama_kv_elastic * borrower) = 0;
    // C17 stats: chunks of other buffers mapped in the banks now (all lenders together), active slots beyond the own
    // ones, experts dropped from parked slots so far
    virtual size_t   lent_chunks() const = 0;
    virtual int64_t  lent_slots()  const = 0;
    virtual uint64_t dropped()     const = 0;

    // C17c: whole mappings ("spans" of one or more chunks, one physical handle each) on top of the bank tails, for
    // AMB_ELASTIC_COMPUTE. give_spans takes up to n spans of src (heads = their first chunks), returns how many
    // ([0, result) moved, zero filled); take_spans moves every span back to the heads of dst (synchronizes the device,
    // parks the slots on them). Spans always sit above the chunks lent by give/take: before give/take run while spans are
    // lent, the evict callback is called (its owner takes them back). n_calls (if given) += the driver calls made
    virtual size_t give_spans(ggml_backend_buffer_t src, const size_t * heads, size_t n, size_t * n_calls) = 0;
    virtual void   take_spans(ggml_backend_buffer_t dst, const size_t * heads, const size_t * lens, size_t n, size_t * n_calls) = 0;
    virtual size_t n_spans() const = 0;
    virtual void   set_span_evict(void (*fn)(void * ud), void * ud) = 0;
};

// AMB_ELASTIC_COMPUTE=1 (C17, default off; needs elastic MoE cache banks, which C4 creates for AMB_ELASTIC_KV=3 or for
// AMB_ELASTIC_COMPUTE=1 alone, qwen4exp and inkling2 targets): the CUDA compute
// buffer of the qwen4exp or inkling2 target context is an elastic buffer (fully mapped, as a plain one); while only small graphs
// (<= llama_moe_cache::max_batch tokens) run, its chunks above the keep region [0, keep) + one guard chunk are lent to the
// MoE cache like parked KV chunks (extra bank slots, zero filled when they arrive); they come back (zero filled, the
// experts in them dropped) before any graph whose assignment reaches past the keep region is given data or computed
int llama_elastic_compute_env(); // 0/1, read once

// amb2: AMB_ELASTIC_COMPUTE_SMALL_MAX=N (1..32, default 32 = llama_moe_cache::max_batch): the keep region is the worst
// graph of <= N tokens, the compute buffer is lent only for batches of <= N tokens, and any graph of more than N tokens
// takes it back like a large graph (9..32-token graphs still use the MoE cache, with its own and KV-lent slots only).
// With AMB_ELASTIC_FLOOR_MIB the value is forced to 32 (the floor must be lent for every graph that uses the cache)
int64_t llama_elastic_compute_small_max();

class llama_compute_elastic {
public:
    // gran: the chunk size; slack: bytes added to a measured high-water mark before rounding to chunks
    llama_compute_elastic(llama_elastic_lender * lender, const llama_elastic_fns & api, size_t gran, size_t slack);
    ~llama_compute_elastic();

    // the keep region from the high-water mark of the worst small graph (measured at reserve time; replaces the earlier
    // value, never below the safety-net floor; takes back first if it grows into the lent region); what: for the log
    void set_keep(size_t high_water, const char * what);
    size_t keep() const { return keep_bytes; }
    size_t guard() const { return guard_chunks*gran; }

    bool lent() const { return n_lent > 0; }
    bool disabled() const { return off_reason != nullptr; }
    void disable(const char * why);

    // lend the chunks of buf (the compute buffer, an elastic buffer with one chunk) above keep + guard; synchronizes the
    // device first. Returns the chunks lent
    size_t give(ggml_backend_buffer_t buf);
    // take every lent chunk back into buf (the buffer they were lent from); `why` for the log
    void take(ggml_backend_buffer_t buf, const char * why);
    // after an allocation: if the assignment of buf reaches past the keep region while lent, take back first; a small
    // graph (n_tokens in [1, max_batch]) doing so raises the keep region (one warning). Returns true if it took back
    bool check_alloc(ggml_backend_buffer_t buf, size_t high_water, int64_t n_tokens, int64_t max_batch);

    void log(const char * event) const;

private:
    llama_elastic_lender * lender;
    const llama_elastic_fns & api;
    size_t gran;
    size_t slack;
    size_t keep_bytes = 0;     // the keep region [0, keep_bytes), a multiple of gran
    size_t keep_floor = 0;     // raised by the safety net, kept by later measurements
    size_t guard_chunks = 1;   // mapped above it, for reads past the end of a tensor

    ggml_backend_buffer_t lent_buf = nullptr; // the buffer the lent chunks belong to
    size_t lent_first = 0;     // the lowest lent chunk
    size_t n_lent     = 0;     // chunks lent
    size_t buf_chunks = 0;     // chunks of lent_buf
    std::vector<size_t> lent_heads; // C17c: the heads of the lent spans (mappings) of lent_buf
    std::vector<size_t> lent_lens;  // their chunk counts

    static void evict_cb(void * ud);
    uint64_t last_calls = 0;   // driver calls of the last transition
    uint64_t n_evict    = 0;   // take-backs asked by KV growth/shrink

    const char * off_reason = nullptr;

    // stats
    uint64_t n_give = 0;
    uint64_t n_take = 0;
    uint64_t n_net  = 0;       // take-backs forced by a small graph past the keep region
    double   ms_give = 0.0, ms_take = 0.0, last_ms = 0.0;
    uint64_t dropped_total = 0;
    uint64_t last_dropped  = 0;
    int64_t  last_slots    = 0; // slots gained by the last give
    uint64_t n_logged = 0;
    bool     warned_net = false;
};

// the mapping state of the K/V (or indexer) tensors of one cache
class llama_kv_elastic {
public:
    struct tensor_desc {
        size_t offset; // in the buffer
        size_t row;    // bytes per cell
        size_t size;   // bytes of the tensor (ggml_nbytes)
        ggml_tensor * t = nullptr; // C4: the tensor (AMB_ELASTIC_POISON fills its newly mapped cells)
    };

    struct buffer_desc {
        ggml_backend_buffer_t    buf; // elastic, owned by the cache
        std::vector<tensor_desc> tensors;
    };

    // the buffers must be elastic and fully mapped; level >= 2 parks the chunks beyond `unit` cells
    llama_kv_elastic(const char * name, int level, uint32_t kv_size, std::vector<buffer_desc> bufs);
    ~llama_kv_elastic();

    // the cells [0, n) are about to be read or written: map them (and a margin) if they are not
    void need(uint32_t n);

    // every cell (K-shift rotates the whole cache)
    void need_all();

    // level 3, at decode entry: the cells [0, n) are in use or about to be; return the chunks past the margin to the
    // lender if at least 2 units can go
    void maybe_shrink(uint32_t n);

    uint32_t mapped_cells() const { return n_mapped; }

    // C4 (AMB_ELASTIC_CHECK=1): abort unless the cells [0, n) are backed by mapped chunks in every tensor; what: for the message
    void check_backed(uint32_t n, const char * what) const;

    // level 3: the parked chunks, and lending them all (as far as the lender takes them)
    size_t parked_bytes() const;
    void lend(llama_elastic_lender * lender);
    void lender_gone();

    // one applied ubatch, for the periodic stats line (AMB_MOE_STATS)
    void tick();

private:
    struct buf_state {
        buffer_desc d;
        ggml_backend_buffer_t spare = nullptr; // owned
        size_t n_spare = 0; // chunks of the spare range
        size_t top     = 0; // the spare chunks [0, top) are mapped (a stack)
        size_t lent    = 0; // chunks of this buffer now mapped in the lender
        size_t lost    = 0; // chunks that were lent when the lender went away: released with its banks, remapped fresh if needed
    };

    std::vector<uint8_t> want_of(const buf_state & bs, uint32_t cells) const;
    void map_to(uint32_t cells, const char * why);
    void shrink_to(uint32_t cells);
    void log(const char * event) const;
    uint32_t cells_backed() const; // cells whose chunks are mapped in every tensor (>= n_mapped)
    void poison_cells(uint32_t c0, uint32_t c1) const; // C4: AMB_ELASTIC_POISON fill of the cells [c0, c1)
    mutable uint64_t n_checks = 0; // C4: AMB_ELASTIC_CHECK checks passed

    std::string name;
    int      level;
    uint32_t kv_size;
    uint32_t unit;
    uint32_t n_mapped; // the cells [0, n_mapped) are mapped in every tensor

    std::vector<buf_state> bufs;

    llama_elastic_lender * lender = nullptr;

    // stats
    uint64_t n_grow   = 0;
    uint64_t n_shrink = 0;
    double   last_ms  = 0.0;  // of the last event that moved or mapped chunks
    size_t   last_chunks = 0; // chunks moved by that event
    double   total_ms = 0.0;
    int64_t  stats_every = 0;
    int64_t  ticks = 0;
};
