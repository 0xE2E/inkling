#include "llama-moe-cache.h"

#include "llama-elastic.h"
#include "llama-ext.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
// AMB_MOE_MISS_OVERLAP (C23): the host poll of the publish flags
#include <atomic>
#include <thread>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define AMB_CPU_PAUSE() _mm_pause()
#else
#define AMB_CPU_PAUSE() ((void) 0)
#endif

namespace {

// ClassLRU: one LRU list per frequency class, over the experts of a group of layers
// slot_map[il][e] is the slot of expert e of layer il, n_slots or later if not cached, the host slot map tensor
// CRF (combined recency and frequency) = uses of an expert, decayed with a half-life of N tokens, retained across evictions
// class = floor(log2(CRF)), at most n_class - 1; each slot is in the LRU list of the class, as of its last use
// victim: an empty slot, or the least recently used slot of each list not in the current plan() with the max age/2^class
struct moe_cache_lru {
    static constexpr int   n_class  = 8; // floor(log2(1/(1 - 2^(-1/crf_half_life)))) = 7, class of CRF_max for half-life of 128
    static constexpr float crf_half_life = 128.0f; // tokens

    int32_t n_expert = 0;
    int32_t n_slots  = 0;
    int32_t empty    = 0; // slot of the experts that are not cached

    // [n_layer] data of the slot maps, the experts at slot n_slots or later are not cached
    // the experts computed on the CPU read zero slots after the cached ones on the device
    std::vector<int32_t *> slot_map;
    std::vector<int32_t>   key_of;   // [n_slots] key of the expert in the slot, -1 if empty

    // one LRU list per class, the head is the least recently used end
    std::vector<int32_t> prev;    // [n_slots] -1 at the ends
    std::vector<int32_t> next;    // [n_slots]
    int32_t head[n_class];
    int32_t tail[n_class];
    std::vector<int8_t>  list_of; // [n_slots] class list of the slot
    int32_t n_filled = 0;         // slots that hold an expert

    std::vector<float>   crf;      // [n_layer*n_expert]
    std::vector<int64_t> last_use; // [n_layer*n_expert] layer clock at the last use, -1 if not used
    std::vector<int64_t> clock;    // [n_layer] tokens seen by the layer (as added by plan() for each ubatch)

    std::vector<uint32_t> seen; // [n_expert] seen[e] == seen_gen if expert e is selected by the current plan()
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;  // the distinct selected experts
    std::vector<int32_t> misses;
    std::vector<uint64_t> token_zero; // [n_tokens] zero slots used by each token

    // AMB_MOE_PREFETCH (see impl::prefetch): empty when prefetch is off, then nothing below changes the behaviour
    // a prefetch upload batch is numbered on its device; the slot is in flight while its batch is not known to be complete,
    // i.e. pf_seq[s] > pf_done: no plan() may evict it (no write-after-write between the upload stream and the compute stream)
    std::vector<uint64_t> pf_seq;    // [n_slots] batch of the last prefetch upload into the slot, 0 if none
    std::vector<uint8_t>  pf_unused; // [n_slots] holds a prefetched expert that no plan() has selected since
    uint64_t pf_done = 0;            // the batches <= pf_done are complete on the device (see impl::prefetch_ack)
    bool     pf_need_wait = false;   // the last plan() selected an in-flight slot: the compute stream must wait for the uploads
    uint64_t pf_hit_done = 0;        // selections served by a prefetched slot whose upload was known complete
    uint64_t pf_hit_wait = 0;        // the same with the upload possibly in flight: served after a stream wait
    uint64_t pf_wasted   = 0;        // prefetched experts evicted without ever being selected

    bool in_flight(int32_t s) const {
        return !pf_seq.empty() && pf_seq[s] > pf_done;
    }

    // AMB_MOE_LOOKAHEAD (C18): empty / unused when lookahead is off (and with AMB_MOE_PREFETCH, which keeps its rules)
    // the pf_* fields above are shared with AMB_MOE_PREFETCH (the two switches exclude each other)
    std::vector<int64_t>  key_round;  // [n_layer*n_expert] round of the last plan() selection of the key, 0 if never
    int64_t               cur_round = 2; // a round = one evaluation through the cache (copy() of the first cached layer)
    std::vector<uint32_t> la_mark;    // [n_slots] la_mark[s] == la_mark_gen: chosen by the lookahead batch being planned
    uint32_t              la_mark_gen = 0;
    std::vector<int32_t>  la_unused;  // slots admitted by lookahead (entries whose pf_unused is clear are stale)
    uint64_t              pf_wait_seq = 0; // the newest batch among the in-flight slots the last plan() selected
    uint64_t              la_first_victims = 0; // real misses placed into a complete, prefetched-unused slot of their layer
    // no victim for a real miss because of in-flight slots: drain the upload stream (sets pf_done), then retry once
    void (*drain_fn)(void *, bool forced) = nullptr;
    void * drain_ud = nullptr;
    int64_t dbg_no_victim_every = 0; // AMB_MOE_LOOKAHEAD_DEBUG_NO_VICTIM (C18b)
    int64_t dbg_no_victim_n     = 0;
    uint64_t n_drain_calls      = 0; // C21: drain_fn calls by victim() of this LRU (per-group view of the "no victim" drains)

    bool any_in_flight() const {
        if (pf_seq.empty()) {
            return false;
        }
        for (int32_t s = 0; s < n_slots; ++s) {
            if (pf_seq[s] > pf_done) {
                return true;
            }
        }
        return false;
    }

    // AMB_ELASTIC_KV level 3: only the slots [0, n_active) exist (their bank memory is mapped); the slots
    // [n_active, n_slots) are parked: in no list, holding no expert, never a victim. n_active == n_slots otherwise.
    int32_t n_active = 0;
    std::vector<uint8_t> parked; // [n_slots]

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots, int32_t n_active_init = -1) {
        GGML_ASSERT(n_class == (int) std::log2(1.0f/(1.0f - std::exp2(-1.0f/crf_half_life))) + 1);
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        this->empty    = n_slots;
        this->n_active = n_active_init < 0 ? n_slots : std::min(n_active_init, n_slots);
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        parked.assign(n_slots, 0);
        for (int32_t s = 0; s < n_slots; ++s) {
            if (s < n_active) {
                prev[s] = s - 1;
                next[s] = s + 1 < n_active ? s + 1 : -1;
            } else {
                prev[s]   = -1;
                next[s]   = -1;
                parked[s] = 1;
            }
        }
        for (int c = 0; c < n_class; ++c) {
            head[c] = -1;
            tail[c] = -1;
        }
        head[0] = n_active > 0 ? 0 : -1;
        tail[0] = n_active - 1;
        list_of.assign(n_slots, 0);
        crf.assign((size_t) n_layer*n_expert, 0.0f);
        last_use.assign((size_t) n_layer*n_expert, -1);
        clock.assign(n_layer, 0);
        seen.assign(n_expert, 0);
    }

    // move slot s to the most recently used end of list c
    void move(int32_t s, int c) {
        const int from = list_of[s];
        if (from == c && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[c];
        next[s] = -1;
        if (tail[c] >= 0) {
            next[tail[c]] = s;
        } else {
            head[c] = s;
        }
        tail[c] = s;
        list_of[s] = c;
    }

    // AMB_MOE_PREFETCH: move slot s to the least recently used end of list c, the lowest priority of that list
    void move_head(int32_t s, int c) {
        const int from = list_of[s];
        if (from == c && s == head[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = -1;
        next[s] = head[c];
        if (head[c] >= 0) {
            prev[head[c]] = s;
        } else {
            tail[c] = s;
        }
        head[c] = s;
        list_of[s] = c;
    }

    // AMB_ELASTIC_KV level 3: take slot s out of its list
    void unlink(int32_t s) {
        const int from = list_of[s];
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }
        prev[s] = -1;
        next[s] = -1;
    }

    // park an active slot (its memory is about to be unmapped): the expert in it is dropped, re-fetched on its next miss
    // returns true if it held an expert
    bool park(int32_t s) {
        GGML_ASSERT(!parked[s]);
        // AMB_MOE_LOOKAHEAD: every path that parks drains the upload stream first (no upload may write unmapped memory)
        GGML_ASSERT(!in_flight(s) && "MoE cache: a slot with an upload in flight is parked");
        if (!pf_unused.empty() && pf_unused[s]) {
            pf_unused[s] = 0;
            pf_wasted++;
        }
        bool dropped = false;
        if (key_of[s] >= 0) {
            slot_map[key_layer(key_of[s])][key_expert(key_of[s])] = empty;
            key_of[s] = -1;
            n_filled--;
            dropped = true;
        }
        unlink(s);
        list_of[s] = 0;
        parked[s]  = 1;
        return dropped;
    }

    // an empty slot whose memory was just mapped: the least recently used end of list 0, so it is the next victim
    void unpark(int32_t s) {
        GGML_ASSERT(parked[s] && key_of[s] < 0);
        parked[s]  = 0;
        prev[s]    = -1;
        next[s]    = head[0];
        if (head[0] >= 0) {
            prev[head[0]] = s;
        } else {
            tail[0] = s;
        }
        head[0]    = s;
        list_of[s] = 0;
    }

    // returns the number of experts dropped
    size_t set_active(int32_t n) {
        GGML_ASSERT(n >= 0 && n <= n_slots);
        size_t dropped = 0;
        for (int32_t s = n; s < n_active; ++s) {
            dropped += park(s) ? 1 : 0;
        }
        for (int32_t s = n_active; s < n; ++s) {
            unpark(s);
        }
        n_active = n;
        return dropped;
    }

    // key = index of an expert of a layer in crf and last_use
    int32_t make_key(int32_t il, int32_t e) const { return il*n_expert + e; }
    int32_t key_layer (int32_t key) const { return key / n_expert; }
    int32_t key_expert(int32_t key) const { return key % n_expert; }

    // for a key after use(): its CRF is >= 1, so log2 is >= 0 and the cast is the floor
    int class_of(int32_t key) const {
        return std::min(n_class - 1, (int) std::log2(crf[key]));
    }

    // the CRF of a key, decayed by the time since its last use
    float decayed_crf(int32_t il, int32_t key) const {
        return crf[key]*std::exp2(-(clock[il] - last_use[key])/crf_half_life);
    }

    // an unused key gets 1 because crf[key] is 0 prior to first use
    void use(int32_t il, int32_t key) {
        crf[key]      = decayed_crf(il, key) + 1.0f;
        last_use[key] = clock[il];
    }

    // the experts selected by this plan() must not be evicted, nor the slots this plan() filled already
    bool is_selected(int32_t il, int32_t key) const {
        if (key < 0) {
            return false; // unused slot
        }
        if (key_layer(key) != il) {
            return false; // seen is per expert index, an expert of another layer can have the same index
        }
        return seen[key_expert(key)] == seen_gen;
    }

    // AMB_MOE_PREFETCH: a key selected in the last 2 tokens of its layer, i.e. by a plan() of the current or the previous
    // decode step (last_use is the layer clock at the use, the clock advances by n_tokens per plan())
    // AMB_MOE_LOOKAHEAD (key_round not empty): selected by a plan() of the current or the previous round instead
    bool recently_used(int32_t key) const {
        if (key < 0) {
            return false;
        }
        if (!key_round.empty()) {
            return key_round[key] > 0 && key_round[key] >= cur_round - 1;
        }
        return last_use[key] >= 0 && last_use[key] >= clock[key_layer(key)] - 2;
    }

    // the least recently used slot of list c that this plan() does not select, -1 if none
    // never a slot with a prefetch upload in flight
    // for a prefetch (pf): also never a recently used slot, see impl::prefetch for why this is safe, nor (lookahead) a slot
    // already chosen for the batch being planned
    int32_t candidate(int c, int32_t il, bool pf = false) const {
        for (int32_t s = head[c]; s >= 0; s = next[s]) {
            if (!is_selected(il, key_of[s]) && !in_flight(s) && !(pf && recently_used(key_of[s])) &&
                    !(pf && !la_mark.empty() && la_mark[s] == la_mark_gen)) {
                return s;
            }
        }
        return -1;
    }

    // pf (AMB_MOE_PREFETCH): the victim for a prefetched expert, -1 if none; il is then the layer whose plan() ran last
    // in this copy(), so its selected slots are excluded
    // AMB_MOE_LOOKAHEAD: for a real miss (not pf) with no victim because slots are in flight, drain_fn finishes the
    // uploads and the search is repeated once (the "no MoE cache victim" abort of C6 on multi-token batches)
    int32_t victim(int32_t il, bool pf = false) {
        // C18b debug: report "no victim" on every n-th search while uploads are in flight, to exercise the path below
        bool forced = false;
        if (!pf && dbg_no_victim_every > 0 && drain_fn != nullptr && any_in_flight() && ++dbg_no_victim_n % dbg_no_victim_every == 0) {
            forced = true;
        }
        int32_t res = forced ? -1 : victim_search(il, pf);
        if (res < 0 && !pf && drain_fn != nullptr && any_in_flight()) {
            n_drain_calls++;
            drain_fn(drain_ud, forced);
            res = victim_search(il, pf);
        }
        if (res < 0 && !pf) {
            GGML_ABORT("no MoE cache victim");
        }
        return res;
    }

    int32_t victim_search(int32_t il, bool pf) const {
        int32_t res = -1;
        double res_score = -1.0;
        for (int c = 0; c < n_class; ++c) {
            const int32_t s = candidate(c, il, pf);
            if (s < 0) {
                continue;
            }
            const int32_t key = key_of[s];
            if (key < 0) {
                return s; // unused slot
            }
            const int64_t age   = clock[key_layer(key)] - last_use[key];
            const double  score = std::ldexp((double) age, -c); // age/2^class
            if (score > res_score) {
                res       = s;
                res_score = score;
            }
        }
        return res;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids (n_tokens tokens) in layer il
    // n_cpu(n) of the n misses are left to the CPU and read a zero slot on the device, they are not cached
    // a zero slot is used at most once per token, a miss that does not get one is cached instead
    // returns false if the ids select more distinct experts than there are slots
    // AMB_MOE_LOOP_DEDUP: il is the representative (physical) layer; tick is false for the later visits of a looped block,
    // so the clock of the physical layer advances once per ubatch (true, the old behaviour, otherwise)
    template <typename F>
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, int64_t n_tokens, int32_t n_used, int32_t n_zero, F && n_cpu,
            std::vector<fill> & fills, std::vector<fill> & cpu, size_t & n_hit, bool tick = true) {
        fills.clear();
        cpu.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_active) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // update the CRF of all selected experts and promote hits before choosing victims for misses
        misses.clear();
        pf_need_wait = false;
        pf_wait_seq  = 0;
        for (int32_t e : uniq) {
            const int32_t key = make_key(il, e);
            use(il, key);
            if (!key_round.empty()) {
                key_round[key] = cur_round; // AMB_MOE_LOOKAHEAD: round-based recency of the lookahead victims
            }
            if (slots[e] < n_slots) {
                move(slots[e], class_of(key));
                n_hit++;
                if (!pf_seq.empty()) {
                    // AMB_MOE_PREFETCH: a hit on a slot whose upload may still run, the caller orders the compute stream after it
                    const int32_t s = slots[e];
                    const bool flight = in_flight(s);
                    pf_need_wait = pf_need_wait || flight;
                    if (flight) {
                        pf_wait_seq = std::max(pf_wait_seq, pf_seq[s]);
                    }
                    if (pf_unused[s]) {
                        pf_unused[s] = 0;
                        (flight ? pf_hit_wait : pf_hit_done)++;
                    }
                }
            } else {
                misses.push_back(e);
            }
        }

        // the misses with the lowest CRF are left to the CPU, a miss that keeps being used gains CRF and enters the cache
        // all misses are used at the same clock, so their CRF compare without decay
        const size_t n_c = std::min(n_cpu(misses.size()), misses.size());
        if (n_c > 0) {
            std::sort(misses.begin(), misses.end(), [&](int32_t a, int32_t b) {
                const float crf_a = crf[make_key(il, a)];
                const float crf_b = crf[make_key(il, b)];
                return crf_a != crf_b ? crf_a < crf_b : a < b;
            });
            GGML_ASSERT((size_t) n_tokens*n_used == n_ids);
            token_zero.assign(n_tokens, 0);
            size_t n_keep = n_c;
            for (size_t i = 0; i < n_keep; ) {
                const int32_t e = misses[i];
                uint64_t used = 0;
                for (int64_t it = 0; it < n_tokens; ++it) {
                    for (int32_t k = 0; k < n_used; ++k) {
                        if (ids[it*n_used + k] == e) {
                            used |= token_zero[it];
                        }
                    }
                }
                int32_t z = 0;
                while (z < 64 && (used >> z & 1)) {
                    z++;
                }
                if (z >= n_zero) {
                    std::swap(misses[i], misses[--n_keep]);
                    continue;
                }
                for (int64_t it = 0; it < n_tokens; ++it) {
                    for (int32_t k = 0; k < n_used; ++k) {
                        if (ids[it*n_used + k] == e) {
                            token_zero[it] |= 1ull << z;
                        }
                    }
                }
                slots[e] = n_slots + z;
                cpu.push_back({ e, n_slots + z });
                i++;
            }
            misses.erase(misses.begin(), misses.begin() + n_keep);
        }

        // fill the misses in expert order: victim() takes unused slots, and slots filled by one plan(), in list order,
        // so consecutive experts often get consecutive slots and one upload
        std::sort(misses.begin(), misses.end());
        for (int32_t e : misses) {
            // AMB_MOE_LOOKAHEAD: a complete prefetched-unused slot of this layer is the first victim (a wrong prediction);
            // it is not selected (a selected one would be a hit and has lost its flag above) and not in flight
            int32_t s = -1;
            for (size_t i = 0; i < la_unused.size() && s < 0; ) {
                const int32_t u = la_unused[i];
                if (!pf_unused[u] || key_of[u] < 0) {
                    la_unused[i] = la_unused.back(); // stale entry
                    la_unused.pop_back();
                    continue;
                }
                if (key_layer(key_of[u]) == il && !in_flight(u) && !is_selected(il, key_of[u])) {
                    s = u;
                    la_unused[i] = la_unused.back();
                    la_unused.pop_back();
                    la_first_victims++;
                    break;
                }
                i++;
            }
            if (s < 0) {
                s = victim(il);
            }
            if (key_of[s] >= 0) {
                slot_map[key_layer(key_of[s])][key_expert(key_of[s])] = empty;
            } else {
                n_filled++;
            }
            if (!pf_unused.empty() && pf_unused[s]) {
                pf_unused[s] = 0; // AMB_MOE_PREFETCH: a wrong prediction
                pf_wasted++;
            }
            const int32_t key = make_key(il, e);
            key_of[s] = key;
            slots[e]  = s;
            move(s, class_of(key));
            fills.push_back({ e, s });
        }
        if (tick) {
            clock[il] += n_tokens;
        }
        return true;
    }

    // AMB_MOE_PREFETCH: give slot s, a victim(il_real, true), to the not cached expert e of layer il, at the lowest priority
    // the expert is not used: its CRF and last use stay, so a wrong prediction is the first victim once its upload completes
    void pf_admit(int32_t il, int32_t e, int32_t s, uint64_t seq) {
        if (key_of[s] >= 0) {
            slot_map[key_layer(key_of[s])][key_expert(key_of[s])] = empty;
            if (pf_unused[s]) {
                pf_wasted++;
            }
        } else {
            n_filled++;
        }
        key_of[s] = make_key(il, e);
        slot_map[il][e] = s;
        move_head(s, 0);
        pf_unused[s] = 1;
        pf_seq[s]    = seq;
        if (!key_round.empty()) {
            la_unused.push_back(s); // AMB_MOE_LOOKAHEAD: candidates for the first-victim rule of plan()
        }
    }

    // C18b (review m2): undo an admission whose upload was never issued (exception unwind): the slot becomes empty
    void la_revert(int32_t s) {
        if (key_of[s] >= 0) {
            slot_map[key_layer(key_of[s])][key_expert(key_of[s])] = empty;
            key_of[s] = -1;
            n_filled--;
        }
        pf_unused[s] = 0;
        pf_seq[s]    = 0;
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

// AMB_MOE_LOOP_DEDUP=1 (C2 loop dedup, default off): looped models (hparams.loop_n_pass > 1, e.g. inkling2) visit the
// physical blocks [loop_core_start, loop_core_end) several times; llama_model_loop_views gives each later visit its own
// tensor views of the SAME host expert data. With the switch on, the cache treats every visit of a physical block as that
// block: all per-layer state that describes expert identity is indexed by the representative layer rep(il) =
// loop_visit(loop_layer(il)) (the first visit, an effective layer index, so the arrays keep their [n_layer] size):
//   - LRU keys make_key(rep, e), crf/last_use, is_selected and the victim protection of plan(rep)
//   - the host slot map: lru.slot_map[il] of every visit points at the array of rep (one physical mapping, so every visit
//     sees the same expert->slot map after every plan(); copy() uploads lru.slot_map[il], i.e. that shared array)
//   - the clock of the physical layer: advanced by n_tokens only by the plan() of the first visit (il == rep), so ages
//     stay in tokens and compare with the other layers of the group
//   - host_bytes (budget split, max slots, the init log, the trace header): each physical tensor counted once
// Unchanged per visit: the slot map tensors (each visit has its own host tensor and device copy, the scheduler and the
// graph see il), the CPU maps and the CPU-share carry (they describe what the graph of il computes), stats, the trace
// records (effective il), the bindings (each view tensor has its own binding; copy_experts reads the shared array).
// Safety (C = compute stream): with the modes accepted below, copy(il) runs after the scheduler synchronized C and
// enqueues the fills and the slot map upload of il on C; every kernel that reads the slots of an earlier visit of the
// same block was enqueued on C before copy(il) (graph order), so a fill of a later visit that evicts a slot an earlier
// visit read is ordered after that read, exactly as an eviction of another layer's slot today. Within one plan() the
// selected experts of the physical block are protected (is_selected on rep). The modes with a second stream and per-layer
// prediction state (PREFETCH, LOOKAHEAD, MISS_OVERLAP) keep invariants keyed on the effective layer (recently_used by the
// layer clock, la_unused "own layer" rule, la_pred_round, slot_map2 staging, publish slots): they are refused, the
// switch then stays off (one log line).
// returns true if the switch is on and accepted; why: the refusal reason, nullptr if not refused
static bool llama_moe_cache_loop_dedup_env(const char ** why) {
    *why = nullptr;
    const char * v = getenv("AMB_MOE_LOOP_DEDUP");
    if (v == nullptr || strcmp(v, "1") != 0) {
        return false;
    }
    if (const char * s = getenv("AMB_MOE_PREFETCH"); s != nullptr && (atoi(s) == 1 || atoi(s) == 4)) {
        *why = "AMB_MOE_PREFETCH is on (second-stream prefetch keyed on the effective layer)";
    } else if (const char * s = getenv("AMB_MOE_LOOKAHEAD"); s != nullptr && *s != '\0' && strcmp(s, "0") != 0) {
        *why = "AMB_MOE_LOOKAHEAD is on (lookahead victims and first-victim rule keyed on the effective layer)";
    } else if (const char * s = getenv("AMB_MOE_MISS_OVERLAP"); s != nullptr && (strcmp(s, "1") == 0 || strcmp(s, "2") == 0)) {
        *why = "AMB_MOE_MISS_OVERLAP is on (miss uploads on the upload stream, per-layer publish/gate state)";
    }
    return *why == nullptr;
}

// the representative layer of each effective layer (identity if dedup is off or the model is not looped)
static std::vector<int32_t> llama_moe_cache_loop_rep(const llama_model & model, bool dedup) {
    std::vector<int32_t> res(model.layers.size());
    for (size_t il = 0; il < res.size(); ++il) {
        res[il] = (int32_t) il;
        if (dedup && model.hparams.loop_n_pass > 1) {
            const int32_t rep = (int32_t) model.hparams.loop_visit(model.hparams.loop_layer((uint32_t) il));
            if (rep >= 0 && rep <= (int32_t) il) {
                res[il] = rep;
            }
        }
    }
    return res;
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;

        // AMB_MOE_PREFETCH: a second backend instance of the same device, only for its own stream: the prefetch uploads
        // batches are numbered from 1, batch b records pf_ev[b % 2] on the upload stream after its uploads
        ggml_backend_t upload = nullptr;
        ggml_backend_event_t pf_ev[2] = { nullptr, nullptr };
        uint64_t pf_last   = 0; // the last batch started
        uint64_t pf_waited = 0; // the compute stream has been made to wait for the batches <= pf_waited
        uint64_t pf_done   = 0; // the batches <= pf_done are complete: pf_waited at the last synchronize of the compute stream

        // AMB_MOE_LOOKAHEAD (C18): upload is shared (the two switches exclude each other); lookahead batches are numbered
        // from 1, batch b records la_ev[b % la_n_ev] on the upload stream after its copies; la_ev_c is recorded on the
        // compute stream after the blocking copies and the slot map upload of the copy() that issues a batch, and the
        // upload stream waits on it before the batch
        // C25: the ring has la_n_ev events (4 as before); AMB_MOE_LA_CHUNK_MIB raises it to la_n_ev_max (more batches per
        // layer, so that an I4 wait finds the event of its own chunk, not a later one), before any batch exists
        static constexpr int la_n_ev_max = 16;
        int la_n_ev = 4;
        ggml_backend_event_t la_ev[la_n_ev_max] = {};
        ggml_backend_event_t la_ev_c = nullptr;
        uint64_t la_last = 0; // the last batch issued
        uint64_t la_done = 0; // the batches <= la_done are known complete (event query or drain)
        // C18c AMB_MOE_LOOKAHEAD_DEBUG_DEFER: the batches <= la_flushed have their copies and event record enqueued on U
        // (always == la_last without DEFER); event queries only look at flushed batches
        uint64_t la_flushed = 0;

        // AMB_MOE_MISS_OVERLAP (C23): recorded on U after the miss uploads of a copy(), C waits on it before the layer;
        // mo_dirty: miss uploads were enqueued on U since U was last synchronized (lookahead_drain synchronizes it)
        ggml_backend_event_t mo_ev = nullptr;
        bool mo_dirty = false;
        // C25 AMB_MOE_MISS_STREAM=1: a third backend instance (its own stream M) for the miss uploads; mo_ev is then
        // recorded on M and mo_dirty means "miss uploads on M since M was last synchronized"; nullptr: misses on U
        ggml_backend_t miss = nullptr;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        int32_t n_phys = 0;               // layers with rep(il) == il (AMB_MOE_LOOP_DEDUP), layers.size() with the switch off
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;            // each physical expert tensor counted once (AMB_MOE_LOOP_DEDUP)
        int32_t n_slots = 0;
        moe_cache_lru lru;
        bool pf_ok = false; // AMB_MOE_PREFETCH: the layers of the group are prefetched
        bool la_ok = false; // AMB_MOE_LOOKAHEAD: the layers of the group get lookahead (configuration only)
        // AMB_MOE_LOOKAHEAD_SMALL (C21): la_ok only by the relaxed slot rule; counters of its lookahead batches
        bool     la_small = false;
        uint64_t la_small_issued = 0, la_small_no_victim = 0, la_small_capped = 0;

        // AMB_ELASTIC_KV level 3 (elastic banks): n_slots is the capacity (base + the most it can borrow), n_base the
        // slots of its own memory (what the budget gives, as without lending), lru.n_active the slots mapped now
        // per bank: its chunks [off/gran, off/gran + m) are mapped, m_base of them are the cache's own, at most m_cap
        struct ebank {
            size_t off    = 0; // in the device buffer, a multiple of the granularity
            size_t nb2    = 0; // bytes per slot
            size_t m      = 0;
            size_t m_base = 0;
            size_t m_cap  = 0;
            // C17c: spans at the top of [m_base, m): { first chunk (absolute in the device buffer), chunks }, a stack
            std::vector<std::pair<size_t, size_t>> sp;
        };
        int32_t n_base = 0;
        // C4b (AMB_ELASTIC_FLOOR_MIB): the slots the graphs may count on (get_slot_map): n_base plus the slots this
        // group is guaranteed from the compute buffer lent before every small graph; n_base without the floor
        int32_t n_graph = 0;
        std::vector<ebank> eb;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        ggml_tensor * slot_map_cpu = nullptr; // looked up in place of slot_map by the graphs with a CPU part, if cpu_share > 0, its data is not used
        // AMB_MOE_MISS_OVERLAP=2 (C24): I32 [2, n_expert], looked up in place of slot_map by split-phase graphs (row 0 the
        // hit-phase slot, row 1 the miss-phase slot, -1 where the phase does not compute the expert); its data is not used,
        // copy() uploads it from pinned staging (mo2_stage)
        ggml_tensor * slot_map2 = nullptr;
        // AMB_MOE_HIT_FIRST (D2): I32 [1, n_expert] each, looked up in place of slot_map by hit-first graphs. slot_map_h:
        // the slots before the plan of the ubatch (-1: not cached), read by the hit phase; slot_map_r: the slots the plan
        // filled (-1: everything else), read by the miss phase. Their data is not used, copy() uploads them from staging
        ggml_tensor * slot_map_h = nullptr;
        ggml_tensor * slot_map_r = nullptr;
        ggml_tensor * cpu_map  = nullptr;   // I32 [1, n_expert] in host memory, if cpu_share > 0
        std::vector<int32_t> cpu_experts;   // experts set in cpu_map
        uint32_t cpu_acc = 0;               // fraction of a miss carried over to the next plan, in 1/65536
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t cpu    = 0; // misses computed on the CPU
        size_t bytes  = 0;
        size_t routes = 0; // selected (token, expert) pairs, hits + misses counts the distinct experts
        size_t calls  = 0; // slot map copies
    };

    // max_batch: llama_moe_cache::max_batch (also the AMB_PF_OVERLAP large-batch threshold in llama-context.cpp)

    int32_t n_expert_used;

    // AMB_MOE_BATCH_COPY=1: the miss fills of a layer in one cudaMemcpyBatchAsync
    struct bc_item {
        ggml_tensor *   bank;
        const void *    src;
        size_t          offset;
        size_t          size;
    };
    typedef bool (*batch_fn_t)(ggml_backend_t, void **, const void **, const size_t *, size_t);
    bool        batch_copy = false;
    batch_fn_t  batch_fn   = nullptr;
    bool        batch_resolved = false;
    std::vector<bc_item>      bc_items;
    std::vector<void *>       bc_dsts;
    std::vector<const void *> bc_srcs;
    std::vector<size_t>       bc_sizes;
    size_t bc_calls = 0, bc_copies = 0, bc_bytes_total = 0, bc_fallbacks = 0;

    // AMB_IDS_ASYNC=1 (C14, default off): the selected expert ids of the next cached layer are read back into pinned host
    // memory by an async D2H copy enqueued right after the split that precedes the lookup split was launched (scheduler
    // split callback, ids_post_split), with an event after it; copy() then waits on that event (already complete after the
    // scheduler's own synchronize before the copy callback) instead of a second readback + synchronize. The slot map is
    // uploaded from a per-layer pinned staging copy. Any unmet condition falls back to the old path (counted).
    struct ids_async_state {
        bool on = false;
        ggml_backend_buffer_ptr pinned;          // [max_pend][cap] ids, then one staging slot map per cached layer
        int32_t * ids_base = nullptr;
        size_t    cap      = 0;                  // int32 per pending entry (n_expert_used * max_batch)
        std::vector<int32_t *> stage;            // [n_layer] staging slot map, nullptr if the layer is not cached
        ggml_backend_event_t ev = nullptr;       // recorded after the readbacks of one ids_post_split
        static constexpr int max_pend = 4;
        struct pending {
            const ggml_tensor * sel = nullptr;
            int32_t   il  = -1;
            int32_t * dst = nullptr;
            size_t    n   = 0;
            // AMB_MOE_LOOKAHEAD: the predicted ids of layer il + 1 read back by the same ids_post_split (same event)
            int32_t * la_dst  = nullptr;
            size_t    la_n    = 0;
            int64_t   la_ntok = 0;
        };
        pending pend[max_pend];
        int n_pend = 0;
        // counters
        uint64_t issued = 0, hits = 0, fb_no_pending = 0, fb_size = 0, fb_prefetch = 0, unused = 0, stage_uploads = 0;
        bool warned = false;
        // AMB_IDS_ONE_READBACK=1 (C21, default off): the real ids of layer il and the predicted ids of layer il + 1 are
        // read back by one batched D2H call (cudaMemcpyBatchAsync, 2 copies) instead of two cudaMemcpyAsync
        bool       one_rb = false;
        batch_fn_t one_fn = nullptr;
        uint64_t   one_merged = 0, one_fb = 0; // pairs read by one call; pairs read by two calls after a failed batch call
    };
    ids_async_state ia;

    // AMB_STEP_EVENTS (C14): marks around the miss uploads of copy(), set by llama_context (measurement only)
    void (*copy_mark_fn)(void * ud, ggml_backend_t backend, int which) = nullptr; // which: 0 = before the uploads, 1 = after
    void * copy_mark_ud = nullptr;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, std::pair<int32_t, bool>> layer_of; // slot map -> layer, and if the graph that looks it up has a CPU part

    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;
    std::vector<moe_cache_lru::fill> cpu;

    // AMB_MOE_LOOP_DEDUP (see llama_moe_cache_loop_dedup_env): rep[il] the representative layer of il, il itself with the
    // switch off (then every use below is the old code path); AMB_MOE_LOOP_DEDUP_CHECK=1: verify the physical mapping
    // after every plan()
    bool loop_dedup = false;
    bool loop_dedup_check = false;
    std::vector<int32_t> rep;
    uint64_t loop_dedup_checks = 0;

    float    cpu_share = 0.0f;
    uint32_t cpu_share_q = 0; // cpu_share in 1/65536, integer so that a simulator can reproduce the rounding
    int32_t  n_zero    = 0; // zero slots per group

    // LLAMA_MOE_CACHE_TRACE: the selected experts of each slot lookup, for a simulator of the cache
    FILE * trace = nullptr;

    // AMB_MOE_STATS=N (N > 0): log the statistics every N rounds, a round is one slot map copy of the first cached layer,
    // i.e. one ubatch of up to max_batch tokens through the cache (one decoded token per round in plain decoding)
    int64_t stats_every  = 0;
    int64_t stats_rounds = 0;
    int32_t stats_layer  = -1; // the first cached layer
    stats   stats_prev_small; // at the previous periodic log
    stats   stats_prev_large;

    // AMB_MOE_TIMING=1|2 (needs AMB_MOE_STATS): host wall time of the phases of copy(), in ns, and of the graph evaluations
    // mode 2 additionally synchronizes the device right after the miss uploads are enqueued: it perturbs the execution,
    // its total time is not a performance number
    struct timing {
        int64_t wait = 0;         // ids readback + synchronize
        int64_t plan = 0;         // hit/miss/eviction planning and CPU map update
        int64_t enqueue = 0;      // host time of the async miss uploads and of the slot map upload
        int64_t upload = 0;       // mode 2: synchronize after the miss uploads were enqueued
        int64_t graph = 0;        // graph evaluations that used the cache, with the GPU sync added in timing mode
        int64_t graphs = 0;
        int64_t copies = 0;       // copy() calls
        int64_t miss_experts = 0; // experts uploaded
        int64_t miss_bytes   = 0;
    };
    using clk = std::chrono::steady_clock;
    static int64_t ns_between(clk::time_point a, clk::time_point b) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
    }
    int    timing_mode = 0;
    timing tm;
    timing tm_prev;
    bool   graph_used = false;

    // AMB_MOE_PREDICT=1 (measurement only): single-token ubatches get early router probes in the graph (qwen4exp), the router
    // of layer il applied to earlier hidden states; their ids are read after the graph evaluation (llama_context::process_ubatch)
    // and compared with the experts this copy() read back for the same evaluation, with the cache state before the plan
    // predictors: see predict_name; set sizes: n_expert_used (10), 16, 24
    static constexpr int n_pred = predict_n_pred;
    static constexpr int n_k    = 3;
    struct pred_rec {
        uint64_t gen = 0;             // pred_gen of the evaluation whose copy() recorded it
        std::vector<int32_t> real;    // the experts selected for the token
        std::vector<uint8_t> cached;  // [n_expert] the expert was cached before the plan of that copy()
    };
    struct pred_acc {
        uint64_t evals = 0;           // (step, layer) pairs compared
        uint64_t real  = 0;           // selected experts
        uint64_t miss  = 0;           // selected experts that were not cached before the plan
        uint64_t hit[n_k]      = {};  // selected experts in the predicted set
        uint64_t miss_hit[n_k] = {};  // selected and not cached experts in the predicted set
        uint64_t waste[n_k]    = {};  // predicted experts neither selected nor cached
    };
    bool     predict  = false;
    uint64_t pred_gen = 0;
    int32_t  pred_k[n_k] = {};
    std::vector<pred_rec> pred_recs;  // [n_layer]
    std::vector<pred_acc> pred_cum;   // [n_pred*n_layer]
    std::vector<pred_acc> pred_prev;  // at the previous periodic log
    uint64_t pred_bad = 0;            // predicted id lists with an id out of range, not counted

    // AMB_MOE_PREFETCH=1|4 (prototype, default off): predictive expert prefetch for single-token ubatches
    // the graph computes the predicted top-k experts of layer L (predictor P1 or P4, see qwen4exp) before the slot lookup of
    // layer L-1, copy() of L-1 reads them back with the real ids (one synchronize), plans L-1 as before, then admits the
    // predicted experts of L that are not cached into victim slots and uploads them on the stream of a second backend instance
    // (device::upload), so the transfer overlaps the FFN of L-1 and the attention of L on the compute stream
    // invariants, with C the compute stream and U the upload stream of a device:
    //  - copy() runs right after a synchronize of C: no kernel enqueued earlier can still read a slot; until the next copy()
    //    C reads only the slots selected by the plan of this copy() (the MUL_MAT_ID of layer L-1); a prefetch victim is never
    //    such a slot, nor any slot selected in the last 2 tokens of its layer, nor a slot whose prefetch upload is in flight,
    //    so U never writes a slot that C may read or write
    //  - a slot written by a prefetch is in flight until its batch is known complete (lru.in_flight); plan() never evicts it,
    //    and a plan() that selects it makes C wait for the event of the last batch (U is in order, so that covers every batch
    //    before it) before the kernels that read it are enqueued; copy_experts does the same before reading the banks
    //  - a batch is known complete once C was made to wait for it and C was synchronized since (pf_done); when no plan needs a
    //    batch, C is made to wait for the batch before the last one, which bounds the in-flight slots to about 3 batches
    //  - the slot maps and LRU lists change at admission time on the host, as for a real fill; the slot map of L is uploaded by
    //    copy() of L as before, after its synchronize
    int      pf_mode      = 0;  // 0 = off, 1 = P1, 4 = P4
    int32_t  pf_k         = 10; // predicted set size
    int32_t  pf_min_layer = 2;  // first prefetched layer
    int32_t  pf_first_layer = -1; // the first cached layer, its single-token copies count the steps
    std::vector<ggml_tensor *> pf_ids_t; // [n_layer] the predicted ids of the current graph, see prefetch_set_ids
    std::vector<int32_t> pf_pred;        // the predicted ids read back
    std::vector<moe_cache_lru::fill> pf_fills;
    struct pf_stats {
        uint64_t batches    = 0; // copy() calls that read predicted ids
        uint64_t considered = 0; // predicted experts
        uint64_t cached     = 0; // predicted experts that were cached already (or in flight)
        uint64_t no_victim  = 0; // predicted experts skipped for want of an eligible victim
        uint64_t started    = 0; // prefetch uploads started (experts)
        uint64_t bytes      = 0;
        uint64_t uploads    = 0; // batches with at least one upload
        uint64_t waits_hit  = 0; // C waited for U because a selection hit an in-flight slot (or copy_experts read the banks)
        uint64_t waits_ack  = 0; // C waited for an older batch nobody selected, to release its slots
        uint64_t bad        = 0; // predicted id lists out of range or unusable, skipped
        uint64_t steps      = 0; // single-token copies of the first cached layer
        uint64_t blocking   = 0; // experts uploaded on C by single-token plans (the misses the step still waits for)
        uint64_t hit_done   = 0; // from the LRUs
        uint64_t hit_wait   = 0;
        uint64_t wasted     = 0;
        int64_t  ns         = 0; // AMB_MOE_TIMING: host time of the prefetch planning and enqueue
    };
    pf_stats pf;
    pf_stats pf_prev; // at the previous periodic log
    int64_t  tm_prev_pf_graphs = 0;

    // AMB_MOE_LOOKAHEAD=1|4 (C18, default off): cross-layer predicted lookahead prefetch on a second stream
    // The graph (qwen4exp) computes, for ubatches of up to la_max_tok (<= 8) tokens and the layers il + 1 >= la_min_layer,
    // the top-la_k experts of layer il + 1 per token (P1 or P4) before the router of layer il; AMB_IDS_ASYNC reads them
    // back with the real ids of layer il (ids_post_split, same event). copy() of layer il, after its blocking miss copies and
    // slot map upload are enqueued on the compute stream C, records la_ev_c on C, plans the predicted experts of layer il + 1
    // that are neither cached nor in flight (per-layer byte budget), admits them into victim slots (flagged prefetched-unused,
    // in flight), makes the upload stream U wait on la_ev_c and issues ONE batched copy on U, then records la_ev[seq % 4].
    // copy() of layer il + 1 queries the events (no host wait); its plan treats an in-flight slot it selects as a hit and C
    // waits on the batch's event (GPU-side) after the blocking copies are enqueued.
    // Stream invariants (one device, the scheduler synchronizes C fully before every copy callback):
    //  I1 at copy(L) C is idle; until copy(L + 1), C reads only the slots selected by plan(L) (MUL_MAT_ID of layer L, MMVQ,
    //     <= 8 tokens) and writes only the blocking fills of plan(L)
    //  I2 a lookahead victim is never selected by plan(L), never in flight, never holds a key selected in the current or the
    //     previous round, never chosen twice in one batch: U never writes a slot C may read or write before copy(L + 1)
    //  I3 plan() never evicts an in-flight slot (no write-after-write between U and C); if that leaves no victim, U is drained
    //  I4 a plan() that selects an in-flight slot makes C wait (event) on the newest batch among those slots before the
    //     kernels that read them; U is in order, so earlier batches are covered too
    //  I5 a slot is in flight until an event query (or a drain) shows its batch complete; host slot maps change at admission
    //     (as for a real fill); the device slot map of a layer is uploaded by its own copy() from pinned staging
    //  I6 U is drained (synchronized, in-flight and prefetched-unused state cleared) before: any ubatch of more than 8 tokens
    //     (MMQ may over-read slot heads; lookahead is off there), large-batch copy_experts, ring borrow, every KV/span
    //     give/take/park, teardown, and when no victim exists for a real miss
    // AMB_MOE_LOOKAHEAD_DRY=1: nodes, readback and planning run, nothing is admitted or uploaded (no cache state changes);
    // counts what would have been useful. AMB_MOE_LOOKAHEAD_DEBUG=1: victims filled with 0xFF and U delayed ~200 us first
    int      la_req       = 0;     // requested predictor (1 or 4), from the environment
    bool     la_build     = false; // the graphs get predictor nodes (decided from the configuration, also in the --fit pass)
    bool     la_on        = false; // runtime state present (upload backend, events, async ids)
    bool     la_dry       = false;
    bool     la_debug     = false;
    int32_t  la_k         = 10;
    int32_t  la_min_layer = 2;
    int32_t  la_max_tok   = 4;
    size_t   la_budget_1  = (size_t) 9 << 20;  // bytes per layer-eval, single-token ubatches
    size_t   la_budget_n  = (size_t) 14 << 20; // multi-token ubatches
    int32_t  la_first_layer = -1;
    // C18b switches (defaults reproduce c18 except la_min_tok, as asked, and the safety fixes m1/m2/m5/m6)
    int32_t  la_min_tok     = 2;     // AMB_MOE_LOOKAHEAD_MIN_TOK: no predictor / lookahead below this many tokens (c18: 1)
    bool     la_rank        = false; // AMB_MOE_LOOKAHEAD_RANK=1: ordered predictor ids, candidates by (tokens, best rank)
    bool     la_post        = false; // AMB_MOE_LOOKAHEAD_ISSUE=post: plan + issue from the post-launch split hook
    bool     la_link_wait   = true;  // AMB_MOE_LOOKAHEAD_LINK_WAIT=0: no compute-event record + upload-stream wait
    bool     la_protect     = true;  // AMB_MOE_LOOKAHEAD_PROTECT=0 (c18): predicted-and-cached slots may be victims
    bool     la_p4_fast     = false; // AMB_MOE_LOOKAHEAD_P4_FAST=1: cheaper P4 predictor mixer
    bool     la_skip_wait   = false; // AMB_MOE_LOOKAHEAD_DEBUG_SKIP_WAIT=1 (with DEBUG): negative control, omit I4 waits
    uint64_t la_spin_ns     = 200000; // AMB_MOE_LOOKAHEAD_DEBUG_SPIN_US (with DEBUG)
    int64_t  la_dbg_no_victim = 0;   // AMB_MOE_LOOKAHEAD_DEBUG_NO_VICTIM=n: every n-th real-miss victim search with slots
                                     // in flight reports "no victim" first (exercises the drain-and-retry path)
    int32_t  la_drain_above = 8;     // drain before ubatches of more tokens: min(8, MMVQ limit of every cached expert type)
    // AMB_MOE_LOOKAHEAD_ISSUE=post: the issue left by copy() for the split it precedes
    struct la_post_t {
        bool active = false;
        const ggml_cgraph * graph = nullptr; // the split graph of copy(), the post-launch hook of that split issues
        ggml_backend_t backend = nullptr;
        int32_t il_real = -1;
        int32_t il = -1;
        int64_t n_tok = 0;
        bool    recorded = false; // la_ev_c was recorded in copy()
        bool    mo = false;       // AMB_MOE_MISS_OVERLAP: take the predicted ids from their publish at the issue
    };
    la_post_t la_pend;
    size_t   la_cap       = 0;     // int32 per pending readback of predicted ids (la_k * la_max_tok)
    int64_t  la_round     = 2;     // see moe_cache_lru::cur_round
    int64_t  la_log_every = 1000;  // rounds between periodic lines without AMB_MOE_STATS
    int32_t la_set_cnt = 0, la_set_dup = 0, la_set_logged = -1; // C19: predictor tensors registered in the current graph, repeats of a layer, last logged count
    std::vector<ggml_tensor *> la_ids_t; // [n_layer] the predictor outputs of the current graph (prefetch_set_ids)
    std::vector<int32_t> la_pred;  // predicted ids of layer il + 1 taken with the ids of layer il
    int64_t  la_pred_ntok = 0;
    bool     la_have      = false;
    std::vector<int64_t> la_pred_round;          // [n_layer] round in which a lookahead was planned for the layer
    std::vector<std::vector<int32_t>> la_dry_set; // [n_layer] DRY: the experts that would have been uploaded
    std::vector<uint16_t> la_count;              // [n_expert] scratch
    std::vector<int32_t>  la_cand;
    std::vector<moe_cache_lru::fill> la_fills;
    std::vector<void *>       la_dsts;
    std::vector<const void *> la_srcs;
    std::vector<size_t>       la_sizes;
    std::vector<ggml_tensor *> la_banks;
    std::vector<uint8_t> la_best; // [n_expert] scratch: best rank of a predicted expert over the tokens (RANK)
    typedef int (*event_query_fn_t)(ggml_backend_event_t);
    typedef bool (*spin_fn_t)(ggml_backend_t, uint64_t);
    typedef bool (*la_memset_fn_t)(ggml_backend_t, void *, size_t, int, size_t, size_t);
    event_query_fn_t la_query  = nullptr;
    spin_fn_t        la_spin   = nullptr;
    la_memset_fn_t   la_memset = nullptr;
    // AMB_STEP_EVENTS set: timing events (1 of every N rounds) around the batches on U and the waits on C
    typedef void * (*tev_new_t)(ggml_backend_t);
    typedef void   (*tev_free_t)(void *);
    typedef bool   (*tev_record_t)(void *, ggml_backend_t);
    typedef bool   (*tev_elapsed_t)(void *, void *, float *);
    tev_new_t      la_tev_new = nullptr;
    tev_free_t     la_tev_free = nullptr;
    tev_record_t   la_tev_record = nullptr;
    tev_elapsed_t  la_tev_elapsed = nullptr;
    int64_t        la_tev_every = 0;
    bool           la_tev_active = false; // this round is timed
    int            la_tev_cls    = 0;
    std::vector<void *> la_tev_pool;
    size_t              la_tev_used = 0;
    std::vector<std::pair<int, int>> la_tev_up;   // event index pairs: begin/end of a batch on U
    std::vector<std::pair<int, int>> la_tev_wait; // before/after a wait of C on U
    // C18c AMB_MOE_LOOKAHEAD_ORDER_PROBE=1 (with AMB_STEP_EVENTS, default off): in timed rounds, for every copy() that
    // selects an in-flight slot (the I4 branch, waited or skipped), compare the end of the batch it needs on U with two
    // events on C: c0 before the blocking miss copies (C is idle there), c1 after the slot map upload (the last H2D copy of
    // C before the layer's kernels). Decides whether C is ordered behind U without the I4 wait, and where
    bool la_probe = false;
    std::vector<std::pair<uint64_t, int>> la_probe_end; // batch seq, index of its end event on U (this timed round)
    struct la_probe_t { int te, c0, c1; };
    std::vector<la_probe_t> la_probe_q;
    uint64_t la_probe_unmatched = 0; // I4 copies in timed rounds whose batch end event is not of this round
    // C18c AMB_MOE_LOOKAHEAD_DEBUG_DEFER=1 (with DEBUG, default off): the copies (and the event record) of a batch are
    // enqueued on U by the host only at the copy() two calls later (C has finished the layer that reads the batch), or right
    // before an I4 wait that needs them, or at a drain. Victims are poisoned at issue. A negative control that does not
    // depend on how the GPU orders the two streams: with SKIP_WAIT the layer reads poisoned slots
    bool la_defer = false;
    int64_t la_copy_calls = 0;
    struct la_defer_t {
        uint64_t seq = 0;
        int32_t  dev = 0;
        int64_t  call = 0;
        std::vector<void *> dsts;
        std::vector<const void *> srcs;
        std::vector<size_t> sizes;
        std::vector<ggml_tensor *> banks;
    };
    std::deque<la_defer_t> la_defer_q;
    uint64_t la_deferred = 0, la_defer_fl_wait = 0, la_defer_fl_late = 0, la_defer_fl_drain = 0;

    // AMB_MOE_MISS_OVERLAP=1 (C23, default off): overlap the miss uploads of layer L with the tail of the split before
    // the lookup of layer L.
    // Graph (llama_moe_cache::build_lookup, called by build_moe_ffn; qwen4exp.cpp): for ubatches of mo_min_tok..mo_max_tok tokens, right
    // after the router/top-k of layer L a GGML_OP_AMB_PUBLISH node writes the selected ids of L to mapped pinned memory
    // (slot 2L) and bumps a sequence flag; then come the predictor of L + 1 with its own publish (slot 2(L+1) + 1) and the
    // shared expert of L; then the lookup of L starts the next split as before.
    // Host: ids_post_split counts the publish nodes of every launched split (mo_expect); the scheduler skips its
    // synchronize before copy(L) when a publish of L is pending (copy_nosync); copy(L) polls the flag (mo_wait) instead of
    // waiting for the whole split, plans as before, enqueues the slot map upload on C (stream-ordered after the tail),
    // issues the miss uploads on U (not C), records mo_ev on U and makes C wait on it (GPU side) before the layer.
    // Stream invariants (C compute stream, U upload stream; they replace I1 for overlap layers, I2..I6 hold as stated):
    //  M1 when copy(L) reads the published ids of L, every C kernel that reads an expert slot (the MUL_MAT_ID of layers
    //     < L) has completed: the publish kernel is a plain launch after them (amb-publish.cu). The C work still
    //     running (the tail: predictor, shared expert) reads and writes no slot and no slot map
    //  M2 the slot map of L is uploaded on C, so it is ordered after the tail of C (its device copy may share memory with
    //     tensors of the tail) and before the kernels of L
    //  M3 the miss uploads of L write only the fills of plan(L): victims that are not in flight, not selected by this plan
    //     (I2/I3 as before); no C kernel enqueued before copy(L) reads them (M1); C reads them only after waiting on
    //     mo_ev, which is recorded on U after them (the wait site in copy())
    //  M4 U is in order: the lookahead batch of L + 1, issued after copy(L), follows the miss uploads of L on U; the
    //     LINK_WAIT event on C is not used (it would order U behind the tail of C)
    //  M5 a C synchronize implies that every miss upload on U has completed (C waited on each mo_ev); lookahead_drain also
    //     synchronizes U when miss uploads were issued since (mo_dirty), before any bank write/move/park
    //  M6 the publish flag carries a sequence number per slot (device counter): copy(L) waits for exactly the number of
    //     publishes of the slot the host has launched, so a value left from an earlier, unconsumed publish is never taken
    bool     mo_req     = false; // AMB_MOE_MISS_OVERLAP=1 or 2 set
    bool     mo_build   = false; // the graphs get the publish nodes and the reordered tail (configuration, also --fit)
    bool     mo_on      = false; // runtime state present (mapped slots, counters, event, upload stream)
    int32_t  mo_min_tok = 2;     // AMB_MOE_MISS_OVERLAP_MIN_TOK (1..8)
    int32_t  mo_max_tok = 8;     // the smallest MMVQ limit of the cached expert types (la_drain_above)
    bool     mo_cmp     = false; // AMB_MOE_MISS_OVERLAP_DEBUG=1: also the old readback, compared (mismatches counted)
    uint64_t mo_spin_ns = 0;     // AMB_MOE_MISS_OVERLAP_DEBUG_SPIN_US=N: poison the fills (0xFF) and delay U before them
    bool     mo_skip_wait = false; // AMB_MOE_MISS_OVERLAP_DEBUG_SKIP_WAIT=1: NEGATIVE CONTROL, C does not wait on mo_ev
    void *   mo_host  = nullptr;   // slot k at mo_host + k*GGML_AMB_PUBLISH_SLOT: [0] flag, [32..] ids (int32)
    void *   mo_sdev  = nullptr;   // device address of slot 0
    void *   mo_cdev  = nullptr;   // device counters, one uint32 per slot
    ggml_backend_t mo_be = nullptr; // the device backend the buffers belong to
    ggml_backend_event_t mo_flush = nullptr; // queried while a poll is long (a driver flush), the device's mo_ev
    typedef void (*mo_free_fn_t)(ggml_backend_t, void *, void *);
    mo_free_fn_t mo_free_fn = nullptr;
    std::vector<uint32_t> mo_expect; // [2*n_layer] publishes launched per slot (cumulative, wraps)
    std::vector<uint32_t> mo_taken;  // [2*n_layer] the publish count consumed last
    std::vector<int32_t>  mo_n;      // [2*n_layer] elements of the last launched publish of the slot
    std::vector<int32_t>  mo_ntok;   // [2*n_layer] rows (ne[1]) of its source tensor
    std::vector<int32_t>  mo_dbg;    // DEBUG scratch
    bool     mo_layer = false;       // the copy() running now uses the published ids
    struct mo_acc {
        uint64_t pub_ids = 0, pub_pred = 0;     // publish nodes launched (ids / predicted ids)
        uint64_t ids_taken = 0, pred_taken = 0; // consumed by copy() / by a lookahead issue
        uint64_t ready = 0;                     // flag already set when the host looked
        uint64_t polled = 0;                    // the host had to poll
        int64_t  poll_ns = 0, poll_max_ns = 0;  // host time spent polling
        uint64_t pred_missing = 0;              // lookahead issue without a pending predicted publish (not issued)
        uint64_t no_publish = 0;                // copy() of an overlap-size ubatch without a launched publish (old path)
        uint64_t miss_batches = 0, miss_bytes = 0; // miss uploads on U
        uint64_t c_waits = 0, skipped_waits = 0;   // C waits on mo_ev; omitted by the negative control
        uint64_t dbg_cmp = 0, dbg_bad = 0, dbg_pred_cmp = 0, dbg_pred_bad = 0; // DEBUG comparisons and mismatches
        uint64_t poisoned = 0;                  // DEBUG_SPIN_US: fill runs poisoned before their upload
        uint64_t flushes = 0;                   // event queries issued during long polls (C24: every 100 us after 100 us)
        // C24
        uint64_t flush_released = 0;            // polls that ended within 20 us after a flush query (WDDM held the work)
        uint64_t poll_hist[5] = {};             // polls ending < 10 us, < 50 us, < 200 us, < 1 ms, >= 1 ms
        uint64_t disabled = 0;                  // runtime disables (mo_runtime_disable)
        uint64_t s2_layers = 0;                 // mode 2: split-phase copies that used the published ids
        uint64_t s2_fallback = 0;               // mode 2: split-phase copies on the synchronized path (all experts hit phase)
        uint64_t s2_selected = 0, s2_pending = 0; // distinct selected experts / of them in the miss phase
        uint64_t s2_records = 0;                // gate events recorded on U
        uint64_t s2_skipped = 0;                // gate events not recorded (negative control)
    };
    mo_acc mo_st;
    mo_acc mo_prev;
    mo_acc mo_prev_c24;          // C24 fields at the last interval line (mo_prev is reset by the first line)
    int64_t mo_prev_round = 0;
    // AMB_MOE_MISS_OVERLAP=2 (C24): the hit/miss split inside a layer. The graph looks up slot_map2 (two rows); the hit
    // phase computes the experts whose slots are complete; GGML_OP_AMB_GATE makes C wait (GPU side) on mo2_ev[L], which
    // copy(L) records on U after the miss uploads of L; the miss phase computes the fills of plan(L) and the selected slots
    // whose lookahead upload may still run. Invariants (in addition to M1..M6):
    //  M7 every (token, slot) pair is computed by exactly one phase: copy() puts each selected expert in exactly one row,
    //     the other row holds -1, and the masked MoE kernel writes -0.0f for -1 (x + -0.0f == x), so the ADD of the two
    //     phases equals the single-phase output bit for bit, and the weighted reduction after it is unchanged
    //  M8 the hit phase reads only slots whose content is complete before copy(L) (not filled by plan(L), not in flight);
    //     the miss phase reads its slots after the gate; mo2_ev[L] is recorded on U after every upload of plan(L) and,
    //     U being in order, after every earlier lookahead batch, so it also covers the in-flight slots (the I4 case); C
    //     therefore needs no wait before the split (the I4 and mo_ev waits of mode 1 are not issued for such layers)
    //  M9 a layer whose miss phase is empty records nothing: its gate finds the last record of mo2_ev[L], complete since
    //     the previous round (C ran graph L of that round before the publish that started this copy(L))
    int32_t  mo_mode    = 0;     // 0 off, 1 C23, 2 C24
    // AMB_MOE_HIT_FIRST=1 (D2, default off): the hit phase of a MoE layer is launched before the host has read the ids.
    // The graph of a 2..hf_max_tok token ubatch looks up two maps per cached layer, each a host weight, so the scheduler
    // starts a split at each lookup:
    //   split H: lookup(slot_map_h) + the expert chain on the experts cached BEFORE this ubatch's plan (masked ids: -1
    //            rows write -0.0f and read no weights, the C24 kernels)
    //   split R: lookup(slot_map_r) + the same chain on the experts the plan filled + the ADD of both + the rest of the
    //            block up to the next router
    // Order on the compute stream C (the only stream): R(L-1) | ids readback + event (ids_post_split) | upload map_H |
    // H(L) | fills of L | upload map_R | R(L). The host: copy(map_H) runs without the scheduler's synchronize (it only
    // snapshots the host slot map), then copy(map_R) waits on the ids event alone, plans, enqueues fills and map_R.
    // While the host does that, C runs H(L): an all-hit layer has no idle compute stream.
    // Invariants:
    //  H1 the plan reads the same ids at the same point of the host program as without the switch (LRU state, victims,
    //     fills and statistics are identical; only the stream position of the ids wait changes)
    //  H2 every selected expert is computed by exactly one phase: cached before the plan <=> slot in map_H (the plan
    //     never moves the slot of a hit), not cached before <=> filled by this plan <=> slot in map_R (no CPU share, no
    //     lookahead / prefetch: both refused). Verified on the host for every plan (abort on a violation)
    //  H3 H reads only slots of selected experts; the plan never evicts a selected expert, and the fills are enqueued on
    //     C after H anyway: no kernel reads a slot while it is written
    //  H4 the staging of map_H / map_R of layer il is rewritten one evaluation later, after C executed the upload that
    //     read it (a later ids event of the same evaluation was waited for, or the evaluation was synchronized)
    // With AMB_IDS_ASYNC off or refused at run time the graphs keep the two lookups and both copies take the scheduler's
    // synchronize (correct, no gain). Debug: _CHECK=1 (ids and device map compared after a synchronize), _SYNC=1 (the
    // split structure with the synchronizes of before: its cost alone), negative controls _DEBUG_SKIP_R=1 (the miss
    // phase of a layer with fills is masked out: the output must change) and _DEBUG_NO_WAIT=1 (the ids event wait is
    // skipped: stale ids, the H2 check must abort)
    bool     hf_build   = false; // the graphs get the two lookups (configuration only, also with no_alloc)
    bool     hf_on      = false; // runtime: no synchronize before the copies (needs the async ids and pinned staging)
    int32_t  hf_min_tok = 2;
    int32_t  hf_max_tok = 8;
    bool     hf_check   = false;
    bool     hf_sync    = false;
    bool     hf_skip_r  = false;
    bool     hf_no_wait = false;
    ggml_backend_buffer_ptr hf_pinned;
    std::vector<int32_t *>  hf_stage_h;          // [n_layer] n_expert int32 (pinned), nullptr if not cached
    std::vector<int32_t *>  hf_stage_r;
    std::vector<std::vector<int32_t>> hf_host;   // [2*n_layer] the same without pinned memory (synchronized path)
    int32_t  hf_h_il    = -1;      // copy(map_H) of this layer ran for the split being launched (-1: none)
    int32_t  hf_h_last  = -1;      // the layer whose map_H snapshot is current (for H2)
    ggml_tensor * hf_h_dst = nullptr; // the device copy of that map_H (check mode)
    std::vector<int32_t> hf_tmp;
    struct hf_stats {
        uint64_t h_copies = 0, r_copies = 0, r_allhit = 0, r_miss = 0, r_async = 0, r_sync = 0;
        uint64_t chk_ids = 0, chk_ids_bad = 0, chk_map = 0, chk_map_bad = 0, skipped_r = 0, no_wait = 0;
    };
    hf_stats hf_st;
    ggml_backend_buffer_ptr mo2_pinned;          // the staging of the split-phase slot maps
    std::vector<int32_t *>  mo2_stage;           // [n_layer] 2*n_expert int32, nullptr if not cached
    std::vector<ggml_backend_event_t> mo2_ev;    // [n_layer] gate events
    bool     mo2_stamps = false; // AMB_STEP_EVENTS: the gate ops time their wait into the stamp slots
    std::vector<uint64_t> mo2_stamp_prev;        // per layer: sum / count at the last log (2 values per layer)

    // C25 (default off, only with AMB_MOE_MISS_OVERLAP=1): the misses of layer L ahead of speculative lookahead bytes.
    // Queueing today (mode 1, ISSUE=post): copy(L) puts miss(L) on U; right after the launch of split L the whole lookahead
    // batch of L + 1 goes on U in ONE batched call (up to the per-layer budget); copy(L + 1) puts miss(L + 1) on U behind
    // it; U is in order, so miss(L + 1), and C's wait on mo_ev, include whatever of that batch is still on the link.
    // AMB_MOE_LA_CHUNK_MIB=N (1..64): the lookahead of L + 1 is planned as before (same candidates, same order, same
    // budget) but admitted and uploaded lazily, in chunks of <= N MiB: the first chunk at the issue point as before, the
    // next ones by the host while it polls for the router publish of L + 1 (mo_wait), one chunk on U at a time, and only if
    // the chunk is predicted to finish before that publish (per-layer EWMA of the host time from the issue to the publish,
    // AMB_MOE_LA_CHUNK_GBPS link estimate, AMB_MOE_LA_CHUNK_SLACK_US; AMB_MOE_LA_CHUNK_DEADLINE=0 feeds until the
    // publish, a control). Candidates not uploaded when the publish arrives are never admitted: the selected ones are
    // ordinary misses of plan(L + 1) ("promoted", they go first on the link), the others cost nothing ("dropped").
    // So miss(L + 1) waits behind at most one chunk. Invariants (in addition to I1..I6, M1..M6):
    //  K1 a chunk is planned and admitted at its issue, with the victim rule of lookahead_issue (I2) on a cache state that
    //     differs from the one at the first chunk only by this lookahead's own admissions and newer completion knowledge
    //     (pf_done): no plan() runs between the first chunk and copy(L + 1)'s plan (feeding only in mo_wait of L + 1, same
    //     round, before the plan); la_mark_gen is unchanged in between (no other lookahead_issue)
    //  K2 un-issued candidates hold no slot and no slot map entry, so dropping them needs no bookkeeping or GPU work
    //  K3 every chunk is a batch with its own seq and event (I4/I5 unchanged); the ring has 16 events with chunking
    // AMB_MOE_MISS_STREAM=1: the miss uploads (and mo_ev) on a third stream M instead of U, so they are not behind U's
    // lookahead copies in stream order. Whether the driver then interleaves or still serialises H2D copies of two streams
    // is not known here (C18c's probe saw C's H2D copy finish after U's batch): a measurement question. Invariants:
    //  S1 C waits on mo_ev (M) for the fills and, by I4, on la_ev (U) for selected in-flight lookahead slots: every byte
    //     layer L reads is covered by an event C waits on (before, mo_ev on U covered both)
    //  S2 no write-after-write between M and U: M writes only fills (victims not in flight, I3); U writes only lookahead
    //     victims, which are not recently used (keys selected in this or the previous round, I2), and every fill key is
    //     selected in its round; M is synchronized at the start of every round (copy() of the first cached layer), so a
    //     slot M wrote is complete before it can be a lookahead victim
    //  S3 M5 for M: lookahead_drain synchronizes M when mo_dirty; a C synchronize implies M is complete (C waited on each
    //     mo_ev)
    // AMB_MOE_C25_STATS=1: one extra event query per miss batch (diagnostic, host cost): was the newest lookahead batch
    // still on U when the miss batch was issued; with the switches above or this one set, a amb_c25 line is logged
    double   ck_mib   = 0.0;     // AMB_MOE_LA_CHUNK_MIB as given
    bool     ck_on    = false;   // chunking active (mode 1 on, accepted in mo_init)
    size_t   ck_bytes = 0;       // chunk size in bytes
    bool     ck_deadline = true; // AMB_MOE_LA_CHUNK_DEADLINE
    double   ck_gbps  = 35.0;    // AMB_MOE_LA_CHUNK_GBPS (C56: about 35 GB/s effective for the miss batches)
    int64_t  ck_slack_ns = 30000; // AMB_MOE_LA_CHUNK_SLACK_US
    bool     ms_req   = false;   // AMB_MOE_MISS_STREAM=1 set
    bool     c25_stats = false;  // AMB_MOE_C25_STATS=1
    struct ck_state {
        bool    active = false;  // a chunked lookahead whose candidates are not all processed
        int32_t il = -1, il_real = -1;
        int64_t n_tok = 0;
        int64_t round = 0;
        int32_t ig = -1;
        std::vector<int32_t> cand; // the candidates in lookahead order
        size_t  pos = 0;         // the next candidate to consider
        size_t  max_n = 0;       // budget in experts (as lookahead_issue)
        size_t  n_issued = 0;    // experts admitted so far (all chunks)
        size_t  n_chunk = 1;     // experts per chunk
        size_t  expert_bytes = 0;
        int64_t t_chunk_ns = 0;  // predicted link time of one chunk
        clk::time_point t_iss;   // host time of the first chunk
        bool    timing = false;  // t_iss valid for the EWMA of this layer
        bool    stopped = false; // the deadline stopped the feeding
    };
    ck_state ck;
    std::vector<int32_t> ck_left;    // the candidates left when the publish arrived (counted after the plan)
    int32_t  ck_left_il = -1;
    std::vector<int64_t> ck_d_ns;    // [n_layer] EWMA of the host time from the first chunk to the publish, 0 = unknown
    struct c25_acc {
        uint64_t ck_layers = 0;      // chunked lookahead issues
        uint64_t ck_first = 0, ck_fed = 0; // chunks issued at the issue point / by the feeder during the poll
        uint64_t ck_fed_bytes = 0;
        uint64_t ck_stop_deadline = 0; // layers whose feeding the deadline stopped
        uint64_t ck_queries = 0;     // completion queries by the feeder
        uint64_t ck_left = 0;        // candidates not uploaded when the publish arrived (within the budget)
        uint64_t ck_promoted = 0;    // of them selected by the real plan (now misses)
        uint64_t ck_dropped = 0;     // of them not selected (never on the link)
        uint64_t ck_saved_bytes = 0; // bytes of the dropped ones
        uint64_t ck_abandoned = 0;   // chunked lookaheads cleared before their copy() (drain, another layer, new issue)
        uint64_t ms_batches = 0, ms_bytes = 0; // miss batches on M
        uint64_t ms_round_syncs = 0; // M synchronized at a round start
        uint64_t st_miss = 0, st_behind = 0; // C25_STATS: miss batches checked / issued while the newest lookahead batch was on U
    };
    c25_acc c25_st;
    c25_acc c25_prev;

    struct la_acc {
        uint64_t rounds = 0;      // evaluations of <= la_max_tok tokens through the cache
        uint64_t evals = 0;       // layer evaluations with predicted ids (lookahead planned)
        uint64_t predicted = 0;   // distinct predicted experts
        uint64_t cached = 0;      // ... already cached or in flight
        uint64_t issued = 0;      // experts uploaded (DRY: would have been)
        uint64_t bytes = 0;
        uint64_t batches = 0;     // batched copies on U
        uint64_t capped = 0;      // skipped by the byte budget
        uint64_t no_victim = 0;   // skipped for want of an eligible victim
        uint64_t bad = 0;         // predicted id lists out of range or unexpected
        uint64_t no_readback = 0; // layer-evals that should have had predicted ids but got no async readback
        uint64_t useful_done = 0; // real selections served by a prefetched slot whose upload was known complete
        uint64_t useful_wait = 0; // ... possibly in flight: C waited on the batch event
        uint64_t dry_useful = 0;  // DRY: would-be uploads that were a real miss of their layer in the same round
        uint64_t blocking = 0;    // experts the real plans still uploaded on C (blocking misses)
        uint64_t covered = 0;     // blocking misses in layers for which a lookahead was planned in the round
        uint64_t waits = 0;       // GPU-side waits of C on U
        int64_t  ns_issue = 0;    // host time of the lookahead planning and issue (behind the scheduler's sync)
        int64_t  ns_other = 0;    // host time of the event queries and the waits
        uint64_t timed = 0;       // rounds with timing events read
        uint64_t timed_lost = 0;  // ... not complete when read, dropped
        double   up_ms = 0.0;     // GPU time of the batches on U
        double   wait_ms = 0.0;   // time C spent at its waits on U (event before/after the wait)
        // C18c ORDER_PROBE (signed GPU times relative to the end of the needed batch on U)
        uint64_t pr_n = 0;        // probed I4 copies
        uint64_t pr_c0_after = 0; // ... whose C idle point (before the blocking copies) completed after U's batch end
        uint64_t pr_c1_after = 0; // ... whose slot map upload completed after U's batch end
        uint64_t pr_c1_near = 0;  // ... of those, within 0.1 ms after it
        double   pr_d0_ms = 0.0;  // sum of c0 - U end
        double   pr_d1_ms = 0.0;  // sum of c1 - U end
    };
    la_acc la_st[2];   // [0] 1-token ubatches, [1] 2..la_max_tok tokens
    la_acc la_prev[2]; // at the previous periodic log
    std::vector<std::pair<const char *, uint64_t>> la_drains; // reason, count
    uint64_t la_unused_at_drain = 0;
    uint64_t la_post_issued = 0, la_post_dropped = 0; // ISSUE=post: issues run from the post-launch hook / dropped
    uint64_t la_skipped_waits = 0;                    // DEBUG_SKIP_WAIT: waits omitted (negative control)
    // stats 7: global counters at the previous periodic line, for interval values
    uint64_t la_prev_wasted = 0, la_prev_first_victims = 0, la_prev_unused_at_drain = 0;
    uint64_t la_hit_done_mark = 0, la_hit_wait_mark = 0; // lru counters before the current plan()

    static const char * predict_name(int p) {
        switch (p) {
            case 0: return "P0 own router input (sanity)";
            case 1: return "P1 FFN input of L-1";
            case 2: return "P2 attention input of L";
            case 3: return "P3 FFN input of L-2";
            case 4: return "P4 FFN mixer of L on the residual before the FFN mixer of L-1";
            case 5: return "P5 FFN mixer of L on the residual before the attention mixer of L";
            default: return "?";
        }
    }

    // AMB_ELASTIC_KV level 3: the banks on one elastic buffer, lent KV chunks extend their tails (see el_lender)
    bool   el      = false;
    bool   el_live = false; // el and allocated (not the --fit pass)
    size_t el_gran = 0;
    ggml_backend_buffer_type_t el_buft = nullptr;
    const llama_elastic_fns * el_fns = nullptr;
    uint64_t el_dropped = 0; // experts dropped from parked slots
    uint64_t el_given   = 0; // chunks received
    uint64_t el_taken   = 0; // chunks returned
    // C4 verification modes (llama-elastic.h): AMB_ELASTIC_CHECK=1 slot-map checks, AMB_ELASTIC_POISON=1 fill of the
    // slots that become active through lending (0xFF, the first el_poison_head bytes of each slot zero)
    bool     el_check   = false;
    size_t   el_floor_bytes = 0; // C4b: AMB_ELASTIC_FLOOR_MIB, 0 = off
    bool     el_poison  = false;
    uint64_t el_checks  = 0;
    uint64_t el_poisoned = 0; // slots filled
    static constexpr size_t el_poison_head = 512;
    typedef bool (*el_memset2d_fn_t)(ggml_backend_t, void *, size_t, int, size_t, size_t);
    el_memset2d_fn_t el_memset2d = nullptr;
    std::vector<llama_kv_elastic *> el_borrowers;

    struct el_lender_t : public llama_elastic_lender {
        impl * p = nullptr;
        size_t give(ggml_backend_buffer_t src, const size_t * chunks, size_t n) override { return p->el_give(src, chunks, n); }
        void   take(ggml_backend_buffer_t dst, const size_t * chunks, size_t n) override { p->el_take(dst, chunks, n); }
        std::string describe() const override { return p->el_describe(); }
        void   attach(llama_kv_elastic * b) override { p->el_borrowers.push_back(b); }
        size_t lent_chunks() const override {
            size_t res = 0;
            for (const group & g : p->groups) {
                if (g.n_slots > 0) {
                    res += impl::el_lent(g);
                }
            }
            return res;
        }
        int64_t lent_slots() const override {
            int64_t res = 0;
            for (const group & g : p->groups) {
                if (g.n_slots > 0) {
                    res += g.lru.n_active - g.n_base;
                }
            }
            return res;
        }
        uint64_t dropped() const override { return p->el_dropped; }
        size_t give_spans(ggml_backend_buffer_t src, const size_t * heads, size_t n, size_t * n_calls) override {
            return p->el_give_spans(src, heads, n, n_calls);
        }
        void take_spans(ggml_backend_buffer_t dst, const size_t * heads, const size_t * lens, size_t n, size_t * n_calls) override {
            p->el_take_spans(dst, heads, lens, n, n_calls);
        }
        size_t n_spans() const override { return p->el_n_spans(); }
        void set_span_evict(void (*fn)(void *), void * ud) override {
            p->el_span_evict    = fn;
            p->el_span_evict_ud = ud;
        }
    };
    el_lender_t el_lender;

    // C17c: spans (whole mappings of the compute buffer, AMB_ELASTIC_COMPUTE) sit on top of each bank's lent chunks;
    // the KV give/take paths first ask their owner to take them back (el_span_evict)
    void (*el_span_evict)(void *) = nullptr;
    void * el_span_evict_ud = nullptr;

    // AMB_PF_OVERLAP + AMB_ELASTIC_KV=3 (C13): the overlap upload ring lives in a fixed slot range [r0, r1) at the top
    // of the own (base) slots of one or more groups, parked for the duration of a large-batch graph only
    struct ring_part {
        int32_t ig;
        int32_t r0;
        int32_t r1;
    };
    std::vector<ring_part> ring_parts;
    std::vector<void *>    ring_ptrs;
    size_t   ring_slot   = 0;     // stride of the ring slots (the overlap slot size, rounded up to 256 bytes)
    bool     ring_out    = false; // borrowed now
    uint64_t ring_n_borrow = 0;
    uint64_t ring_dropped  = 0;   // experts dropped from the ring range when it was borrowed

    // C13c: ring_return clears the first ring_head_zero bytes of every ring-range slot of every bank (one strided memset per
    // bank on the compute stream, CUDA proc address "ggml_backend_cuda_memset2d_async"); AMB_RING_POISON (debug, default
    // off): 1 = fill the whole ring range with 0xFF first, 2 = fill with 0xFF and skip the clearing (reproduces the hazard)
    typedef bool (*memset2d_fn_t)(ggml_backend_t, void *, size_t, int, size_t, size_t);
    static constexpr size_t ring_head_zero = 512;
    memset2d_fn_t ring_memset2d = nullptr;
    int           ring_poison   = 0;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share, bool elastic_layout, size_t lend_bytes) :
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()), cpu_share(std::clamp(cpu_share, 0.0f, 1.0f)) {
        el_lender.p = this;
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // AMB_MOE_LOOP_DEDUP: decided from the environment and the model only, so the --fit pass sizes the same split
        {
            const char * why = nullptr;
            loop_dedup = llama_moe_cache_loop_dedup_env(&why);
            if (why != nullptr) {
                LLAMA_LOG_WARN("%s: amb_loop_dedup: refused, off: %s\n", __func__, why);
            }
            loop_dedup = loop_dedup && model.hparams.loop_n_pass > 1;
            rep = llama_moe_cache_loop_rep(model, loop_dedup);
            loop_dedup_check = loop_dedup && [] {
                const char * v = getenv("AMB_MOE_LOOP_DEDUP_CHECK");
                return v != nullptr && strcmp(v, "1") == 0;
            }();
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            // AMB_MOE_LOOP_DEDUP: a later visit shares the experts of its representative only if that one is cached in the
            // same group (same device, same layout) with the same host data; otherwise it stays its own physical layer
            if (rep[il] != (int32_t) il) {
                const int32_t r = rep[il];
                bool same = std::find(it->layers.begin(), it->layers.end(), r) != it->layers.end();
                const auto r_experts = llama_moe_cache_layer_experts(model.layers[r]);
                same = same && r_experts.size() == experts.size();
                for (size_t k = 0; same && k < experts.size(); ++k) {
                    same = r_experts[k]->data == experts[k]->data && ggml_nbytes(r_experts[k]) == ggml_nbytes(experts[k]);
                }
                if (!same) {
                    rep[il] = (int32_t) il;
                }
            }
            it->layers.push_back(il);
            if (rep[il] != (int32_t) il) {
                continue; // the physical tensors are counted with the representative
            }
            it->n_phys++;
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // with misses on the CPU, the n_zero slots after the cached ones stay zero: the device computes these misses as zero
        // one extra slot at the end, CUDA MMQ can read past the last expert
        n_zero = cpu_share > 0.0f ? std::min<int32_t>(n_expert_used, 64) : 0;
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + n_zero + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        cpu_share_q = (uint32_t) std::lround(cpu_share*65536.0f);
        if (const char * path = getenv("LLAMA_MOE_CACHE_TRACE"); path != nullptr && *path != '\0') {
            trace = fopen(path, "wb");
            if (trace == nullptr) {
                throw std::runtime_error(std::string("failed to open the MoE cache trace ") + path);
            }
            write_trace_header(split_sum);
        }
        // AMB_ELASTIC_KV level 3: the banks on an elastic buffer, so lent KV chunks can extend their tails
        // decided from the environment and the configuration only, so the --fit pass (no_alloc) sizes the same base
        if (elastic_layout) {
            const char * why = nullptr;
            const char * pf  = getenv("AMB_MOE_PREFETCH");
            int n_dev_host = 0;
            for (const device & d : devices) {
                n_dev_host += d.host_bytes > 0 ? 1 : 0;
            }
            if (cpu_share > 0.0f) {
                why = "the cache has a CPU share (--moe-cache-cpu-share)";
            } else if (trace != nullptr) {
                why = "LLAMA_MOE_CACHE_TRACE is set";
            } else if (pf != nullptr && (atoi(pf) == 1 || atoi(pf) == 4)) {
                why = "AMB_MOE_PREFETCH is on";
            } else if (devices.size() != 1 || n_dev_host != 1) {
                why = "more than one GPU";
            } else {
                el_buft = llama_elastic_buft_of(devices[0].buft);
                el_fns  = el_buft ? &llama_elastic_fns_of(ggml_backend_buft_get_device(el_buft)) : nullptr;
                el_gran = el_fns && el_fns->resolved ? el_fns->buft_granularity(el_buft) : 0;
                if (el_gran == 0) {
                    why = "no CUDA virtual memory";
                }
                // quantized tensors whose rows are not a multiple of MATRIX_ROW_PADDING (512) get a tail pad of less than
                // one row (TAG_ALLOC_SIZE_EXPAND, MMQ reads up to that far past the last row): it is reserved at the end of
                // the bank's range (see el_alloc), and the reads past the last active slot land in the mapped guard slot
                for (const group & g : groups) {
                    for (const ggml_tensor * t : g.ref) {
                        const size_t pad = ggml_backend_buft_get_alloc_size(devices[0].buft, t) - ggml_nbytes(t);
                        if (pad > 0) {
                            LLAMA_LOG_INFO("%s: amb_elastic: %s (%s, ne0 = %lld) has a %zu byte row pad, the slot is %zu bytes\n", __func__,
                                t->name, ggml_type_name(t->type), (long long) t->ne[0], pad, (size_t) t->nb[2]);
                        }
                        if (!why && pad >= (size_t) t->nb[2]) {
                            why = "an expert tensor has a row pad of a whole slot or more";
                        }
                    }
                }
            }
            if (why) {
                LLAMA_LOG_WARN("%s: amb_elastic: no elastic MoE cache banks: %s (no lending: AMB_ELASTIC_KV=3 stays at level 2, "
                    "AMB_ELASTIC_COMPUTE is refused)\n", __func__, why);
            } else {
                el = true;
                el_floor_bytes = llama_elastic_floor_bytes();
                el_check  = llama_elastic_check_env() != 0;
                el_poison = llama_elastic_poison_env() != 0;
                if (el_poison) {
                    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(devices[0].backend));
                    el_memset2d = reg ? (el_memset2d_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_memset2d_async") : nullptr;
                    if (el_memset2d == nullptr) {
                        LLAMA_LOG_WARN("%s: amb_elastic: AMB_ELASTIC_POISON=1: the backend has no ggml_backend_cuda_memset2d_async, "
                            "lent slots are not poisoned\n", __func__);
                        el_poison = false;
                    }
                }
            }
        }
        // chunks of the banks of g with n slots (and the over-read slot), each bank starting on a chunk boundary
        auto el_chunks_b = [&](size_t nb2, int64_t n) { return (nb2*(size_t) (n + 1) + el_gran - 1)/el_gran; };
        auto el_chunks = [&](const group & g, int64_t n) {
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += el_chunks_b(t->nb[2], n);
            }
            return res;
        };

        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        // AMB_MOE_MISS_OVERLAP=2 (C24): the split-phase slot maps are created whenever the variable asks for mode 2
        const bool mo2_env = [] {
            const char * v = getenv("AMB_MOE_MISS_OVERLAP");
            return v != nullptr && strcmp(v, "2") == 0;
        }();
        // AMB_MOE_HIT_FIRST (D2): its two slot maps per layer are created whenever the variable is set (also in --fit)
        const bool hf_env = [] {
            const char * v = getenv("AMB_MOE_HIT_FIRST");
            return v != nullptr && strcmp(v, "1") == 0;
        }();
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.n_phys*n_expert; // == layers.size()*n_expert without AMB_MOE_LOOP_DEDUP
            if (el) {
                // the base: the slots the budget pays for with whole chunks (at most one chunk per bank less than without)
                while (g.n_slots < max_slots && el_chunks(g, g.n_slots + 1)*el_gran <= budget) {
                    g.n_slots++;
                }
            } else {
                while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                    g.n_slots++;
                }
            }
            // C4b: the slots of the compute floor this group is guaranteed: its share (by the size of its experts) of the
            // floor, less one span per bank and one more for the greedy span placement (llama_moe_cache el_give_spans)
            int32_t n_floor = 0;
            if (el && el_floor_bytes > 0) {
                size_t slot_bytes = 0;
                for (const ggml_tensor * t : g.ref) {
                    slot_bytes += t->nb[2];
                }
                const double share = (double) g.host_bytes/d.host_bytes;
                const double loss  = (double) (g.ref.size() + 1)*llama_elastic_span_bytes(el_gran);
                const double avail = (double) el_floor_bytes*share - loss;
                n_floor = avail > 0.0 ? (int32_t) std::min<double>(avail/slot_bytes, (double) max_slots) : 0;
            }
            if (g.n_slots + n_floor < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.n_base  = g.n_slots;
            g.n_graph = std::min(max_slots, g.n_base + n_floor);
            if (el) {
                // the capacity: the base plus this group's share of the lendable chunks (by the size of its experts)
                // C4b: the floor is address space here too (in the --fit pass as well, so both passes build the same graphs)
                const size_t base_chunks = el_chunks(g, g.n_base);
                const size_t lend_chunks = (size_t) ((double) (std::max(lend_bytes, el_floor_bytes)/el_gran)*g.host_bytes/d.host_bytes) +
                    (el_floor_bytes > 0 ? 2*g.ref.size()*(llama_elastic_span_bytes(el_gran)/el_gran) : 0);
                int32_t n_cap = g.n_base;
                while (n_cap < max_slots && el_chunks(g, n_cap + 1) <= base_chunks + lend_chunks) {
                    n_cap++;
                }
                g.n_slots = n_cap;
                g.eb.clear();
                for (const ggml_tensor * t : g.ref) {
                    group::ebank b;
                    b.nb2    = t->nb[2];
                    b.m_base = el_chunks_b(b.nb2, g.n_base);
                    // the range also holds the tail row pad of the tensor (never read: see the refusal checks above)
                    const size_t pad = ggml_backend_buft_get_alloc_size(d.buft, t) - ggml_nbytes(t);
                    b.m_cap  = (b.nb2*(size_t) (n_cap + 1) + pad + el_gran - 1)/el_gran;
                    b.m      = b.m_base;
                    g.eb.push_back(b);
                }
                g.lru.init(model.layers.size(), n_expert, g.n_slots, el_active(g));
            } else {
                g.lru.init(model.layers.size(), n_expert, g.n_slots);
            }
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size()*(n_zero > 0 ? 3 : 1); // slot maps and CPU map
            n_tensors_host  += mo2_env ? g.layers.size() : 0;          // C24 split-phase slot maps
            n_tensors_host  += hf_env ? 2*g.layers.size() : 0;         // D2 hit-first slot maps
        }
        if (el) {
            // bank offsets: consecutive, each on a chunk boundary
            size_t off = 0;
            for (group & g : groups) {
                for (auto & b : g.eb) {
                    b.off = off;
                    off  += b.m_cap*el_gran;
                }
            }
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + n_zero + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots + n_zero, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = { il, false };
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                if (mo2_env) {
                    // AMB_MOE_MISS_OVERLAP=2 (C24): created from the environment alone (also in the --fit pass); used only
                    // if the switch is accepted (mo_config)
                    l.slot_map2 = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 2, g.ref[0]->ne[2]);
                    ggml_format_name(l.slot_map2, "moe_cache.slot_map2-%d", il);
                    layer_of[l.slot_map2] = { il, false };
                    buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map2), alignment_host);
                }
                if (hf_env) {
                    l.slot_map_h = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.slot_map_h, "moe_cache.slot_map_h-%d", il);
                    layer_of[l.slot_map_h] = { il, false };
                    buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map_h), alignment_host);
                    l.slot_map_r = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.slot_map_r, "moe_cache.slot_map_r-%d", il);
                    layer_of[l.slot_map_r] = { il, false };
                    buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map_r), alignment_host);
                }
                if (n_zero > 0) {
                    l.slot_map_cpu = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.slot_map_cpu, "moe_cache.slot_map_cpu-%d", il);
                    layer_of[l.slot_map_cpu] = { il, true };
                    buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map_cpu), alignment_host);
                    l.cpu_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.cpu_map, "moe_cache.cpu_map-%d", il);
                    buf_host_size += GGML_PAD(ggml_nbytes(l.cpu_map), alignment_host);
                }
            }
            // AMB_ELASTIC_KV level 3: only the cache's own chunks count, lent ones belong to the KV cache
            devices[g.id].buf_size += el ? el_chunks(g, g.n_base)*el_gran : alloc_size(g, g.n_slots);
        }

        // with no_alloc, dummy buffers are attached so the memory use can still be measured, see llama_context::memory_breakdown
        for (device & d : devices) {
            if (!d.ctx) {
                continue;
            }
            if (el && !no_alloc) {
                el_alloc(d);
                continue;
            }
            d.buf = llama_memory_bufs::alloc_buf(d.ctx.get(), d.buft, no_alloc, "MoE cache");
            if (!no_alloc) {
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
        }
        el_live = el && !no_alloc;
        buf_host = llama_memory_bufs::alloc_buf(ctx_host.get(), buft_host, no_alloc, "MoE cache slot maps");
        if (!no_alloc) {
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // the CPU maps are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        int32_t * slots = (int32_t *) layers[il].slot_map->data;
                        std::fill(slots, slots + layers[il].slot_map->ne[1], g.lru.empty);
                        // AMB_MOE_LOOP_DEDUP invariant D1: every visit of a physical block reads and writes the one host
                        // slot map of its representative (g.layers is in layer order, so rep[il] < il is set already);
                        // the visit's own tensor data stays unused (copy() uploads lru.slot_map[il])
                        g.lru.slot_map[il] = rep[il] != il ? g.lru.slot_map[rep[il]] : slots;
                        GGML_ASSERT(g.lru.slot_map[il] != nullptr);
                    }
                }
            }
        }
        if (loop_dedup) {
            int32_t n_eff = 0, n_phys = 0;
            size_t host = 0;
            for (const group & g : groups) {
                if (g.n_slots > 0) {
                    n_eff  += (int32_t) g.layers.size();
                    n_phys += g.n_phys;
                }
            }
            for (const device & d : devices) {
                host += d.host_bytes;
            }
            LLAMA_LOG_INFO("amb_loop_dedup: on, %d effective layers -> %d physical, host experts %.2f MiB%s\n",
                n_eff, n_phys, host/1024.0/1024.0, loop_dedup_check ? " (AMB_MOE_LOOP_DEDUP_CHECK=1)" : "");
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts, %.0f%% of the misses on the CPU\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0, 100.0*cpu_share);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%, %.1f tokens worth)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.n_phys*g.ref[0]->ne[2]),
                        (double) g.n_slots/(g.layers.size()*n_expert_used));
                    if (el) {
                        LLAMA_LOG_INFO("%s: amb_elastic: elastic banks: %d own slots, %d active, capacity %d with lent KV chunks\n", __func__,
                            g.n_base, g.lru.n_active, g.n_slots);
                    }
                    if (el && el_floor_bytes > 0) {
                        LLAMA_LOG_WARN("%s: amb_elastic: AMB_ELASTIC_FLOOR_MIB=%zu: %d own + %d floor slots = %d slots the graphs count on "
                            "(the floor comes from the compute buffer lent before every small graph)\n", __func__,
                            el_floor_bytes/1024/1024, g.n_base, g.n_graph - g.n_base, g.n_graph);
                    }
                }
            }
        }

        if (const char * v = getenv("AMB_MOE_STATS"); v != nullptr && atoll(v) > 0) {
            stats_every = atoll(v);
            for (size_t il = 0; il < layers.size() && stats_layer < 0; ++il) {
                if (layers[il].ig >= 0) {
                    stats_layer = (int32_t) il;
                }
            }
            LLAMA_LOG_INFO("%s: AMB_MOE_STATS: logging the MoE cache statistics every %lld rounds of layer %d\n", __func__,
                (long long) stats_every, stats_layer);
            if (const char * t = getenv("AMB_MOE_TIMING"); t != nullptr) {
                timing_mode = std::clamp(atoi(t), 0, 2);
                if (timing_mode > 0) {
                    LLAMA_LOG_INFO("%s: AMB_MOE_TIMING=%d: timing the MoE cache copies and the graph evaluations%s\n", __func__, timing_mode,
                        timing_mode == 2 ? " (mode 2 synchronizes after the miss uploads: perturbs the execution)" : "");
                }
            }
        }

        if (const char * v = getenv("AMB_MOE_BATCH_COPY"); v != nullptr && strcmp(v, "1") == 0) {
            batch_copy = true;
        }

        if (const char * v = getenv("AMB_MOE_PREDICT"); v != nullptr && strcmp(v, "1") == 0) {
            predict  = true;
            pred_k[0] = std::min<int32_t>(n_expert_used, predict_n_ids);
            pred_k[1] = std::min<int32_t>(16, predict_n_ids);
            pred_k[2] = predict_n_ids;
            pred_recs.resize(layers.size());
            pred_cum.assign((size_t) n_pred*layers.size(), pred_acc{});
            pred_prev = pred_cum;
            LLAMA_LOG_INFO("%s: AMB_MOE_PREDICT=1: measuring early router predictions of single-token ubatches, "
                "sets of %d/%d/%d experts (measurement only: extra graph nodes and a sync after each evaluation)\n", __func__,
                pred_k[0], pred_k[1], pred_k[2]);
        }

        if (const char * v = getenv("AMB_MOE_PREFETCH"); v != nullptr && (atoi(v) == 1 || atoi(v) == 4) && !no_alloc) {
            prefetch_init(atoi(v));
        }

        // AMB_MOE_LOOKAHEAD (C18): the configuration decides whether the graphs get predictor nodes (also with no_alloc,
        // so the --fit pass and every reserve see the same graphs); the runtime part needs the async ids
        lookahead_config(no_alloc);

        if (const char * v = getenv("AMB_IDS_ASYNC"); v != nullptr && strcmp(v, "1") == 0 && !no_alloc) {
            ids_async_init();
        }

        if (la_build && !no_alloc) {
            lookahead_init();
        }

        // AMB_MOE_MISS_OVERLAP (C23): configuration after the lookahead's (it needs its upload stream), runtime after it
        mo_config(no_alloc);
        if (mo_build && !no_alloc) {
            mo_init();
        }
        // C25 (default off): after the miss overlap's configuration and runtime state
        c25_config();
        if (!no_alloc) {
            c25_init();
        }
        // AMB_MOE_HIT_FIRST (D2): after everything it excludes
        hf_config(no_alloc);
    }

    // AMB_MOE_HIT_FIRST: configuration (also with no_alloc: the graphs of the --fit pass and of every reserve are the
    // same) and, with buffers, the pinned staging
    void hf_config(bool no_alloc) {
        const char * v = getenv("AMB_MOE_HIT_FIRST");
        if (v == nullptr || *v == '\0' || strcmp(v, "0") == 0) {
            return;
        }
        auto env_on = [](const char * name) {
            const char * s = getenv(name);
            return s != nullptr && strcmp(s, "1") == 0;
        };
        const char * why = nullptr;
        static char why_buf[200];
        int n_dev_host = 0;
        for (const device & d : devices) {
            n_dev_host += d.host_bytes > 0 ? 1 : 0;
        }
        if (strcmp(v, "1") != 0) {
            why = "the only value is 1";
        } else if (devices.size() != 1 || n_dev_host != 1) {
            why = "the cache must be on exactly one device";
        } else if (cpu_share > 0.0f || n_zero > 0) {
            why = "--moe-cache-cpu-share > 0 (a miss left to the CPU is in neither phase)";
        } else if (pf_mode != 0 || getenv("AMB_MOE_PREFETCH") != nullptr) {
            why = "AMB_MOE_PREFETCH is set";
        } else if (la_req || la_build) {
            why = "AMB_MOE_LOOKAHEAD is set (slots with an upload in flight)";
        } else if (mo_req || mo_build) {
            why = "AMB_MOE_MISS_OVERLAP is set";
        } else if (predict) {
            why = "AMB_MOE_PREDICT=1";
        } else if (timing_mode == 2) {
            why = "AMB_MOE_TIMING=2";
        } else if (std::any_of(layers.begin(), layers.end(), [](const layer & l) { return l.ig >= 0 && (l.slot_map_h == nullptr || l.slot_map_r == nullptr); })) {
            why = "its slot maps were not created (internal)";
        }
        if (why == nullptr) {
            // the masked ids exist only in the multi-token MMVQ MoE kernel of quantized experts
            typedef int (*mmvq_max_fn_t)(ggml_backend_t, int);
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(devices[0].backend));
            const mmvq_max_fn_t mmvq_max = reg ? (mmvq_max_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_mmvq_mmid_max_batch") : nullptr;
            hf_max_tok = 8;
            if (mmvq_max == nullptr) {
                why = "the backend has no ggml_backend_cuda_mmvq_mmid_max_batch";
            } else {
                for (const group & g : groups) {
                    for (const ggml_tensor * t : g.ref) {
                        const int lim = g.n_slots > 0 ? mmvq_max(devices[0].backend, (int) t->type) : 8;
                        if (lim <= 0 && why == nullptr) {
                            snprintf(why_buf, sizeof(why_buf), "the cached expert type %s has no MMVQ MoE kernel (%s)",
                                ggml_type_name(t->type), lim == 0 ? "not a CUDA device" : "not quantized");
                            why = why_buf;
                        } else if (lim > 0 && lim < hf_max_tok) {
                            hf_max_tok = lim;
                        }
                    }
                }
            }
            if (why == nullptr && hf_max_tok < hf_min_tok) {
                why = "the MMVQ batch limit of an expert type is below 2 tokens";
            }
        }
        if (why != nullptr) {
            LLAMA_LOG_WARN("%s: amb_moe_hit_first: AMB_MOE_HIT_FIRST=%s refused: %s; one lookup per layer as before\n", __func__, v, why);
            return;
        }
        hf_build   = true;
        hf_check   = env_on("AMB_MOE_HIT_FIRST_CHECK");
        hf_sync    = env_on("AMB_MOE_HIT_FIRST_SYNC");
        hf_skip_r  = env_on("AMB_MOE_HIT_FIRST_DEBUG_SKIP_R");
        hf_no_wait = env_on("AMB_MOE_HIT_FIRST_DEBUG_NO_WAIT");
        hf_host.assign(2*layers.size(), {});
        if (no_alloc) {
            return;
        }
        const char * why_rt = nullptr;
        if (!ia.on) {
            why_rt = "AMB_IDS_ASYNC=1 is not active";
        } else {
            ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(devices[0].backend));
            size_t total = 0;
            for (const layer & l : layers) {
                total += l.slot_map_h != nullptr ? 2*GGML_PAD((size_t) l.slot_map_h->ne[1], 64) : 0;
            }
            if (host_buft != nullptr && total > 0) {
                hf_pinned.reset(ggml_backend_buft_alloc_buffer(host_buft, total*sizeof(int32_t)));
            }
            if (!hf_pinned || !ggml_backend_buffer_is_host(hf_pinned.get())) {
                hf_pinned.reset();
                why_rt = "no pinned staging buffer";
            } else {
                int32_t * p = (int32_t *) ggml_backend_buffer_get_base(hf_pinned.get());
                hf_stage_h.assign(layers.size(), nullptr);
                hf_stage_r.assign(layers.size(), nullptr);
                for (size_t il = 0; il < layers.size(); ++il) {
                    if (layers[il].slot_map_h != nullptr) {
                        const size_t n = GGML_PAD((size_t) layers[il].slot_map_h->ne[1], 64);
                        hf_stage_h[il] = p;
                        hf_stage_r[il] = p + n;
                        p += 2*n;
                    }
                }
                hf_on = true;
            }
        }
        LLAMA_LOG_INFO("%s: amb_moe_hit_first: AMB_MOE_HIT_FIRST=1: ubatches of %d..%d tokens run each cached MoE layer in two "
            "splits (experts cached before the plan, then the experts the plan filled); %s%s%s%s%s\n", __func__,
            hf_min_tok, hf_max_tok,
            hf_on && !hf_sync ? "the hit phase is launched before the ids are read (no synchronize before its slot map copy)" :
            hf_sync ? "AMB_MOE_HIT_FIRST_SYNC=1: both copies behind the scheduler's synchronize (structure cost only)" :
            "INACTIVE at run time, both copies synchronized: ",
            hf_on || hf_sync ? "" : why_rt,
            hf_check ? "; CHECK: ids and device slot map compared after a synchronize" : "",
            hf_skip_r ? "; DEBUG_SKIP_R: NEGATIVE CONTROL, the miss phase of layers with fills is masked out, outputs are WRONG" : "",
            hf_no_wait ? "; DEBUG_NO_WAIT: NEGATIVE CONTROL, the ids event is not waited for, expect an abort" : "");
    }

    // the graph of a ubatch of n_tokens tokens looks up the two hit-first maps of layer il (configuration, layout, n_tokens)
    bool hit_first(int32_t il, int64_t n_tokens) const {
        return hf_build && n_tokens >= hf_min_tok && n_tokens <= hf_max_tok &&
            get_slot_map(il, n_tokens, n_expert_used, false) != nullptr &&
            layers[il].slot_map_h != nullptr && layers[il].slot_map_r != nullptr;
    }

    int32_t * hf_stage(int32_t il, int which) {
        if (hf_on) {
            return which == 0 ? hf_stage_h[il] : hf_stage_r[il];
        }
        std::vector<int32_t> & v = hf_host[2*(size_t) il + which];
        v.resize((size_t) layers[il].slot_map->ne[1]);
        return v.data();
    }

    // copy() of slot_map_h: the slots as they are now, before the plan of this ubatch. No ids, no plan, no cache change
    bool hf_copy_h(ggml_backend_t backend, int32_t il, ggml_tensor * dst) {
        const layer & l = layers[il];
        const group & g = groups[l.ig];
        const int32_t * slots = g.lru.slot_map[il];
        const int64_t   n_e   = l.slot_map->ne[1];
        int32_t * st = hf_stage(il, 0);
        GGML_ASSERT(st != nullptr && ggml_nbytes(dst) == (size_t) n_e*sizeof(int32_t));
        for (int64_t e = 0; e < n_e; ++e) {
            st[e] = slots[e] < g.lru.n_slots ? slots[e] : -1;
        }
        ggml_backend_tensor_set_async(backend, dst, st, 0, ggml_nbytes(dst));
        hf_h_il   = il;
        hf_h_last = il;
        hf_h_dst  = dst;
        hf_st.h_copies++;
        return true;
    }

    void hf_log(const char * what) const {
        if (!hf_build) {
            return;
        }
        LLAMA_LOG_INFO("llama_moe_cache: amb_moe_hit_first: %s: round %lld: hit-phase copies = %llu, miss-phase copies = %llu "
            "(all-hit = %llu, with fills = %llu), ids by event = %llu, ids by synchronize = %llu, check: ids compared = %llu "
            "(different = %llu), device maps compared = %llu (different = %llu), debug: miss phases skipped = %llu, waits "
            "skipped = %llu\n", what, (long long) stats_rounds,
            (unsigned long long) hf_st.h_copies, (unsigned long long) hf_st.r_copies, (unsigned long long) hf_st.r_allhit,
            (unsigned long long) hf_st.r_miss, (unsigned long long) hf_st.r_async, (unsigned long long) hf_st.r_sync,
            (unsigned long long) hf_st.chk_ids, (unsigned long long) hf_st.chk_ids_bad, (unsigned long long) hf_st.chk_map,
            (unsigned long long) hf_st.chk_map_bad, (unsigned long long) hf_st.skipped_r, (unsigned long long) hf_st.no_wait);
    }

    // AMB_MOE_MISS_OVERLAP: environment and configuration-only checks (no allocation); combinations not reasoned through
    // are refused with one line and the switch stays off (graphs and every path as without it)
    void mo_config(bool no_alloc) {
        const char * v = getenv("AMB_MOE_MISS_OVERLAP");
        if (v == nullptr || (strcmp(v, "1") != 0 && strcmp(v, "2") != 0)) {
            return;
        }
        mo_req  = true;
        mo_mode = atoi(v); // 1: C23 (tail overlap); 2: C24, plus the hit/miss split inside the layer
        auto env_i = [](const char * name, int def) {
            const char * s = getenv(name);
            return s != nullptr && *s != '\0' ? atoi(s) : def;
        };
        auto env_on = [](const char * name) {
            const char * s = getenv(name);
            return s != nullptr && strcmp(s, "1") == 0;
        };
        mo_min_tok   = std::clamp(env_i("AMB_MOE_MISS_OVERLAP_MIN_TOK", 2), 1, 8);
        if (mo_mode == 2 && mo_min_tok < 2) {
            mo_min_tok = 2; // the masked ids exist only in the multi-token MMVQ MoE kernel
        }
        mo_max_tok   = std::min<int32_t>(8, la_drain_above);
        mo_cmp       = env_on("AMB_MOE_MISS_OVERLAP_DEBUG");
        mo_spin_ns   = (uint64_t) std::clamp(env_i("AMB_MOE_MISS_OVERLAP_DEBUG_SPIN_US", 0), 0, 100000)*1000ull;
        mo_skip_wait = env_on("AMB_MOE_MISS_OVERLAP_DEBUG_SKIP_WAIT");
        const char * why = nullptr;
        if (!la_build) {
            why = "AMB_MOE_LOOKAHEAD must be on and accepted (the miss uploads use its upload stream; see its log line)";
        } else if (la_dry) {
            why = "AMB_MOE_LOOKAHEAD_DRY=1 (no upload stream)";
        } else if (!batch_copy) {
            why = "AMB_MOE_BATCH_COPY=1 is required";
        } else if (la_defer) {
            why = "AMB_MOE_LOOKAHEAD_DEBUG_DEFER=1 (its two-copy() rule assumes the scheduler's synchronize before copy())";
        } else if (la_probe) {
            why = "AMB_MOE_LOOKAHEAD_ORDER_PROBE=1 (it records a compute-stream idle point before the copies)";
        } else if (timing_mode == 2) {
            why = "AMB_MOE_TIMING=2 (it synchronizes the compute stream after the miss uploads)";
        } else if (predict) {
            why = "AMB_MOE_PREDICT=1 (not reasoned through)";
        } else if (pf_mode != 0) {
            why = "AMB_MOE_PREFETCH is on";
        } else if (cpu_share > 0.0f || n_zero > 0) {
            why = "--moe-cache-cpu-share > 0";
        } else if (const char * go = getenv("GGML_CUDA_GRAPH_OPT"); go != nullptr && atoi(go) == 1) {
            why = "GGML_CUDA_GRAPH_OPT=1 (concurrent streams inside a split: the publish would not follow every earlier node, M1)";
        } else if (env_on("AMB_INPUT_ASYNC")) {
            why = "AMB_INPUT_ASYNC=1 (C21; the combination is not reasoned through)";
        } else if (env_on("AMB_IDS_ONE_READBACK")) {
            why = "AMB_IDS_ONE_READBACK=1 (C21; the readbacks it merges are replaced by the publish)";
        } else if (mo_min_tok > mo_max_tok) {
            why = "AMB_MOE_MISS_OVERLAP_MIN_TOK is above the MMVQ batch limit of the cached expert types";
        } else if ((int64_t) n_expert_used*mo_max_tok > GGML_AMB_PUBLISH_MAX || (int64_t) la_k*la_max_tok > GGML_AMB_PUBLISH_MAX) {
            why = "more ids per layer than one publish slot holds";
        } else if (mo_mode == 2 && std::any_of(layers.begin(), layers.end(), [](const layer & l) { return l.ig >= 0 && l.slot_map2 == nullptr; })) {
            why = "mode 2 without its split-phase slot maps (internal)";
        } else if (mo_skip_wait && mo_spin_ns == 0) {
            why = "AMB_MOE_MISS_OVERLAP_DEBUG_SKIP_WAIT=1 needs AMB_MOE_MISS_OVERLAP_DEBUG_SPIN_US > 0 (else it proves nothing)";
        }
        if (why != nullptr) {
            LLAMA_LOG_WARN("%s: amb_miss_overlap: AMB_MOE_MISS_OVERLAP=%d refused: %s; the miss uploads stay on the compute "
                "stream (no publish nodes)\n", __func__, mo_mode, why);
            mo_mode = 0;
            return;
        }
        mo_build = true;
        // M4: U is in order behind the miss uploads; the compute-stream event would make U wait for the tail of C
        la_link_wait = false;
        GGML_UNUSED(no_alloc);
    }

    // C25: environment and configuration-only checks, after mo_config (nothing here changes the graphs); the runtime part
    // is c25_init after mo_init. Refused unless AMB_MOE_MISS_OVERLAP=1 was accepted (so also in the no-draft profile,
    // where the lookahead and with it the miss overlap are off)
    void c25_config() {
        auto env_i = [](const char * name, int def) {
            const char * s = getenv(name);
            return s != nullptr && *s != '\0' ? atoi(s) : def;
        };
        auto env_on = [](const char * name) {
            const char * s = getenv(name);
            return s != nullptr && strcmp(s, "1") == 0;
        };
        if (const char * s = getenv("AMB_MOE_LA_CHUNK_MIB"); s != nullptr && *s != '\0') {
            ck_mib = atof(s);
            if (!(ck_mib > 0.0)) {
                ck_mib = 0.0; // "0" or junk: off
            }
        }
        ms_req = env_on("AMB_MOE_MISS_STREAM");
        const bool st_req = env_on("AMB_MOE_C25_STATS");
        if (ck_mib <= 0.0 && !ms_req && !st_req) {
            return;
        }
        if (ck_mib > 0.0) {
            ck_mib      = std::clamp(ck_mib, 0.25, 64.0);
            ck_bytes    = (size_t) (ck_mib*1024.0*1024.0);
            ck_deadline = env_i("AMB_MOE_LA_CHUNK_DEADLINE", 1) != 0;
            if (const char * s = getenv("AMB_MOE_LA_CHUNK_GBPS"); s != nullptr && *s != '\0') {
                ck_gbps = std::clamp(atof(s), 1.0, 100.0);
            }
            ck_slack_ns = (int64_t) std::clamp(env_i("AMB_MOE_LA_CHUNK_SLACK_US", 30), 0, 10000)*1000;
        }
        const char * why = nullptr;
        if (!mo_build) {
            why = "AMB_MOE_MISS_OVERLAP=1 is not on (see its log line)";
        } else if (mo_mode != 1) {
            why = "only with AMB_MOE_MISS_OVERLAP=1 (mode 2 is not reasoned through)";
        }
        if (why != nullptr) {
            LLAMA_LOG_WARN("%s: amb_c25: AMB_MOE_LA_CHUNK_MIB / AMB_MOE_MISS_STREAM / AMB_MOE_C25_STATS refused: %s; "
                "lookahead and miss uploads as without them\n", __func__, why);
            ck_mib   = 0.0;
            ck_bytes = 0;
            ms_req   = false;
            return;
        }
        c25_stats = st_req;
    }

    // C25: runtime part, after a successful mo_init (U, mo_ev exist; no batch has been issued yet)
    void c25_init() {
        if (!mo_on || mo_mode != 1) {
            return;
        }
        device * dp = nullptr;
        for (device & d : devices) {
            if (d.host_bytes > 0) {
                dp = &d;
            }
        }
        if (dp == nullptr || dp->upload == nullptr) {
            return;
        }
        ggml_backend_dev_t dev = ggml_backend_get_device(dp->backend);
        if (ck_mib > 0.0) {
            // K3: the 16-event ring, created before the first batch (la_last == 0: no seq was mapped to an event yet)
            const char * why = nullptr;
            if (dp->la_last != 0) {
                why = "lookahead batches were issued already (internal)";
            } else {
                for (int i = dp->la_n_ev; i < device::la_n_ev_max && why == nullptr; ++i) {
                    dp->la_ev[i] = ggml_backend_event_new(dev);
                    if (dp->la_ev[i] == nullptr) {
                        why = "no events for the larger ring";
                    }
                }
            }
            if (why == nullptr) {
                dp->la_n_ev = device::la_n_ev_max;
                ck_d_ns.assign(layers.size(), 0);
                ck_on = true;
                LLAMA_LOG_INFO("%s: amb_c25: AMB_MOE_LA_CHUNK_MIB=%.2f: the lookahead of a layer is admitted and uploaded in "
                    "chunks of <= %.2f MiB in rank order, the first at the issue point, the next ones while the host polls for "
                    "the layer's publish, one at a time, %s; candidates left at the publish are not sent (the selected ones "
                    "become misses); event ring %d\n", __func__, ck_mib, ck_bytes/1024.0/1024.0,
                    ck_deadline ? "only if predicted to finish before the publish (deadline on)" : "until the publish (deadline off)",
                    dp->la_n_ev);
                if (ck_deadline) {
                    LLAMA_LOG_INFO("%s: amb_c25: deadline: per-layer EWMA (1/8) of the host time from the first chunk to the "
                        "publish, chunk time at %.1f GB/s, slack %lld us\n", __func__, ck_gbps, (long long) (ck_slack_ns/1000));
                }
            } else {
                LLAMA_LOG_WARN("%s: amb_c25: AMB_MOE_LA_CHUNK_MIB refused: %s; one lookahead batch per layer as before\n",
                    __func__, why);
                for (int i = 4; i < device::la_n_ev_max; ++i) {
                    if (dp->la_ev[i] != nullptr && i >= dp->la_n_ev) {
                        ggml_backend_event_free(dp->la_ev[i]);
                        dp->la_ev[i] = nullptr;
                    }
                }
            }
        }
        if (ms_req) {
            dp->miss = ggml_backend_dev_init(dev, nullptr);
            if (dp->miss == nullptr) {
                LLAMA_LOG_WARN("%s: amb_c25: AMB_MOE_MISS_STREAM=1 refused: no third backend instance for the device; the "
                    "miss uploads stay on the upload stream\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: amb_c25: AMB_MOE_MISS_STREAM=1: the miss uploads and their event go on a third stream "
                    "(not behind the lookahead copies in stream order; whether the driver interleaves the H2D copies of the two "
                    "streams is a measurement question); the compute stream waits on the miss event and, for selected "
                    "in-flight lookahead slots, on their batch event\n", __func__);
            }
        }
        if (c25_stats) {
            LLAMA_LOG_INFO("%s: amb_c25: AMB_MOE_C25_STATS=1 (diagnostic): one event query per miss batch\n", __func__);
        }
    }

    // AMB_MOE_MISS_OVERLAP: mapped slots, device counters, the event; stays off (graphs unchanged) on failure
    void mo_init() {
        const char * why = nullptr;
        device * dp = nullptr;
        for (device & d : devices) {
            if (d.host_bytes > 0) {
                dp = &d;
            }
        }
        typedef bool (*mo_alloc_fn_t)(ggml_backend_t, size_t, void **, void **, void **);
        mo_alloc_fn_t alloc_fn = nullptr;
        if (!la_on || dp == nullptr || dp->upload == nullptr) {
            why = "the lookahead upload stream is not present (see the amb_moe_lookahead lines)";
        } else {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(dp->backend));
            alloc_fn   = reg ? (mo_alloc_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_amb_publish_alloc") : nullptr;
            mo_free_fn = reg ? (mo_free_fn_t)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_amb_publish_free") : nullptr;
            if (alloc_fn == nullptr || mo_free_fn == nullptr) {
                why = "the backend has no ggml_backend_cuda_amb_publish_alloc/free";
            } else if (mo_spin_ns > 0 && (la_spin == nullptr || la_memset == nullptr)) {
                why = "AMB_MOE_MISS_OVERLAP_DEBUG_SPIN_US needs ggml_backend_cuda_spin_async and ggml_backend_cuda_memset2d_async";
            } else if (batch_fn == nullptr) {
                why = "no batched copy function (AMB_MOE_BATCH_COPY)";
            }
        }
        // slots [0, 2 n_layer): publishes; [2 n_layer, 3 n_layer): C24 gate-wait stamps of mode 2 (one per layer)
        const size_t n_slots = 2*layers.size();
        const size_t n_alloc = 3*layers.size();
        if (why == nullptr && !alloc_fn(dp->backend, n_alloc, &mo_host, &mo_sdev, &mo_cdev)) {
            why = "the mapped pinned slots or the device counters could not be allocated";
        }
        if (why == nullptr) {
            dp->mo_ev = ggml_backend_event_new(ggml_backend_get_device(dp->backend));
            if (dp->mo_ev == nullptr) {
                why = "no event for the device";
            }
        }
        // AMB_MOE_MISS_OVERLAP=2 (C24): one gate event per cached layer, and the pinned staging of the split-phase maps
        if (why == nullptr && mo_mode == 2) {
            ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(dp->backend));
            size_t n_stage = 0;
            for (const layer & l : layers) {
                n_stage += l.slot_map2 != nullptr ? GGML_PAD(2*(size_t) l.slot_map2->ne[1], 64) : 0;
            }
            mo2_pinned.reset(host_buft ? ggml_backend_buft_alloc_buffer(host_buft, std::max<size_t>(n_stage, 64)*sizeof(int32_t)) : nullptr);
            if (!mo2_pinned || !ggml_backend_buffer_is_host(mo2_pinned.get())) {
                why = "no pinned buffer for the split-phase slot maps";
            } else {
                int32_t * p = (int32_t *) ggml_backend_buffer_get_base(mo2_pinned.get());
                mo2_stage.assign(layers.size(), nullptr);
                mo2_ev.assign(layers.size(), nullptr);
                for (size_t il = 0; il < layers.size() && why == nullptr; ++il) {
                    if (layers[il].slot_map2 == nullptr) {
                        continue;
                    }
                    mo2_stage[il] = p;
                    p += GGML_PAD(2*(size_t) layers[il].slot_map2->ne[1], 64);
                    mo2_ev[il] = ggml_backend_event_new(ggml_backend_get_device(dp->backend));
                    if (mo2_ev[il] == nullptr) {
                        why = "no gate event for a layer";
                    }
                }
            }
            if (const char * s = getenv("AMB_STEP_EVENTS"); s != nullptr && atoll(s) > 0) {
                mo2_stamps = true; // the gate ops of the graphs also time their wait (two tiny kernels per layer)
            }
        }
        if (why != nullptr) {
            LLAMA_LOG_WARN("%s: amb_miss_overlap: AMB_MOE_MISS_OVERLAP=%d refused: %s; the miss uploads stay on the compute "
                "stream (no publish nodes)\n", __func__, mo_mode, why);
            mo_free();
            mo_build = false;
            mo_mode  = 0;
            return;
        }
        mo_be    = dp->backend;
        mo_flush = dp->mo_ev;
        mo_expect.assign(n_slots, 0);
        mo_taken.assign(n_slots, 0);
        mo_n.assign(n_slots, 0);
        mo_ntok.assign(n_slots, 0);
        mo_on = true;
        LLAMA_LOG_INFO("%s: amb_miss_overlap: AMB_MOE_MISS_OVERLAP=%d: ubatches of %d..%d tokens: the selected ids of a layer "
            "are published to mapped pinned memory right after its router, the predictor of the next layer and the shared "
            "expert run after that, the host polls the publish (no synchronize before the copy callback), the miss uploads "
            "go on the upload stream and the compute stream waits on their event; %zu slots, %.1f KiB mapped; lookahead link "
            "wait off\n", __func__, mo_mode, mo_min_tok, mo_max_tok, n_slots, n_alloc*GGML_AMB_PUBLISH_SLOT/1024.0);
        if (mo_mode == 2) {
            LLAMA_LOG_INFO("%s: amb_miss_overlap: mode 2 (C24): the routed experts of a layer run in two phases: the resident "
                "experts first, then a gate (GPU-side wait on the layer's miss-upload event), then the missed and still-uploading "
                "ones; outputs merged by an ADD with -0.0f, the weighted sum unchanged; gate-wait stamps %s\n", __func__,
                mo2_stamps ? "on (AMB_STEP_EVENTS)" : "off");
        }
        if (mo_cmp) {
            LLAMA_LOG_WARN("%s: amb_miss_overlap: DEBUG=1: every published id list is also read back the old way (after the "
                "split, with a synchronize) and compared; mismatches are counted and the old ids are used\n", __func__);
        }
        if (mo_spin_ns > 0) {
            LLAMA_LOG_WARN("%s: amb_miss_overlap: DEBUG_SPIN_US=%llu: the fill slots are set to 0xFF and the upload stream is "
                "delayed before every miss upload batch\n", __func__, (unsigned long long) (mo_spin_ns/1000));
        }
        if (mo_skip_wait) {
            LLAMA_LOG_WARN("%s: amb_miss_overlap: ********************************************************************\n", __func__);
            LLAMA_LOG_WARN("%s: amb_miss_overlap: DEBUG_SKIP_WAIT=1: NEGATIVE CONTROL, the compute stream does NOT wait for\n", __func__);
            LLAMA_LOG_WARN("%s: amb_miss_overlap: the miss uploads: OUTPUTS ARE EXPECTED TO BE WRONG. Test use only.\n", __func__);
            LLAMA_LOG_WARN("%s: amb_miss_overlap: ********************************************************************\n", __func__);
        }
    }

    // AMB_MOE_MISS_OVERLAP: U must be idle (lookahead_free / the destructor synchronize it first)
    void mo_free() {
        for (device & d : devices) {
            // C25 AMB_MOE_MISS_STREAM: M finished first (no upload may write a freed bank; mo_ev may be recorded on it)
            if (d.miss != nullptr) {
                ggml_backend_synchronize(d.miss);
                ggml_backend_free(d.miss);
                d.miss = nullptr;
                d.mo_dirty = false;
            }
            if (d.mo_ev != nullptr) {
                ggml_backend_event_free(d.mo_ev);
                d.mo_ev = nullptr;
            }
        }
        ck_abandon(); // C25: chunking needs the poll of the miss overlap
        ck_on = false;
        if (mo_free_fn != nullptr && (mo_host != nullptr || mo_cdev != nullptr)) {
            mo_free_fn(mo_be != nullptr ? mo_be : devices[0].backend, mo_host, mo_cdev);
        }
        mo_host = mo_sdev = mo_cdev = nullptr;
        mo_flush = nullptr;
        mo_on = false;
        // C24 mode 2
        for (ggml_backend_event_t & ev : mo2_ev) {
            if (ev != nullptr) {
                ggml_backend_event_free(ev);
                ev = nullptr;
            }
        }
        mo2_ev.clear();
        mo2_stage.clear();
        mo2_pinned.reset();
    }

    // C24 fix a: before any graph is built
    void miss_overlap_refuse(const char * reason) {
        if (!mo_build && !mo_on) {
            return;
        }
        LLAMA_LOG_WARN("%s: amb_miss_overlap: AMB_MOE_MISS_OVERLAP=%d refused: %s; the miss uploads stay on the compute "
            "stream (no publish nodes)\n", __func__, mo_mode, reason);
        mo_on    = false;
        mo_build = false;
        mo_mode  = 0;
        ck_abandon(); // C25: chunking needs the poll of the miss overlap (M stays allocated, unused: no miss goes there)
        ck_on = false;
    }

    // C24 fix b: the publish bookkeeping no longer matches the device (e.g. a split failed after some of its nodes ran):
    // the rest of the context takes the synchronized path. The graphs keep their nodes (publishes still execute, nothing
    // reads them; split-phase graphs get an all-hit slot map and their misses on the compute stream, see copy())
    void mo_runtime_disable(const char * reason) {
        if (!mo_on) {
            return;
        }
        LLAMA_LOG_WARN("llama_moe_cache: amb_miss_overlap: disabled for the rest of the context: %s; the copies are "
            "synchronized as without the switch from now on\n", reason);
        mo_on = false;
        mo_st.disabled++;
        // C25: no more chunks (the poll that feeds them is gone); the chunks already issued are ordinary batches (I4/I5).
        // M keeps the misses issued so far; mo_dirty still makes lookahead_drain synchronize it (S3)
        ck_abandon();
        ck_on = false;
        // C24 review S3/S4: graphs built from now on get no publish / gate nodes and no split; graphs built before (and
        // reused while their topology repeats) keep them: their publishes still run (nothing reads them), their gates wait
        // on old, complete records, their miss phase computes nothing (all-hit map, see copy())
        mo_build = false;
    }

    // AMB_MOE_LOOKAHEAD: environment and configuration-only checks (no allocation)
    void lookahead_config(bool no_alloc) {
        const char * v = getenv("AMB_MOE_LOOKAHEAD");
        if (v == nullptr || *v == '\0' || strcmp(v, "0") == 0) {
            return;
        }
        auto env_i = [](const char * name, int def) {
            const char * s = getenv(name);
            return s != nullptr && *s != '\0' ? atoi(s) : def;
        };
        const int mode = atoi(v);
        la_k         = std::clamp(env_i("AMB_MOE_LOOKAHEAD_K", 10), 1, 24);
        la_min_layer = std::max(1, env_i("AMB_MOE_LOOKAHEAD_MIN_LAYER", 2));
        la_max_tok   = std::clamp(env_i("AMB_MOE_LOOKAHEAD_MAX_TOK", 4), 1, 8); // never MMQ batches (> 8 tokens)
        if (const char * s = getenv("AMB_MOE_LOOKAHEAD_MAX_MIB"); s != nullptr && *s != '\0') {
            la_budget_1 = la_budget_n = (size_t) std::max(0.0, atof(s)*1024.0*1024.0);
        }
        if (const char * s = getenv("AMB_MOE_LOOKAHEAD_MAX_MIB_1"); s != nullptr && *s != '\0') {
            la_budget_1 = (size_t) std::max(0.0, atof(s)*1024.0*1024.0);
        }
        la_dry   = env_i("AMB_MOE_LOOKAHEAD_DRY", 0) == 1;
        la_debug = env_i("AMB_MOE_LOOKAHEAD_DEBUG", 0) == 1 && !la_dry;
        // C18b
        la_min_tok   = std::clamp(env_i("AMB_MOE_LOOKAHEAD_MIN_TOK", 2), 1, 8);
        la_rank      = env_i("AMB_MOE_LOOKAHEAD_RANK", 0) == 1;
        la_link_wait = env_i("AMB_MOE_LOOKAHEAD_LINK_WAIT", 1) != 0;
        la_protect   = env_i("AMB_MOE_LOOKAHEAD_PROTECT", 1) != 0;
        la_p4_fast   = env_i("AMB_MOE_LOOKAHEAD_P4_FAST", 0) == 1;
        if (const char * s = getenv("AMB_MOE_LOOKAHEAD_ISSUE"); s != nullptr && strcmp(s, "post") == 0) {
            la_post = true;
        }
        la_skip_wait     = la_debug && env_i("AMB_MOE_LOOKAHEAD_DEBUG_SKIP_WAIT", 0) == 1;
        la_spin_ns       = (uint64_t) std::clamp(env_i("AMB_MOE_LOOKAHEAD_DEBUG_SPIN_US", 200), 0, 100000)*1000ull;
        la_dbg_no_victim = la_dry ? 0 : std::max(0, env_i("AMB_MOE_LOOKAHEAD_DEBUG_NO_VICTIM", 0));
        la_defer         = la_debug && env_i("AMB_MOE_LOOKAHEAD_DEBUG_DEFER", 0) == 1;
        la_probe         = !la_dry && env_i("AMB_MOE_LOOKAHEAD_ORDER_PROBE", 0) == 1; // effective with AMB_STEP_EVENTS
        const char * why = nullptr;
        int n_dev_host = 0;
        for (const device & d : devices) {
            n_dev_host += d.host_bytes > 0 ? 1 : 0;
        }
        if (mode != 1 && mode != 4) {
            why = "the value must be 1 (P1) or 4 (P4); P5 has no issue point (no split boundary between the residual before "
                  "the attention mixer of a layer and the lookup of that layer)";
        } else if (const char * pf = getenv("AMB_MOE_PREFETCH"); pf != nullptr && (atoi(pf) == 1 || atoi(pf) == 4)) {
            why = "AMB_MOE_PREFETCH is on (the two exclude each other)";
        } else if (cpu_share > 0.0f || n_zero > 0) {
            why = "--moe-cache-cpu-share > 0";
        } else if (devices.size() != 1 || n_dev_host != 1) {
            why = "the cache must be on exactly one device";
        } else if (const char * ia_v = getenv("AMB_IDS_ASYNC"); ia_v == nullptr || strcmp(ia_v, "1") != 0) {
            why = "AMB_IDS_ASYNC=1 is required (the predicted ids are read back with the async ids)";
        } else if (la_min_tok > la_max_tok) {
            why = "AMB_MOE_LOOKAHEAD_MIN_TOK is above AMB_MOE_LOOKAHEAD_MAX_TOK";
        }
        // C18b (review m6): every ubatch that can run while an upload is in flight must take the MMVQ path for every cached
        // expert type (MMQ reads past the last row into the next slot); the drain threshold is the smallest such limit
        static char why_buf[256];
        if (why == nullptr) {
            typedef int (*mmvq_max_fn_t)(ggml_backend_t, int);
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(devices[0].backend));
            const mmvq_max_fn_t mmvq_max = reg ? (mmvq_max_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_mmvq_mmid_max_batch") : nullptr;
            la_drain_above = 8;
            if (mmvq_max == nullptr) {
                why = "the backend has no ggml_backend_cuda_mmvq_mmid_max_batch (cannot check the MMVQ batch limit)";
            } else {
                for (const group & g : groups) {
                    for (const ggml_tensor * t : g.ref) {
                        const int lim = mmvq_max(devices[0].backend, (int) t->type);
                        if (lim == 0) {
                            why = "the cache device is not a CUDA device";
                        } else if (lim > 0 && lim < la_drain_above) {
                            la_drain_above = lim;
                        }
                        if (why == nullptr && lim > 0 && lim < la_max_tok) {
                            snprintf(why_buf, sizeof(why_buf), "AMB_MOE_LOOKAHEAD_MAX_TOK = %d is above the MMVQ batch limit %d of "
                                "the expert type %s (such batches would take MMQ)", la_max_tok, lim, ggml_type_name(t->type));
                            why = why_buf;
                        }
                    }
                }
            }
        }
        int n_groups = 0;
        // AMB_MOE_LOOKAHEAD_SMALL=<m> (C21, default off, m = 1..4): a group that fails the rule below still gets lookahead
        // if its own slots cover m times (the distinct experts of one plan at la_max_tok tokens + the largest lookahead batch
        // the per-layer byte budget admits for its expert size). The rule only sizes the margin against "no victim" drains:
        // victim() never picks a selected, in-flight, recently used or batch-marked slot whatever the group size, a real
        // miss with no victim drains the upload stream and retries, and I4 orders the compute stream after in-flight slots
        const int la_small = std::clamp(env_i("AMB_MOE_LOOKAHEAD_SMALL", 0), 0, 4);
        if (why == nullptr) {
            for (group & g : groups) {
                // the selections of a plan, the in-flight batches and their exclusions, with a wide margin
                g.la_ok = g.n_slots > 0 && la_k <= g.ref[0]->ne[2] &&
                    (int64_t) g.n_base >= 4*(int64_t) (n_expert_used + la_k)*la_max_tok;
                g.la_small = false;
                if (!g.la_ok && la_small > 0 && g.n_slots > 0 && la_k <= g.ref[0]->ne[2]) {
                    size_t expert_bytes = 0;
                    for (const ggml_tensor * t : g.ref) {
                        expert_bytes += t->nb[2];
                    }
                    const int64_t n_cand  = (int64_t) la_k*la_max_tok; // distinct predicted experts of one batch, at most
                    const int64_t n_bytes = expert_bytes > 0 ? (int64_t) (std::max(la_budget_1, la_budget_n)/expert_bytes) : n_cand;
                    const int64_t n_batch = std::min(n_cand, n_bytes);
                    const int64_t need    = (int64_t) la_small*((int64_t) n_expert_used*la_max_tok + n_batch);
                    if ((int64_t) g.n_base >= need) {
                        g.la_ok    = true;
                        g.la_small = true;
                    }
                    if (!no_alloc) {
                        std::string ls;
                        for (int32_t il : g.layers) {
                            ls += (ls.empty() ? "" : ",") + std::to_string(il);
                        }
                        LLAMA_LOG_INFO("%s: amb_moe_lookahead: AMB_MOE_LOOKAHEAD_SMALL=%d: group of layers [%s] (%s, %.2f MiB per "
                            "expert): %d own slots %s %d x (%d x %d + %lld) = %lld (the standard rule needs %lld): lookahead %s\n",
                            __func__, la_small, ls.c_str(), ggml_type_name(g.ref.back()->type), expert_bytes/1024.0/1024.0,
                            g.n_base, g.la_small ? ">=" : "<", la_small, n_expert_used, la_max_tok, (long long) n_batch,
                            (long long) need, (long long) (4*(int64_t) (n_expert_used + la_k)*la_max_tok),
                            g.la_small ? "ON (relaxed)" : "still off");
                    }
                }
                n_groups += g.la_ok ? 1 : 0;
            }
            if (n_groups == 0) {
                why = "no group has enough slots";
            }
        }
        if (why != nullptr) {
            if (!no_alloc) {
                LLAMA_LOG_WARN("%s: amb_moe_lookahead: AMB_MOE_LOOKAHEAD=%s refused: %s; lookahead is off\n", __func__, v, why);
            }
            for (group & g : groups) {
                g.la_ok    = false;
                g.la_small = false;
            }
            return;
        }
        la_req   = mode;
        la_build = true;
        la_cap   = (size_t) la_k*(size_t) la_max_tok;
        for (size_t il = 0; il < layers.size() && la_first_layer < 0; ++il) {
            if (layers[il].ig >= 0) {
                la_first_layer = (int32_t) il;
            }
        }
        if (const char * s = getenv("AMB_MOE_STATS"); s != nullptr && atoll(s) > 0) {
            la_log_every = atoll(s);
        }
    }

    // AMB_MOE_LOOKAHEAD: upload backend, events, functions; lookahead stays off (and the graphs get no predictor) on failure
    void lookahead_init() {
        const char * why = nullptr;
        if (!ia.on) {
            why = "AMB_IDS_ASYNC is not active (see its log line)";
        }
        ggml_backend_reg_t reg = nullptr;
        if (why == nullptr) {
            reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(devices[0].backend));
            la_query  = reg ? (event_query_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_event_query_status") : nullptr;
            la_spin   = reg ? (spin_fn_t)        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_spin_async") : nullptr;
            la_memset = reg ? (la_memset_fn_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_memset2d_async") : nullptr;
            if (!batch_resolved) {
                batch_fn = reg ? (batch_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_memcpy_batch_async") : nullptr;
            }
            if (la_query == nullptr) {
                why = "the backend has no ggml_backend_cuda_event_query_status";
            } else if (la_debug && (la_spin == nullptr || la_memset == nullptr)) {
                why = "AMB_MOE_LOOKAHEAD_DEBUG needs ggml_backend_cuda_spin_async and ggml_backend_cuda_memset2d_async";
            }
        }
        if (why == nullptr && !la_dry) {
            for (device & d : devices) {
                if (d.host_bytes == 0) {
                    continue;
                }
                ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
                // a second backend instance of the same device, for its own stream (no buffers of its own)
                d.upload  = ggml_backend_dev_init(dev, nullptr);
                d.la_ev_c = d.upload ? ggml_backend_event_new(dev) : nullptr;
                bool ok = d.upload != nullptr && d.la_ev_c != nullptr;
                for (int i = 0; i < d.la_n_ev; ++i) { // C25: the first la_n_ev (4) only, as before
                    ggml_backend_event_t & ev = d.la_ev[i];
                    ev = ok ? ggml_backend_event_new(dev) : nullptr;
                    ok = ok && ev != nullptr;
                }
                if (!ok) {
                    why = "no second backend instance or events for the device";
                }
            }
        }
        if (why != nullptr) {
            LLAMA_LOG_WARN("%s: amb_moe_lookahead: AMB_MOE_LOOKAHEAD=%d refused: %s; lookahead is off (no predictor nodes)\n",
                __func__, la_req, why);
            lookahead_free();
            la_build = false;
            for (group & g : groups) {
                g.la_ok    = false;
                g.la_small = false;
            }
            return;
        }
        for (group & g : groups) {
            if (!g.la_ok) {
                continue;
            }
            moe_cache_lru & lru = g.lru;
            lru.pf_seq.assign(g.n_slots, 0);
            lru.pf_unused.assign(g.n_slots, 0);
            lru.key_round.assign((size_t) layers.size()*g.ref[0]->ne[2], 0);
            lru.la_mark.assign(g.n_slots, 0);
            lru.cur_round = la_round;
            lru.drain_fn  = [](void * ud, bool forced) {
                ((impl *) ud)->lookahead_drain(forced ? "forced no victim (AMB_MOE_LOOKAHEAD_DEBUG_NO_VICTIM)" :
                                                        "no victim for a real miss (in-flight slots)");
            };
            lru.drain_ud  = this;
            lru.dbg_no_victim_every = la_dbg_no_victim;
        }
        if (la_skip_wait) {
            LLAMA_LOG_WARN("%s: amb_moe_lookahead: ********************************************************************\n", __func__);
            LLAMA_LOG_WARN("%s: amb_moe_lookahead: AMB_MOE_LOOKAHEAD_DEBUG_SKIP_WAIT=1: NEGATIVE CONTROL, the compute stream does NOT\n", __func__);
            LLAMA_LOG_WARN("%s: amb_moe_lookahead: wait for in-flight lookahead uploads: OUTPUTS ARE EXPECTED TO BE WRONG. Test use only.\n", __func__);
            LLAMA_LOG_WARN("%s: amb_moe_lookahead: ********************************************************************\n", __func__);
        }
        if (la_dbg_no_victim > 0) {
            LLAMA_LOG_WARN("%s: amb_moe_lookahead: AMB_MOE_LOOKAHEAD_DEBUG_NO_VICTIM=%lld (debug): every %lld-th real-miss victim search "
                "with uploads in flight first reports no victim (drain + retry)\n", __func__, (long long) la_dbg_no_victim, (long long) la_dbg_no_victim);
        }
        la_ids_t.assign(layers.size(), nullptr);
        la_pred_round.assign(layers.size(), 0);
        la_dry_set.assign(layers.size(), {});
        if (const char * s = getenv("AMB_STEP_EVENTS"); s != nullptr && atoll(s) > 0 && !la_dry && reg != nullptr) {
            la_tev_new     = (tev_new_t)     ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_timing_event_new");
            la_tev_free    = (tev_free_t)    ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_timing_event_free");
            la_tev_record  = (tev_record_t)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_timing_event_record");
            la_tev_elapsed = (tev_elapsed_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_timing_event_elapsed");
            if (la_tev_new && la_tev_free && la_tev_record && la_tev_elapsed) {
                la_tev_every = std::max<int64_t>(1, atoll(s));
            }
        }
        la_on = true;
        LLAMA_LOG_INFO("%s: amb_moe_lookahead: AMB_MOE_LOOKAHEAD=%d%s%s: P%d, k = %d per token, layers >= %d, ubatches <= %d tokens, "
            "budget %.1f MiB (1 token) / %.1f MiB (more) per layer, %s; batched copies %s%s\n", __func__, la_req,
            la_dry ? " DRY (nothing uploaded, no cache state changes)" : "",
            la_debug ? " DEBUG (0xFF victims + upload delay, see below)" : "",
            la_req, la_k, la_min_layer, la_max_tok, la_budget_1/1024.0/1024.0, la_budget_n/1024.0/1024.0,
            la_dry ? "no upload stream" : "second stream", batch_fn ? "on" : "unavailable (one copy per run)",
            la_tev_every > 0 ? ", timing events (AMB_STEP_EVENTS)" : "");
        LLAMA_LOG_INFO("%s: amb_moe_lookahead: C18b: min tokens = %d, rank = %d, issue = %s, link wait = %d, protect cached = %d, "
            "P4 fast = %d, drain above %d tokens (smallest MMVQ batch limit of the cached expert types)%s\n", __func__,
            la_min_tok, (int) la_rank, la_post ? (la_dry ? "post (ignored in DRY: copy)" : "post (split launch hook)") : "copy",
            (int) la_link_wait, (int) la_protect, (int) la_p4_fast, la_drain_above,
            la_debug ? "" : "");
        if (la_debug) {
            LLAMA_LOG_WARN("%s: amb_moe_lookahead: DEBUG: spin %llu us before each batch, skip wait = %d\n", __func__,
                (unsigned long long) (la_spin_ns/1000), (int) la_skip_wait);
        }
        if (la_defer) {
            LLAMA_LOG_WARN("%s: amb_moe_lookahead: DEBUG_DEFER=1: the copies of each batch are enqueued by the host two copy() "
                "calls later (or before an I4 wait that needs them, or at a drain); victims poisoned at issue\n", __func__);
        }
        if (la_probe && la_tev_every <= 0) {
            LLAMA_LOG_WARN("%s: amb_moe_lookahead: ORDER_PROBE=1 needs AMB_STEP_EVENTS=N (timing events); probe off\n", __func__);
            la_probe = false;
        } else if (la_probe) {
            LLAMA_LOG_INFO("%s: amb_moe_lookahead: ORDER_PROBE=1: compute-stream events before the blocking copies and after "
                "the slot map upload of every I4 copy() in timed rounds, compared with the end of the needed batch on U\n", __func__);
        }
    }

    // AMB_MOE_LOOKAHEAD: before the banks go (destructor) or on a refused init
    void lookahead_free() {
        // DEBUG_DEFER: enqueue the pending copies (the slots stay consistent with the host slot maps)
        while (!la_defer_q.empty()) {
            la_flush_front();
            la_defer_fl_drain++;
        }
        for (device & d : devices) {
            if (d.upload != nullptr) {
                ggml_backend_synchronize(d.upload);
            }
            for (auto & ev : d.la_ev) {
                if (ev != nullptr) {
                    ggml_backend_event_free(ev);
                    ev = nullptr;
                }
            }
            if (d.la_ev_c != nullptr) {
                ggml_backend_event_free(d.la_ev_c);
                d.la_ev_c = nullptr;
            }
            if (d.upload != nullptr && pf_mode == 0) {
                ggml_backend_free(d.upload);
                d.upload = nullptr;
            }
        }
        if (la_tev_free != nullptr) {
            for (void * ev : la_tev_pool) {
                la_tev_free(ev);
            }
        }
        la_tev_pool.clear();
        la_on = false;
    }

    // AMB_IDS_ASYNC=1: preconditions, pinned memory and the event; stays off (one warning) if any is missing
    void ids_async_init() {
        const char * why = nullptr;
        int n_dev = 0;
        for (const device & d : devices) {
            n_dev += d.host_bytes > 0 ? 1 : 0;
        }
        ggml_backend_t backend = nullptr;
        for (const device & d : devices) {
            if (d.host_bytes > 0) {
                backend = d.backend;
            }
        }
        ggml_backend_buffer_type_t host_buft = backend ? ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend)) : nullptr;
        if (n_dev != 1 || backend == nullptr) {
            why = "the cache must be on exactly one device";
        } else if (pf_mode != 0) {
            why = "AMB_MOE_PREFETCH is on (it reads its predicted ids in the same readback)";
        } else if (cpu_share > 0.0f || n_zero > 0) {
            why = "--moe-cache-cpu-share > 0";
        } else if (host_buft == nullptr) {
            why = "the device has no pinned host buffer type";
        }
        size_t n_stage = 0;
        if (why == nullptr) {
            ia.cap = (size_t) n_expert_used*(size_t) llama_moe_cache::max_batch;
            for (const layer & l : layers) {
                n_stage += l.slot_map != nullptr ? (size_t) l.slot_map->ne[1] : 0;
            }
            // 64-int32 (256 B) alignment of every area
            const size_t cap_pad = GGML_PAD(ia.cap, 64);
            const size_t la_pad  = la_build ? GGML_PAD(la_cap, 64) : 0; // AMB_MOE_LOOKAHEAD: predicted ids per entry
            size_t total = ia.max_pend*(cap_pad + la_pad);
            for (const layer & l : layers) {
                total += l.slot_map != nullptr ? GGML_PAD((size_t) l.slot_map->ne[1], 64) : 0;
            }
            ia.pinned.reset(ggml_backend_buft_alloc_buffer(host_buft, total*sizeof(int32_t)));
            ia.ev = ggml_backend_event_new(ggml_backend_get_device(backend));
            if (!ia.pinned || ia.ev == nullptr || !ggml_backend_buffer_is_host(ia.pinned.get())) {
                why = "no pinned buffer or event";
            } else {
                int32_t * p = (int32_t *) ggml_backend_buffer_get_base(ia.pinned.get());
                ia.ids_base = p;
                for (int k = 0; k < ia.max_pend; ++k) {
                    ia.pend[k].dst = p + k*cap_pad;
                }
                p += ia.max_pend*cap_pad;
                for (int k = 0; k < ia.max_pend; ++k) {
                    ia.pend[k].la_dst = la_pad > 0 ? p + k*la_pad : nullptr;
                }
                p += ia.max_pend*la_pad;
                ia.stage.assign(layers.size(), nullptr);
                for (size_t il = 0; il < layers.size(); ++il) {
                    if (layers[il].slot_map != nullptr) {
                        ia.stage[il] = p;
                        p += GGML_PAD((size_t) layers[il].slot_map->ne[1], 64);
                    }
                }
            }
        }
        if (why != nullptr) {
            LLAMA_LOG_WARN("%s: amb_ids_async: AMB_IDS_ASYNC=1 refused: %s; the ids are read back as before\n", __func__, why);
            ids_async_free();
            return;
        }
        ia.on = true;
        LLAMA_LOG_INFO("%s: amb_ids_async: AMB_IDS_ASYNC=1: ids read back into pinned memory right after the producing split, "
            "copy() waits on an event; slot maps uploaded from pinned staging (%zu ids per readback, %zu slot map entries, %.1f KiB pinned)\n",
            __func__, ia.cap, n_stage, ggml_backend_buffer_get_size(ia.pinned.get())/1024.0);

        // C21 AMB_IDS_ONE_READBACK=1 (default off): only changes how the two readbacks of ids_post_split are enqueued
        if (const char * v = getenv("AMB_IDS_ONE_READBACK"); v != nullptr && strcmp(v, "1") == 0) {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
            ia.one_fn = reg ? (batch_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_memcpy_batch_async") : nullptr;
            ia.one_rb = ia.one_fn != nullptr;
            if (ia.one_rb) {
                LLAMA_LOG_INFO("%s: amb_ids_async: AMB_IDS_ONE_READBACK=1: the real ids of a layer and the predicted ids of the "
                    "next layer are read back by one batched D2H call (counted in the amb_ids_one_readback line)\n", __func__);
            } else {
                LLAMA_LOG_WARN("%s: amb_ids_async: AMB_IDS_ONE_READBACK=1 refused: the backend has no "
                    "ggml_backend_cuda_memcpy_batch_async; two readbacks as before\n", __func__);
            }
        }
    }

    void ids_async_free() {
        ia.on = false;
        if (ia.ev != nullptr) {
            ggml_backend_event_free(ia.ev);
            ia.ev = nullptr;
        }
        ia.pinned.reset();
        ia.ids_base = nullptr;
        ia.stage.clear();
        ia.n_pend = 0;
    }

    void log_ids_async(const char * what) const {
        if (!ia.on) {
            return;
        }
        LLAMA_LOG_INFO("llama_moe_cache: amb_ids_async: %s: round %lld: async hits = %llu, issued = %llu, fallbacks = %llu "
            "(no pending readback = %llu, too many ids = %llu, prefetch = %llu), issued but unused = %llu, pinned slot map uploads = %llu\n",
            what, (long long) stats_rounds, (unsigned long long) ia.hits, (unsigned long long) ia.issued,
            (unsigned long long) (ia.fb_no_pending + ia.fb_size + ia.fb_prefetch),
            (unsigned long long) ia.fb_no_pending, (unsigned long long) ia.fb_size, (unsigned long long) ia.fb_prefetch,
            (unsigned long long) ia.unused, (unsigned long long) ia.stage_uploads);
        if (ia.one_rb) {
            LLAMA_LOG_INFO("llama_moe_cache: amb_ids_one_readback: %s: round %lld: pairs read by one batched call = %llu, "
                "pairs read by two calls after a failed batched call = %llu\n", what, (long long) stats_rounds,
                (unsigned long long) ia.one_merged, (unsigned long long) ia.one_fb);
        }
    }

    // AMB_IDS_ASYNC: the scheduler launched a split; if the next split looks up a slot map, read its ids back now
    void ids_post_split(const ggml_backend_sched_split_info * info) {
        if (!ia.on) {
            return;
        }
        // AMB_MOE_HIT_FIRST: the split just launched is the hit phase of layer hf_h_il. The readback of its ids (enqueued
        // after the split before it) stays pending: copy() of the miss-phase map, in the next split, takes it
        if (hf_h_il >= 0) {
            hf_h_il = -1;
            return;
        }
        // AMB_MOE_MISS_OVERLAP (M6): the publishes of the split just launched
        if (mo_on) {
            mo_scan(info->graph);
        }
        // C18b AMB_MOE_LOOKAHEAD_ISSUE=post: the split whose copy() deferred the lookahead was just launched. Since that
        // copy() no plan ran and no slot changed state (the launch runs no cache code), and C reads only the slots of that
        // plan until the next copy(): the exclusion set of the victim search is the one copy() would have used (I1, I2)
        if (la_pend.active && info->graph == la_pend.graph && info->backend == la_pend.backend) {
            la_pend.active = false;
            // AMB_MOE_MISS_OVERLAP: the predicted ids come from their publish (in the tail before the lookup of il_real)
            if (!la_pend.mo || mo_take_pred(la_pend.il, la_pend.n_tok)) {
                lookahead_issue(la_pend.backend, la_pend.il_real, la_pend.il, la_pend.n_tok, false);
                la_post_issued++;
            }
        }
        // the pending readbacks were for the split launched now: its copy() has used them, or never will
        ids_async_reset();
        if (info->next_graph == nullptr || info->next_backend != info->backend) {
            return;
        }
        ggml_cgraph * next = info->next_graph;
        const int n_nodes = ggml_graph_n_nodes(next);
        for (int i = 0; i < info->n_next_inputs && ia.n_pend < ia.max_pend; ++i) {
            const auto it = layer_of.find(info->next_inputs[i]);
            if (it == layer_of.end()) {
                continue;
            }
            const int32_t il = it->second.first;
            const layer & l = layers[il];
            if (l.ig < 0 || devices[groups[l.ig].id].backend != info->backend) {
                continue;
            }
            const ggml_tensor * dst = info->next_input_copies[i];
            if (dst == nullptr) {
                continue;
            }
            // AMB_MOE_MISS_OVERLAP: the ids of this lookup were published (copy() takes them); DEBUG also reads them back
            if (mo_pending(il, 0) && !mo_cmp) {
                continue;
            }
            // the lookup of this slot map in the next split, as copy() finds it
            const ggml_tensor * lookup = nullptr;
            for (int j = 0; j < n_nodes && lookup == nullptr; ++j) {
                const ggml_tensor * node = ggml_graph_node(next, j);
                if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                    lookup = node;
                }
            }
            if (lookup == nullptr) {
                continue;
            }
            const ggml_tensor * sel = lookup->src[1];
            if (sel == nullptr || sel->op != GGML_OP_RESHAPE || !ggml_is_contiguous(sel) || sel->data == nullptr ||
                sel->buffer == nullptr || ggml_backend_buffer_is_host(sel->buffer) ||
                (size_t) ggml_nelements(sel) > ia.cap) {
                continue;
            }
            // computed before the next split (copy() aborts otherwise), i.e. already enqueued on this stream
            bool in_next = false;
            for (int j = 0; j < n_nodes && !in_next; ++j) {
                const ggml_tensor * node = ggml_graph_node(next, j);
                in_next = node == sel || node == sel->view_src;
            }
            if (in_next) {
                continue;
            }
            ids_async_state::pending & p = ia.pend[ia.n_pend++];
            p.sel = sel;
            p.il  = il;
            p.n   = (size_t) ggml_nelements(sel);
            p.la_n    = 0;
            p.la_ntok = 0;
            // C21 AMB_IDS_ONE_READBACK: the readback of sel is enqueued below, together with the predicted ids if any (no
            // stream work in between: the code up to there only inspects tensors on the host)
            const ggml_tensor * one_t = nullptr;
            if (!ia.one_rb) {
                ggml_backend_tensor_get_async(info->backend, sel, p.dst, 0, ggml_nbytes(sel));
            }
            // AMB_MOE_LOOKAHEAD: the predicted ids of layer il + 1, computed before the real router of layer il (so in this
            // split or an earlier one: not in the next split), read with the same event
            if (la_on && p.la_dst != nullptr && il + 1 < (int32_t) la_ids_t.size() && la_ids_t[il + 1] != nullptr) {
                const ggml_tensor * t = la_ids_t[il + 1];
                const int64_t n_tok = sel->src[0] != nullptr ? sel->src[0]->ne[1] : -1;
                bool ok = t->type == GGML_TYPE_I32 && ggml_is_contiguous(t) && t->ne[0] == la_k && t->ne[1] >= 1 && t->ne[1] <= n_tok &&
                    (size_t) ggml_nelements(t) <= la_cap && t->data != nullptr && t->buffer != nullptr &&
                    !ggml_backend_buffer_is_host(t->buffer); // the same test as for sel (one device: lookahead_config)
                for (int j = 0; j < n_nodes && ok; ++j) {
                    ok = ggml_graph_node(next, j) != t;
                }
                if (ok) {
                    p.la_n    = (size_t) ggml_nelements(t);
                    p.la_ntok = t->ne[1]; // C18b m5: the output rows only for the cropped last layer
                    if (ia.one_rb) {
                        one_t = t;
                    } else {
                        ggml_backend_tensor_get_async(info->backend, t, p.la_dst, 0, ggml_nbytes(t));
                    }
                }
            }
            if (ia.one_rb) {
                // the same bytes into the same pinned destinations as the two get_async calls (srcs are the device
                // addresses those calls read: tensor->data + 0), in one call on the same stream, before the same event
                bool merged = false;
                if (one_t != nullptr) {
                    void *       dsts [2] = { p.dst, p.la_dst };
                    const void * srcs [2] = { sel->data, one_t->data };
                    size_t       sizes[2] = { ggml_nbytes(sel), ggml_nbytes(one_t) };
                    merged = ia.one_fn(info->backend, dsts, srcs, sizes, 2);
                }
                if (merged) {
                    ia.one_merged++;
                } else {
                    // no predicted ids (one readback anyway), or the batched call failed (nothing enqueued is assumed;
                    // if part was enqueued, the same bytes are copied to the same places again, in stream order)
                    ggml_backend_tensor_get_async(info->backend, sel, p.dst, 0, ggml_nbytes(sel));
                    if (one_t != nullptr) {
                        ggml_backend_tensor_get_async(info->backend, one_t, p.la_dst, 0, ggml_nbytes(one_t));
                        ia.one_fb++;
                    }
                }
            }
        }
        if (ia.n_pend > 0) {
            ggml_backend_event_record(ia.ev, info->backend);
            ia.issued += (uint64_t) ia.n_pend;
        }
    }

    // AMB_IDS_ASYNC: forget the pending readbacks (also before each graph evaluation, so none survives an aborted one)
    void ids_async_reset() {
        hf_h_il = -1; // AMB_MOE_HIT_FIRST: (ids_post_split returns before this call while the mark is set)
        for (int k = 0; k < ia.n_pend; ++k) {
            ia.unused += ia.pend[k].sel != nullptr ? 1 : 0;
            ia.pend[k].sel  = nullptr;
            ia.pend[k].il   = -1;
            ia.pend[k].la_n = 0;
        }
        ia.n_pend = 0;
        // C18b ISSUE=post: a deferred issue never outlives the launch of its split (ids_post_split consumes it before this
        // call) nor an evaluation (this is also called before every graph evaluation); nothing was admitted for it
        if (la_pend.active) {
            la_pend.active = false;
            la_post_dropped++;
        }
    }

    // AMB_IDS_ASYNC: the ids of layer il read back by ids_post_split (consumed), nullptr if there is no such readback
    const ids_async_state::pending * ids_async_take(const ggml_tensor * sel, int32_t il) {
        for (int k = 0; k < ia.n_pend; ++k) {
            if (ia.pend[k].sel == sel && ia.pend[k].il == il) {
                ia.pend[k].sel = nullptr; // consumed, not "unused"
                return &ia.pend[k];
            }
        }
        return nullptr;
    }

    // AMB_MOE_PREFETCH: the upload backends and events, prefetch stays off if any of them is missing
    void prefetch_init(int mode) {
        if (const char * v = getenv("AMB_MOE_PREFETCH_K"); v != nullptr && *v != '\0') {
            pf_k = std::clamp(atoi(v), 1, 24);
        }
        if (const char * v = getenv("AMB_MOE_PREFETCH_MIN_LAYER"); v != nullptr && *v != '\0') {
            pf_min_layer = std::max(1, atoi(v));
        }
        bool ok = true;
        int n_groups = 0;
        for (group & g : groups) {
            // the slots of the in-flight batches and of the selections they must not evict, with a wide margin
            g.pf_ok = g.n_slots > 0 && pf_k <= g.ref[0]->ne[2] && g.n_slots >= 8*(n_expert_used + pf_k);
            if (!g.pf_ok) {
                continue;
            }
            n_groups++;
            device & d = devices[g.id];
            if (d.upload == nullptr) {
                ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
                d.upload = ggml_backend_dev_init(dev, nullptr);
                d.pf_ev[0] = d.upload ? ggml_backend_event_new(dev) : nullptr;
                d.pf_ev[1] = d.upload ? ggml_backend_event_new(dev) : nullptr;
                ok = ok && d.upload != nullptr && d.pf_ev[0] != nullptr && d.pf_ev[1] != nullptr;
            }
        }
        if (!ok || n_groups == 0) {
            LLAMA_LOG_WARN("%s: AMB_MOE_PREFETCH=%d: %s, prefetch is off\n", __func__, mode,
                n_groups == 0 ? "no group has enough slots" : "no second backend or events for the device");
            prefetch_free();
            for (group & g : groups) {
                g.pf_ok = false;
            }
            return;
        }
        for (group & g : groups) {
            if (g.pf_ok) {
                g.lru.pf_seq.assign(g.n_slots, 0);
                g.lru.pf_unused.assign(g.n_slots, 0);
            }
        }
        pf_mode = mode;
        for (size_t il = 0; il < layers.size() && pf_first_layer < 0; ++il) {
            if (layers[il].ig >= 0) {
                pf_first_layer = (int32_t) il;
            }
        }
        pf_ids_t.assign(layers.size(), nullptr);
        pf_pred.resize(pf_k);
        LLAMA_LOG_INFO("%s: AMB_MOE_PREFETCH=%d: prefetching the predicted experts (P%d, k = %d) of the layers >= %d of single-token "
            "ubatches on a second stream, %d of %zu groups\n", __func__, mode, mode, pf_k, pf_min_layer, n_groups, groups.size());
    }

    // wait for the upload streams, so no upload writes a freed bank
    void prefetch_free() {
        for (device & d : devices) {
            if (d.upload != nullptr) {
                ggml_backend_synchronize(d.upload);
            }
            for (ggml_backend_event_t & ev : d.pf_ev) {
                if (ev != nullptr) {
                    ggml_backend_event_free(ev);
                    ev = nullptr;
                }
            }
            if (d.upload != nullptr) {
                ggml_backend_free(d.upload);
                d.upload = nullptr;
            }
        }
    }

    //
    // AMB_ELASTIC_KV level 3: elastic banks
    //
    // Invariants (one device): bank b of group g has its chunks [off/gran, off/gran + m) mapped and the rest of its range
    // [.., off/gran + m_cap) unmapped; m_base <= m <= m_cap; the chunks m - m_base are lent (they belong to a KV cache).
    // lru.n_active = the slots whose memory, plus the over-read slot after them, is mapped in every bank of the group
    // (el_active), never below n_base; the slots [n_active, n_slots) are parked (no expert, in no list). A slot is parked
    // before its memory is unmapped, and activated only after its memory is mapped and zero filled.

    int32_t el_active(const group & g) const {
        int64_t s = g.n_slots;
        for (const auto & b : g.eb) {
            s = std::min<int64_t>(s, (int64_t) (b.m*el_gran/b.nb2) - 1);
        }
        GGML_ASSERT(s >= g.n_base);
        return (int32_t) s;
    }

    // after chunks/spans arrived in the bank tails (zero filled): the slots they complete become active; with
    // AMB_ELASTIC_POISON=1 these slots are filled with 0xFF except their first el_poison_head bytes (a lookup that reads
    // a lent slot it did not fill sees NaN; the MMQ over-read into the next slot head still sees zeros)
    void el_activate(const char * what) {
        for (group & g : groups) {
            if (g.n_slots == 0) {
                continue;
            }
            const int32_t a = g.lru.n_active;
            g.lru.set_active(el_active(g));
            const int32_t b = g.lru.n_active;
            if (el_poison && b > a) {
                ggml_backend_t backend = devices[g.id].backend;
                for (ggml_tensor * bank : g.banks) {
                    const size_t nb2 = bank->nb[2];
                    char * base = (char *) bank->data + (size_t) a*nb2;
                    const size_t n = (size_t) (b - a);
                    if (!el_memset2d(backend, base, n*nb2, 0xFF, n*nb2, 1) ||
                        !el_memset2d(backend, base, nb2, 0, std::min(el_poison_head, nb2), n)) {
                        GGML_ABORT("amb_elastic: AMB_ELASTIC_POISON: memset of lent slots failed");
                    }
                }
                ggml_backend_synchronize(backend);
                el_poisoned += (uint64_t) (b - a);
            }
            el_verify(g, what);
        }
    }

    // AMB_ELASTIC_CHECK=1: every slot-map entry of the group (all layers, all loop visits) points to an active,
    // unparked slot that holds that expert; parked slots hold nothing; the active slots are backed (n_active <= el_active)
    void el_verify(const group & g, const char * what) {
        if (!el_check || !el_live || g.n_slots == 0) {
            return;
        }
        const moe_cache_lru & lru = g.lru;
        if (lru.n_active > el_active(g) || lru.n_active < g.n_base) {
            GGML_ABORT("amb_elastic: check (%s): %d active slots, %d backed, %d own", what, lru.n_active, el_active(g), g.n_base);
        }
        for (int32_t s = 0; s < lru.n_slots; ++s) {
            if ((s >= lru.n_active) != (lru.parked[s] != 0) && !(ring_out && s < g.n_base)) {
                GGML_ABORT("amb_elastic: check (%s): slot %d parked=%d with %d active", what, s, (int) lru.parked[s], lru.n_active);
            }
            if (lru.parked[s] && lru.key_of[s] >= 0) {
                GGML_ABORT("amb_elastic: check (%s): parked slot %d holds key %d", what, s, lru.key_of[s]);
            }
        }
        for (int32_t il : g.layers) {
            const int32_t * map = lru.slot_map[il];
            if (map == nullptr) {
                continue;
            }
            const int32_t ip = rep[il];
            for (int32_t e = 0; e < lru.n_expert; ++e) {
                const int32_t s = map[e];
                if (s >= lru.n_slots) {
                    continue;
                }
                if (s >= lru.n_active || lru.parked[s] || lru.key_of[s] != lru.make_key(ip, e)) {
                    GGML_ABORT("amb_elastic: check (%s): layer %d expert %d maps to slot %d (active %d, parked %d, key %d)",
                        what, il, e, s, lru.n_active, (int) lru.parked[s], lru.key_of[s]);
                }
            }
        }
        el_checks++;
    }

    static size_t el_lent(const group & g) {
        size_t res = 0;
        for (const auto & b : g.eb) {
            res += b.m - b.m_base;
        }
        return res;
    }

    // the device buffer: a reserved range for every bank at its capacity, the base chunks get their own memory
    void el_alloc(device & d) {
        size_t va = 0;
        std::vector<size_t> base;
        for (const group & g : groups) {
            if (g.n_slots == 0 || &devices[g.id] != &d) {
                continue;
            }
            for (const auto & b : g.eb) {
                va = std::max(va, b.off + b.m_cap*el_gran);
                for (size_t c = 0; c < b.m_base; ++c) {
                    base.push_back(b.off/el_gran + c);
                }
            }
        }
        d.buf.reset(el_fns->reserve(el_buft, va));
        if (!d.buf) {
            throw std::runtime_error("amb_elastic: failed to reserve the MoE cache range");
        }
        if (!el_fns->map_new(d.buf.get(), base.data(), base.size())) {
            throw std::runtime_error("failed to allocate the MoE cache (amb_elastic)");
        }
        char * ptr = (char *) ggml_backend_buffer_get_base(d.buf.get());
        // as a compute buffer while the banks are placed: the CUDA init_tensor then does not zero the tail row pad of a
        // quantized bank, which lies in the unmapped part of its range (the usage becomes WEIGHTS below, as before)
        ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        for (const group & g : groups) {
            if (g.n_slots == 0 || &devices[g.id] != &d) {
                continue;
            }
            for (size_t ip = 0; ip < g.banks.size(); ++ip) {
                GGML_ASSERT(ggml_backend_buft_get_alloc_size(el_buft, g.banks[ip]) <= g.eb[ip].m_cap*el_gran);
                if (ggml_backend_tensor_alloc(d.buf.get(), g.banks[ip], ptr + g.eb[ip].off) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("amb_elastic: failed to place a MoE cache bank");
                }
            }
        }
        for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
            if (t->view_src != nullptr && t->buffer == nullptr) {
                if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("amb_elastic: failed to init a MoE cache view");
                }
            }
        }
        ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_clear(d.buf.get(), 0); // the mapped chunks only
        d.buf_size = base.size()*el_gran;
    }

    // chunks of src into the bank tails: each to the group that borrowed least for the size of its experts, in it to the
    // bank that limits its slots; the caller guarantees no GPU work uses the src chunks
    // C17c: the KV chunk paths below work on the top of the bank tails, which must hold no span
    void el_evict_spans() {
        if (el_n_spans() > 0 && el_span_evict != nullptr) {
            el_span_evict(el_span_evict_ud);
        }
        GGML_ASSERT(el_n_spans() == 0 && "amb_elastic: KV chunks moved while compute spans are lent");
    }

    size_t el_n_spans() const {
        size_t n = 0;
        for (const group & g : groups) {
            for (const auto & b : g.eb) {
                n += b.sp.size();
            }
        }
        return n;
    }

    // C17c: spans of src (heads[i], chunk count from the buffer) onto the bank tails: each to the group that borrowed
    // least for the size of its experts (among those with a bank that has room for it), in it to the bank that limits
    // its slots (among those with room); stops at the first span nothing has room for. Zero filled at the destination
    size_t el_give_spans(ggml_backend_buffer_t src, const size_t * heads, size_t n, size_t * n_calls) {
        GGML_ASSERT(el_live);
        GGML_ASSERT(!ring_out && "amb_elastic: compute spans given while the overlap ring is borrowed");
        lookahead_drain("compute spans given");
        device & d = devices[0];
        std::vector<size_t> dst;
        dst.reserve(n);
        size_t chunks = 0;
        for (size_t i = 0; i < n; ++i) {
            const size_t k = el_fns->span_of(src, heads[i]);
            GGML_ASSERT(k > 0);
            group * gb = nullptr;
            double  gs = 0.0;
            for (group & g : groups) {
                if (g.n_slots == 0 || g.host_bytes == 0) {
                    continue;
                }
                bool room = false;
                for (const auto & b : g.eb) {
                    room = room || b.m + k <= b.m_cap;
                }
                const double s = (double) el_lent(g)/g.host_bytes;
                if (room && (gb == nullptr || s < gs)) {
                    gb = &g;
                    gs = s;
                }
            }
            if (gb == nullptr) {
                break;
            }
            group::ebank * bb = nullptr;
            for (auto & b : gb->eb) {
                if (b.m + k <= b.m_cap && (bb == nullptr || b.m*el_gran/b.nb2 < bb->m*el_gran/bb->nb2)) {
                    bb = &b;
                }
            }
            const size_t start = bb->off/el_gran + bb->m;
            dst.push_back(start);
            bb->sp.push_back({ start, k });
            bb->m  += k;
            chunks += k;
        }
        if (dst.empty()) {
            return 0;
        }
        if (!el_fns->move_spans(src, heads, d.buf.get(), dst.data(), dst.size(), true, n_calls)) {
            GGML_ABORT("amb_elastic: moving %zu spans into the MoE cache failed (driver error)", dst.size());
        }
        el_activate("give");
        el_given += chunks;
        return dst.size();
    }

    // C17c: every span back to the heads of dst (n must be the number of spans; each head gets a span of its length)
    void el_take_spans(ggml_backend_buffer_t dst, const size_t * heads, const size_t * lens, size_t n, size_t * n_calls) {
        GGML_ASSERT(el_live);
        GGML_ASSERT(!ring_out && "amb_elastic: compute spans taken back while the overlap ring is borrowed");
        GGML_ASSERT(n == el_n_spans() && "amb_elastic: span count mismatch");
        lookahead_drain("compute spans taken back");
        device & d = devices[0];
        // the kernels of the last graph may still read the banks, the upload streams may still write slots
        el_fns->sync(d.buf.get());
        // pair the spans with the heads by length: all compute spans have the same length except possibly a remainder
        std::vector<std::pair<size_t, size_t>> src; // { first chunk, chunks }
        src.reserve(n);
        size_t chunks = 0;
        for (group & g : groups) {
            for (auto & b : g.eb) {
                while (!b.sp.empty()) {
                    const auto s = b.sp.back();
                    GGML_ASSERT(s.first + s.second == b.off/el_gran + b.m && "amb_elastic: a span is not at the top of its bank");
                    b.m -= s.second;
                    b.sp.pop_back();
                    src.push_back(s);
                    chunks += s.second;
                }
            }
        }
        // pair each destination head with a span of its length (the compute buffer's spans have one length, except a
        // remainder of single chunks)
        std::vector<size_t> src_heads(n);
        std::vector<bool>   used(n, false);
        for (size_t i = 0; i < n; ++i) {
            bool found = false;
            for (size_t j = 0; j < n && !found; ++j) {
                if (!used[j] && src[j].second == lens[i]) {
                    used[j]      = true;
                    src_heads[i] = src[j].first;
                    found        = true;
                }
            }
            GGML_ASSERT(found && "amb_elastic: no lent span has the length of a compute buffer head");
        }
        // park the slots on these spans before they are unmapped (host only, the device is idle)
        for (group & g : groups) {
            if (g.n_slots > 0) {
                el_dropped += g.lru.set_active(el_active(g));
                el_verify(g, "take");
            }
        }
        if (!el_fns->move_spans(d.buf.get(), src_heads.data(), dst, heads, n, true, n_calls)) {
            GGML_ABORT("amb_elastic: moving %zu spans out of the MoE cache failed (driver error)", n);
        }
        el_taken += chunks;
    }

    size_t el_give(ggml_backend_buffer_t src, const size_t * chunks, size_t n) {
        GGML_ASSERT(el_live);
        GGML_ASSERT(!ring_out && "amb_elastic: KV shrink while the overlap ring is borrowed");
        lookahead_drain("KV chunks given (KV shrink)");
        el_evict_spans();
        device & d = devices[0];
        std::vector<size_t> dst;
        dst.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            group * gb = nullptr;
            double  gs = 0.0;
            for (group & g : groups) {
                if (g.n_slots == 0 || g.host_bytes == 0) {
                    continue;
                }
                bool room = false;
                for (const auto & b : g.eb) {
                    room = room || b.m < b.m_cap;
                }
                const double s = (double) el_lent(g)/g.host_bytes;
                if (room && (gb == nullptr || s < gs)) {
                    gb = &g;
                    gs = s;
                }
            }
            if (gb == nullptr) {
                break;
            }
            group::ebank * bb = nullptr;
            for (auto & b : gb->eb) {
                if (b.m < b.m_cap && (bb == nullptr || b.m*el_gran/b.nb2 < bb->m*el_gran/bb->nb2)) {
                    bb = &b;
                }
            }
            dst.push_back(bb->off/el_gran + bb->m);
            bb->m++;
        }
        if (dst.empty()) {
            return 0;
        }
        if (!el_fns->move(src, chunks, d.buf.get(), dst.data(), dst.size(), true)) {
            GGML_ABORT("amb_elastic: moving %zu chunks into the MoE cache failed (driver error)", dst.size());
        }
        el_activate("give");
        el_given += dst.size();
        return dst.size();
    }

    // n lent chunks back to dst: from the group that borrowed most, in it from the bank with the most slots to spare
    void el_take(ggml_backend_buffer_t dst, const size_t * chunks, size_t n) {
        GGML_ASSERT(el_live);
        GGML_ASSERT(!ring_out && "amb_elastic: KV growth while the overlap ring is borrowed");
        lookahead_drain("KV chunks taken back (KV growth)");
        el_evict_spans();
        device & d = devices[0];
        // the kernels of the last graph may still read the banks, the upload streams (C9) may still write slots
        el_fns->sync(d.buf.get());
        std::vector<size_t> src;
        src.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            group * gb = nullptr;
            double  gs = 0.0;
            for (group & g : groups) {
                if (g.n_slots == 0 || g.host_bytes == 0 || el_lent(g) == 0) {
                    continue;
                }
                const double s = (double) el_lent(g)/g.host_bytes;
                if (gb == nullptr || s > gs) {
                    gb = &g;
                    gs = s;
                }
            }
            if (gb == nullptr) {
                GGML_ABORT("amb_elastic: %zu chunks asked back, %zu lent: the chunk accounting is broken", n, i);
            }
            group::ebank * bb = nullptr;
            for (auto & b : gb->eb) {
                if (b.m > b.m_base && (bb == nullptr || (b.m - 1)*el_gran/b.nb2 > (bb->m - 1)*el_gran/bb->nb2)) {
                    bb = &b;
                }
            }
            src.push_back(bb->off/el_gran + bb->m - 1);
            bb->m--;
        }
        // park the slots on these chunks before they are unmapped (host only, the device is idle)
        for (group & g : groups) {
            if (g.n_slots > 0) {
                el_dropped += g.lru.set_active(el_active(g));
                el_verify(g, "take");
            }
        }
        if (!el_fns->move(d.buf.get(), src.data(), dst, chunks, n, true)) {
            GGML_ABORT("amb_elastic: moving %zu chunks out of the MoE cache failed (driver error)", n);
        }
        el_taken += n;
    }

    //
    // AMB_PF_OVERLAP ring in the banks (C13)
    //
    // The ring takes the slots [r0, r1 = n_base) of a group: inside the cache's own chunks, so always mapped and at the same
    // addresses for the life of the context (the overlap redirects scheduler copies to these addresses, and a reused graph
    // replays its CUDA graph without re-checking pointers). r1 <= n_base <= n_active: never the guard slot (n_active) nor a
    // slot parked by lending, and KV growth/shrink only moves chunks at m > m_base. Each bank of the group frees
    // [off + r0*nb2, off + r1*nb2); ring slots are cut from these ranges at 256 byte alignment.

    static constexpr size_t ring_align = 256;

    // ring slots of `slot` bytes that fit when group g lends its slots [n_base - p, n_base)
    int32_t ring_fit(const group & g, int32_t p, size_t slot, std::vector<void *> * ptrs) const {
        const char * base = (const char *) ggml_backend_buffer_get_base(devices[g.id].buf.get());
        int32_t n = 0;
        for (const auto & b : g.eb) {
            size_t start = b.off + (size_t) (g.n_base - p)*b.nb2;
            start = (start + ring_align - 1)/ring_align*ring_align;
            const size_t end = b.off + (size_t) g.n_base*b.nb2;
            for (size_t a = start; a + slot <= end; a += slot) {
                if (ptrs) {
                    ptrs->push_back((void *) (base + a));
                }
                n++;
            }
        }
        return n;
    }

    int32_t ring_plan(size_t slot_size, int32_t n_want) {
        ring_parts.clear();
        ring_ptrs.clear();
        if (!el_live || n_want <= 0) {
            return 0;
        }
        // C13c: the slot heads are cleared at every return, no ring without the strided memset
        {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(devices[0].backend));
            ring_memset2d = reg ? (memset2d_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_memset2d_async") : nullptr;
        }
        if (ring_memset2d == nullptr) {
            LLAMA_LOG_WARN("%s: amb_elastic: the backend has no ggml_backend_cuda_memset2d_async, no overlap ring in the MoE cache banks\n", __func__);
            return 0;
        }
        ring_poison = 0;
        if (const char * v = getenv("AMB_RING_POISON"); v != nullptr && *v != '\0') {
            ring_poison = std::clamp(atoi(v), 0, 2);
        }
        ring_slot = (slot_size + ring_align - 1)/ring_align*ring_align;
        // the groups with the most own slots first
        std::vector<int32_t> order;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            if (groups[ig].n_slots > 0 && !groups[ig].eb.empty()) {
                order.push_back((int32_t) ig);
            }
        }
        std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return groups[a].n_base > groups[b].n_base; });
        int32_t have = 0;
        for (int32_t ig : order) {
            const group & g = groups[ig];
            const int32_t need = n_want - have;
            if (need <= 0) {
                break;
            }
            // the fewest slots that give `need` ring slots, else all own slots and what they give
            int32_t p = 1;
            while (p < g.n_base && ring_fit(g, p, ring_slot, nullptr) < need) {
                p++;
            }
            const int32_t got = std::min(need, ring_fit(g, p, ring_slot, nullptr));
            if (got <= 0) {
                continue;
            }
            std::vector<void *> ptrs;
            ring_fit(g, p, ring_slot, &ptrs);
            ptrs.resize(got);
            ring_ptrs.insert(ring_ptrs.end(), ptrs.begin(), ptrs.end());
            ring_parts.push_back({ ig, g.n_base - p, g.n_base });
            have += got;
        }
        if (have > 0 && ring_poison > 0) {
            LLAMA_LOG_WARN("%s: amb_elastic: AMB_RING_POISON=%d (debug): at each ring return the ring range is filled with 0xFF%s\n",
                __func__, ring_poison, ring_poison == 2 ? " and the slot heads are NOT cleared (reproduces the MMQ over-read hazard)" :
                ", then the slot heads are cleared as usual");
        }
        return have;
    }

    // before the uploads into the ring are enqueued; the caller orders them after all earlier compute-stream work
    // returns the experts dropped from the ring range
    size_t ring_borrow() {
        GGML_ASSERT(el_live && !ring_out);
        lookahead_drain("ring borrow");
        size_t dropped = 0;
        for (const ring_part & rp : ring_parts) {
            moe_cache_lru & lru = groups[rp.ig].lru;
            GGML_ASSERT(rp.r1 <= lru.n_active);
            for (int32_t s = rp.r0; s < rp.r1; ++s) {
                dropped += lru.park(s) ? 1 : 0;
            }
        }
        ring_out = true;
        ring_n_borrow++;
        ring_dropped += dropped;
        return dropped;
    }

    // after the graph was enqueued and the upload stream drained: later writes to these slots come from the compute
    // stream (ordered after the kernels that read the ring) or after a device synchronize (KV growth)
    //
    // C13c: the ring range holds leftovers of whole host tensors at arbitrary offsets. A fill writes exactly nb2 bytes into
    // a slot, and CUDA MMQ reads past the last row of an expert into the head of the next slot (K is stepped by 256: the
    // 640-wide down rows read 4 IQ4_NL blocks = 72 B or 2 Q2_0 blocks = 36 B too far, at most 7/3 blocks by the kernel's
    // structure); a leftover there can decode as an Inf/NaN scale, and NaN x 0 = NaN. So the first ring_head_zero bytes of
    // every slot [r0, r1) of every bank are cleared, on the compute stream of the device that owns the banks: after the
    // kernels of this graph that read the ring, before any later fill (fills are enqueued on the same stream). Slot r1 =
    // n_base is never written by the ring (ring slots end at or before it) and may hold a cached (lent) expert: not touched.
    // Nothing reads before a slot start. Host bookkeeping (unpark) after the clear is enqueued.
    int64_t ring_return() {
        GGML_ASSERT(ring_out);
        for (const ring_part & rp : ring_parts) {
            const group & g = groups[rp.ig];
            ggml_backend_t backend = devices[g.id].backend;
            const size_t n = (size_t) (rp.r1 - rp.r0);
            for (size_t ip = 0; ip < g.banks.size(); ++ip) {
                const size_t nb2 = g.banks[ip]->nb[2];
                char * base = (char *) g.banks[ip]->data + (size_t) rp.r0*nb2;
                if (ring_poison > 0 && !ring_memset2d(backend, base, n*nb2, 0xFF, n*nb2, 1)) {
                    GGML_ABORT("amb_elastic: AMB_RING_POISON: memset of the ring range failed");
                }
                if (ring_poison != 2 && !ring_memset2d(backend, base, nb2, 0, std::min(ring_head_zero, nb2), n)) {
                    GGML_ABORT("amb_elastic: clearing the ring slot heads failed");
                }
            }
        }
        int64_t unparked = 0;
        for (const ring_part & rp : ring_parts) {
            moe_cache_lru & lru = groups[rp.ig].lru;
            for (int32_t s = rp.r1 - 1; s >= rp.r0; --s) {
                lru.unpark(s);
                unparked++;
            }
        }
        ring_out = false;
        return unparked;
    }

    int64_t ring_slots_parked() const {
        int64_t n = 0;
        for (const ring_part & rp : ring_parts) {
            n += rp.r1 - rp.r0;
        }
        return n;
    }

    std::string el_describe() const {
        std::string res = "slots=";
        bool first = true;
        for (const group & g : groups) {
            if (g.n_slots == 0) {
                continue;
            }
            char buf[64];
            snprintf(buf, sizeof(buf), "%s%d+%d%s", first ? "" : ",", g.n_base, g.lru.n_active - g.n_base,
                el_floor_bytes > 0 && g.lru.n_active < g.n_graph ? "(<floor)" : "");
            res += buf;
            first = false;
        }
        char buf[128];
        snprintf(buf, sizeof(buf), " dropped=%llu chunks_in=%llu chunks_out=%llu spans=%zu",
            (unsigned long long) el_dropped, (unsigned long long) el_given, (unsigned long long) el_taken, el_n_spans());
        return res + buf;
    }

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    // trace format, little endian: "MCT1", then u32 n_layer, n_expert_used, n_group
    // per group: u32 device, n_expert, n_layers, the layers (u32), f64 split, split_sum, u64 group host bytes, device host bytes, alignment,
    //            u32 n_tensors, the bytes of one expert of each tensor (u64)
    // then one record per slot lookup: i32 layer, experts per token, n_ids, the ids (i32)
    void write_trace_header(double split_sum) {
        auto u32 = [&](uint32_t v) { fwrite(&v, sizeof(v), 1, trace); };
        auto u64 = [&](uint64_t v) { fwrite(&v, sizeof(v), 1, trace); };
        auto f64 = [&](double   v) { fwrite(&v, sizeof(v), 1, trace); };
        fwrite("MCT1", 1, 4, trace);
        u32(layers.size());
        u32(n_expert_used);
        u32(groups.size());
        for (const group & g : groups) {
            const device & d = devices[g.id];
            u32(g.id);
            u32(g.ref[0]->ne[2]);
            u32(g.layers.size());
            for (int32_t il : g.layers) {
                u32(il);
            }
            f64(d.split);
            f64(split_sum);
            u64(g.host_bytes);
            u64(d.host_bytes);
            u64(ggml_backend_buft_get_alignment(d.buft));
            u32(g.ref.size());
            for (const ggml_tensor * t : g.ref) {
                u64(t->nb[2]);
            }
        }
    }

    void write_trace(int32_t il, int32_t n_used) {
        const int32_t rec[3] = { il, n_used, (int32_t) ids.size() };
        fwrite(rec, sizeof(int32_t), 3, trace);
        fwrite(ids.data(), sizeof(int32_t), ids.size(), trace);
    }

    ~impl() {
        // AMB_ELASTIC_KV level 3: the lent chunks go with the banks (released when the buffer is freed); the KV caches
        // must not ask for them anymore
        if (el_live) {
            LLAMA_LOG_INFO("amb_elastic: final cache=moe %s\n", el_describe().c_str());
            if (el_check || el_poison) {
                LLAMA_LOG_INFO("amb_elastic: AMB_ELASTIC_CHECK=%d: %llu slot-map checks passed; AMB_ELASTIC_POISON=%d: %llu lent slots poisoned\n",
                    (int) el_check, (unsigned long long) el_checks, (int) el_poison, (unsigned long long) el_poisoned);
            }
        }
        for (llama_kv_elastic * b : el_borrowers) {
            b->lender_gone();
        }
        el_borrowers.clear();
        log_stats();
        if (loop_dedup_check) {
            LLAMA_LOG_INFO("amb_loop_dedup: check: %llu plans verified\n", (unsigned long long) loop_dedup_checks);
        }
        log_ids_async("final");
        hf_log("final");
        ids_async_free();
        if (predict) {
            log_predict("final     ", false);
        }
        if (pf_mode != 0) {
            log_prefetch("final     ", pf_totals(), pf_stats{}, stats_rounds);
        }
        // AMB_MOE_LOOKAHEAD: no upload may write a freed bank; then the last timing events, the final lines
        if (la_on) {
            lookahead_drain("cache destruction");
            lookahead_timing_read();
            log_lookahead("final     ", false);
        }
        // AMB_MOE_MISS_OVERLAP: U was synchronized by the drain above (M5), the context synchronized C before
        mo_log("final     ", false);
        mo_free();
        lookahead_free();
        // before the members free the banks
        prefetch_free();
        if (trace != nullptr) {
            fclose(trace);
        }
    }

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    // cpu_part: the graph computes the misses in the CPU map of the layer (see get_cpu_map)
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, bool cpu_part) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        // n_base: the slots that always exist (n_slots without AMB_ELASTIC_KV lending), so a reused graph stays valid
        // C4b: n_graph = n_base + the compute floor (AMB_ELASTIC_FLOOR_MIB), n_base without it
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_graph) {
            return nullptr;
        }
        GGML_ASSERT(!cpu_part || l.slot_map_cpu != nullptr);
        return cpu_part ? l.slot_map_cpu : l.slot_map;
    }

    // the experts of layer il computed on the CPU: I32 [1, n_expert] in host memory, e for these experts, -1 for the others
    // nullptr if no misses of layer il are computed on the CPU for ubatches of n_tokens tokens
    ggml_tensor * get_cpu_map(int32_t il, int64_t n_tokens) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].cpu_map == nullptr) {
            return nullptr;
        }
        // the scheduler would run the CPU part on a device that offloads such batches from host weights
        ggml_tensor probe = {};
        probe.op    = GGML_OP_MUL_MAT_ID;
        probe.type  = GGML_TYPE_F32;
        probe.ne[0] = 1;
        probe.ne[1] = n_expert_used;
        probe.ne[2] = n_tokens;
        probe.ne[3] = 1;
        for (const device & d : devices) {
            if (ggml_backend_offload_op(d.backend, &probe)) {
                return nullptr;
            }
        }
        return layers[il].cpu_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // AMB_MOE_LOOKAHEAD: the banks are read on the compute stream below, no upload may still write them
        lookahead_drain("large-batch expert copies from the cache");

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] >= g.n_slots) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] < g.n_slots && slots[e + n] == slots[e] + n) {
            n++;
        }

        // AMB_MOE_PREFETCH: the slot may hold an expert whose prefetch upload is in flight
        if (pf_mode != 0) {
            device & d = devices[g.id];
            if (d.upload != nullptr && d.pf_waited < d.pf_last) {
                ggml_backend_event_wait(backend, d.pf_ev[d.pf_last % 2]);
                d.pf_waited = d.pf_last;
                pf.waits_hit++;
            }
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    // AMB_MOE_PREFETCH: the layers the graph builds a predictor for, see qwen4exp
    // layer il is prefetched by copy() of layer il - 1, so both are cached on the same device
    int prefetch_mode(int32_t il, int64_t n_tokens) const {
        if (pf_mode == 0 || n_tokens != 1 || il < pf_min_layer || il < 1 || il >= (int32_t) layers.size()) {
            return 0;
        }
        if (get_slot_map(il, 1, n_expert_used, false) == nullptr || get_slot_map(il - 1, 1, n_expert_used, false) == nullptr) {
            return 0;
        }
        const group & g = groups[layers[il].ig];
        if (!g.pf_ok || g.id != groups[layers[il - 1].ig].id || devices[g.id].upload == nullptr) {
            return 0;
        }
        return pf_mode;
    }

    // the predicted ids of layer il + 1 that copy() of layer il can read with its readback, nullptr if none
    const ggml_tensor * prefetch_ids(int32_t il, int64_t n_tokens) {
        if (pf_mode == 0 || n_tokens != 1 || il + 1 >= (int32_t) layers.size() || pf_ids_t[il + 1] == nullptr) {
            return nullptr;
        }
        const ggml_tensor * t = pf_ids_t[il + 1];
        const device & d = devices[groups[layers[il].ig].id];
        const bool ok = prefetch_mode(il + 1, 1) != 0 &&
            t->type == GGML_TYPE_I32 && t->ne[0] == pf_k && ggml_nelements(t) == pf_k && ggml_is_contiguous(t) &&
            t->buffer != nullptr && t->data != nullptr && ggml_backend_buffer_get_type(t->buffer) == d.buft;
        if (!ok) {
            pf.bad++;
            return nullptr;
        }
        return t;
    }

    // right after a synchronize of the compute stream of device id: the batches it was made to wait for are complete
    void prefetch_ack(int32_t id) {
        device & d = devices[id];
        if (d.upload == nullptr || d.pf_done >= d.pf_waited) {
            return;
        }
        d.pf_done = d.pf_waited;
        for (group & g : groups) {
            if (g.id == id && g.pf_ok) {
                g.lru.pf_done = d.pf_done;
            }
        }
    }

    // before the kernels of a split: need = the plan selected an in-flight slot, C waits for the last batch (it covers all
    // batches, U is in order); else, C waits for the batch before the last one if nobody did, so that its slots are released
    // at the next synchronize (it was started two copies ago, so the wait is expected to be short)
    void prefetch_order(ggml_backend_t backend, int32_t id, bool need) {
        device & d = devices[id];
        if (d.upload == nullptr || d.pf_last == 0) {
            return;
        }
        if (need) {
            if (d.pf_waited < d.pf_last) {
                ggml_backend_event_wait(backend, d.pf_ev[d.pf_last % 2]);
                d.pf_waited = d.pf_last;
                pf.waits_hit++;
            }
        } else if (d.pf_last >= 2 && d.pf_waited < d.pf_last - 1) {
            ggml_backend_event_wait(backend, d.pf_ev[(d.pf_last - 1) % 2]);
            d.pf_waited = d.pf_last - 1;
            pf.waits_ack++;
        }
    }

    // admit the predicted experts of layer il (in pf_pred) that are not cached and upload them on U, after the plan of il_real
    // safety valve: a predicted expert without an eligible victim is skipped, the real path never waits for a prefetch it
    // does not use (except the release wait of prefetch_order)
    void prefetch(int32_t il_real, int32_t il) {
        layer & l = layers[il];
        group & g = groups[l.ig];
        device & d = devices[g.id];
        moe_cache_lru & lru = g.lru;
        const int32_t n_expert = (int32_t) l.slot_map->ne[1];

        pf.batches++;
        for (int32_t e : pf_pred) {
            if (e < 0 || e >= n_expert) {
                pf.bad++;
                return;
            }
        }

        // all of the batch gets the number of the batch, its slots are in flight from now on
        const uint64_t seq = d.pf_last + 1;
        const int32_t * slots = lru.slot_map[il];
        pf_fills.clear();
        for (int32_t e : pf_pred) {
            pf.considered++;
            if (slots[e] < lru.n_slots) {
                pf.cached++; // cached or in flight (also a repeated id)
                continue;
            }
            const int32_t s = lru.victim(il_real, true);
            if (s < 0) {
                pf.no_victim++;
                continue;
            }
            lru.pf_admit(il, e, s, seq);
            pf_fills.push_back({ e, s });
        }
        if (pf_fills.empty()) {
            return;
        }

        // consecutive experts going to consecutive slots are uploaded together, as in copy()
        std::sort(pf_fills.begin(), pf_fills.end(), [](const moe_cache_lru::fill & a, const moe_cache_lru::fill & b) { return a.expert < b.expert; });
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < pf_fills.size();) {
                size_t n = 1;
                while (i + n < pf_fills.size() && pf_fills[i + n].expert == pf_fills[i].expert + (int32_t) n && pf_fills[i + n].slot == pf_fills[i].slot + (int32_t) n) {
                    n++;
                }
                ggml_backend_tensor_set_async(d.upload, bank, (const uint8_t *) w->data + pf_fills[i].expert*expert_size, pf_fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }
        ggml_backend_event_record(d.pf_ev[seq % 2], d.upload);
        d.pf_last = seq;

        pf.started += pf_fills.size();
        pf.bytes   += bytes;
        pf.uploads++;
    }

    pf_stats pf_totals() const {
        pf_stats t = pf;
        for (const group & g : groups) {
            t.hit_done += g.lru.pf_hit_done;
            t.hit_wait += g.lru.pf_hit_wait;
            t.wasted   += g.lru.pf_wasted;
        }
        return t;
    }

    // AMB_MOE_PREFETCH: one line, for the interval since prev (all zero: cumulative)
    // hits complete = the upload was known complete when a plan selected the slot; hits waited = it may have been in flight,
    // the compute stream waited for its event (the GPU wait is short or zero if the upload had already finished)
    void log_prefetch(const char * what, const pf_stats & t, const pf_stats & prev, int64_t rounds) const {
        pf_stats d = t;
        d.batches    -= prev.batches;
        d.considered -= prev.considered;
        d.cached     -= prev.cached;
        d.no_victim  -= prev.no_victim;
        d.started    -= prev.started;
        d.bytes      -= prev.bytes;
        d.uploads    -= prev.uploads;
        d.waits_hit  -= prev.waits_hit;
        d.waits_ack  -= prev.waits_ack;
        d.bad        -= prev.bad;
        d.steps      -= prev.steps;
        d.blocking   -= prev.blocking;
        d.hit_done   -= prev.hit_done;
        d.hit_wait   -= prev.hit_wait;
        d.wasted     -= prev.wasted;
        d.ns         -= prev.ns;
        size_t pending = 0;
        for (const group & g : groups) {
            for (uint8_t u : g.lru.pf_unused) {
                pending += u;
            }
        }
        const double st = d.steps > 0 ? (double) d.steps : 1.0;
        LLAMA_LOG_INFO("llama_moe_cache: amb_moe_prefetch: round %lld: %s: P%d k=%d: steps = %llu, predicted = %llu (cached %llu), "
            "started = %llu (%.2f MiB, %llu batches), no victim = %llu, hits = %llu (complete %llu, waited %llu), wasted = %llu, "
            "unused now = %zu, stream waits = %llu hit/large-batch + %llu release, skipped = %llu | per step: blocking misses = %.2f, "
            "prefetched = %.2f, hits = %.2f, wasted = %.2f\n",
            (long long) rounds, what, pf_mode, pf_k, (unsigned long long) d.steps, (unsigned long long) d.considered,
            (unsigned long long) d.cached, (unsigned long long) d.started, d.bytes/1024.0/1024.0, (unsigned long long) d.uploads,
            (unsigned long long) d.no_victim, (unsigned long long) (d.hit_done + d.hit_wait), (unsigned long long) d.hit_done,
            (unsigned long long) d.hit_wait, (unsigned long long) d.wasted, pending, (unsigned long long) d.waits_hit,
            (unsigned long long) d.waits_ack, (unsigned long long) d.bad, d.blocking/st, d.started/st, (d.hit_done + d.hit_wait)/st,
            d.wasted/st);
    }

    //
    // AMB_MOE_LOOKAHEAD (C18)
    //

    // see llama_moe_cache::lookahead_mode: configuration, cache layout and n_tokens only
    int lookahead_mode(int32_t il, int64_t n_tokens) const {
        if (!la_build || n_tokens < la_min_tok || n_tokens > la_max_tok || il < la_min_layer || il < 1 || il >= (int32_t) layers.size()) {
            return 0;
        }
        if (layers[il].ig < 0 || layers[il - 1].ig < 0) {
            return 0;
        }
        if (get_slot_map(il, n_tokens, n_expert_used, false) == nullptr || get_slot_map(il - 1, n_tokens, n_expert_used, false) == nullptr) {
            return 0;
        }
        const group & g = groups[layers[il].ig];
        if (!g.la_ok || g.id != groups[layers[il - 1].ig].id) {
            return 0;
        }
        return la_req;
    }

    //
    // AMB_MOE_MISS_OVERLAP (C23), see the member comment for the invariants M1..M6
    //

    // the graph publishes the ids of layer il for a ubatch of n_tokens rows: configuration, cache layout and n_tokens only
    bool miss_overlap(int32_t il, int64_t n_tokens) const {
        return mo_build && n_tokens >= mo_min_tok && n_tokens <= mo_max_tok &&
            get_slot_map(il, n_tokens, n_expert_used, false) != nullptr;
    }

    ggml_tensor * miss_overlap_publish(ggml_context * ctx, ggml_tensor * a, int32_t il, int kind) const {
        GGML_ASSERT(mo_build && il >= 0 && il < (int32_t) layers.size() && (kind == 0 || kind == 1));
        if (!ggml_is_contiguous(a)) {
            a = ggml_cont(ctx, a);
        }
        const size_t k = 2*(size_t) il + (size_t) kind;
        // no_alloc (--fit) graphs are never computed: no buffers there
        void * slot = mo_sdev != nullptr ? (void *) ((char *) mo_sdev + k*GGML_AMB_PUBLISH_SLOT) : nullptr;
        void * ctr  = mo_cdev != nullptr ? (void *) ((uint32_t *) mo_cdev + k) : nullptr;
        ggml_tensor * pub = ggml_amb_publish(ctx, a, slot, ctr, il, kind);
        ggml_format_name(pub, "amb_publish%d-%d", kind, il);
        return pub;
    }

    bool mo_pending(int32_t il, int kind) const {
        const size_t k = 2*(size_t) il + (size_t) kind;
        return mo_on && il >= 0 && k < mo_expect.size() && mo_expect[k] != mo_taken[k];
    }

    // the scheduler may skip its synchronize before copy() of this slot map: copy() will take the published ids (M1, M2)
    bool copy_nosync(const ggml_tensor * src) const {
        // AMB_MOE_HIT_FIRST: copy() of map_H only reads host state; copy() of map_R waits on the ids event itself when
        // the readback is pending (else the scheduler synchronizes and copy() reads the ids back as before)
        if (hf_on && !hf_sync) {
            const auto ih = layer_of.find(src);
            if (ih != layer_of.end()) {
                const layer & l = layers[ih->second.first];
                if (src == l.slot_map_h) {
                    return true;
                }
                if (src == l.slot_map_r) {
                    for (int k = 0; k < ia.n_pend; ++k) {
                        if (ia.pend[k].sel != nullptr && ia.pend[k].il == ih->second.first) {
                            return true;
                        }
                    }
                    return false;
                }
            }
        }
        if (!mo_on) {
            return false;
        }
        const auto it = layer_of.find(src);
        return it != layer_of.end() && !it->second.second && mo_pending(it->second.first, 0);
    }

    // after the launch of a split: count its publish nodes (each launched node executes exactly once, M6)
    void mo_scan(const ggml_cgraph * graph) {
        if (graph == nullptr) {
            return;
        }
        const int n_nodes = ggml_graph_n_nodes(const_cast<ggml_cgraph *>(graph));
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(const_cast<ggml_cgraph *>(graph), i);
            if (node->op != GGML_OP_AMB_PUBLISH) {
                continue;
            }
            const int32_t * p = (const int32_t *) node->op_params;
            const int32_t il = p[4], kind = p[5];
            const size_t k = 2*(size_t) il + (size_t) kind;
            GGML_ASSERT(il >= 0 && (kind == 0 || kind == 1) && k < mo_expect.size());
            mo_expect[k]++;
            mo_n[k]    = p[6];
            mo_ntok[k] = (int32_t) node->src[0]->ne[1];
            (kind == 0 ? mo_st.pub_ids : mo_st.pub_pred)++;
        }
    }

    // wait until slot k holds its newest launched publish (M6); nullptr (and the switch disabled, C24 fix b) when the
    // device is ahead of the launches the host counted; bounded: 2 s, then abort.
    // C24 poll policy: pause-spin (no yield) for the first 2 ms: in C23 a yield after 50 us could hand the core to another
    // ready thread for a whole time slice, which is one explanation of the 2-3 ms maximum polls; the draft does not run
    // during a verify round, so the spinning thread competes with nothing it waits for. From 100 us on, one non-blocking
    // event query every 100 us (a WDDM flush, in case submitted work is held in the driver queue); polls that end within
    // 20 us after such a query are counted (flush_released) so the tester can see whether the flush is what releases them.
    // After 2 ms the thread yields on every poll
    const int32_t * mo_wait(size_t k) {
        volatile const uint32_t * flag = (volatile const uint32_t *) ((const char *) mo_host + k*GGML_AMB_PUBLISH_SLOT);
        const uint32_t target = mo_expect[k];
        uint32_t v = *flag;
        int64_t ns = 0;
        if (v == target) {
            mo_st.ready++;
        } else {
            const clk::time_point t0 = clk::now();
            int64_t last_flush = -1;
            for (uint64_t it = 0;; ++it) {
                v = *flag;
                if (v == target) {
                    break;
                }
                if ((int32_t) (v - target) > 0) {
                    // C24 fix b: e.g. a split whose launch failed after some of its nodes had run
                    char why[160];
                    snprintf(why, sizeof(why), "publish slot %zu holds sequence %u, ahead of the %u launches counted", k, v, target);
                    mo_runtime_disable(why);
                    return nullptr;
                }
                if ((it & 63) == 0) {
                    ns = ns_between(t0, clk::now());
                    if (ns >= 100000ll && (last_flush < 0 || ns - last_flush >= 100000ll) && la_query != nullptr && mo_flush != nullptr) {
                        (void) la_query(mo_flush);
                        mo_st.flushes++;
                        last_flush = ns;
                    }
                    if (ns > 2000000000ll) {
                        // C24 review S2: no abort; the synchronized path takes over (its readback waits for the stream)
                        char why[160];
                        snprintf(why, sizeof(why), "the publish of slot %zu (layer %zu, kind %zu) did not arrive within 2 s "
                            "(seen %u, want %u)", k, k/2, k%2, v, target);
                        mo_runtime_disable(why);
                        return nullptr;
                    }
                    // C25 AMB_MOE_LA_CHUNK_MIB: the next lookahead chunk of this layer while its publish is pending
                    if (ck_on && ck.active) {
                        ck_feed(k);
                    }
                }
                if (ns < 2000000ll) {
                    AMB_CPU_PAUSE();
                } else {
                    std::this_thread::yield();
                }
            }
            ns = ns_between(t0, clk::now());
            if (ck_on) {
                ck_publish_seen(k); // C25: the deadline estimate of the chunked layer
            }
            mo_st.polled++;
            mo_st.poll_ns += ns;
            mo_st.poll_max_ns = std::max(mo_st.poll_max_ns, ns);
            if (last_flush >= 0 && ns - last_flush < 20000ll) {
                mo_st.flush_released++;
            }
        }
        mo_st.poll_hist[ns < 10000 ? 0 : ns < 50000 ? 1 : ns < 200000 ? 2 : ns < 1000000 ? 3 : 4]++;
        // the ids were written before the flag (system-scope fences on the device): read them after the flag
        std::atomic_thread_fence(std::memory_order_acquire);
        mo_taken[k] = target;
        return (const int32_t *) ((const char *) mo_host + k*GGML_AMB_PUBLISH_SLOT) + GGML_AMB_PUBLISH_DATA;
    }

    // AMB_MOE_MISS_OVERLAP=2 (C24)
    bool miss_overlap_split(int32_t il, int64_t n_tokens) const {
        return mo_mode == 2 && miss_overlap(il, n_tokens) && layers[il].slot_map2 != nullptr;
    }

    ggml_tensor * get_slot_map2(int32_t il) const {
        return il >= 0 && il < (int32_t) layers.size() ? layers[il].slot_map2 : nullptr;
    }

    ggml_tensor * miss_overlap_gate(ggml_context * ctx, ggml_tensor * a, int32_t il) const {
        GGML_ASSERT(mo_mode == 2 && il >= 0 && il < (int32_t) layers.size());
        if (!ggml_is_contiguous(a)) {
            a = ggml_cont(ctx, a);
        }
        // no_alloc (--fit) graphs are never computed: no event there
        void * ev    = il < (int32_t) mo2_ev.size() ? (void *) mo2_ev[il] : nullptr;
        void * stamp = mo2_stamps && mo_sdev != nullptr ?
            (void *) ((char *) mo_sdev + (2*layers.size() + (size_t) il)*GGML_AMB_PUBLISH_SLOT) : nullptr;
        ggml_tensor * g = ggml_amb_gate(ctx, a, ev, stamp, il);
        ggml_format_name(g, "amb_gate-%d", il);
        return g;
    }

    // mode 2 with AMB_STEP_EVENTS: the gate-wait GPU time per round since the last call, from the stamp slots
    double mo2_gate_ms(uint64_t & waits) {
        waits = 0;
        if (!mo2_stamps || mo_host == nullptr) {
            return 0.0;
        }
        mo2_stamp_prev.resize(2*layers.size(), 0);
        uint64_t sum = 0;
        for (size_t il = 0; il < layers.size(); ++il) {
            volatile const unsigned long long * st = (volatile const unsigned long long *)
                ((const char *) mo_host + (2*layers.size() + il)*GGML_AMB_PUBLISH_SLOT);
            const uint64_t s1 = st[1], c1 = st[2];
            sum   += s1 - mo2_stamp_prev[2*il];
            waits += c1 - mo2_stamp_prev[2*il + 1];
            mo2_stamp_prev[2*il]     = s1;
            mo2_stamp_prev[2*il + 1] = c1;
        }
        return sum*1e-6;
    }

    // the predicted ids of layer il (published in the tail before the lookup of il - 1) into la_pred; false if none pending
    bool mo_take_pred(int32_t il, int64_t n_tok) {
        if (!mo_pending(il, 1)) {
            mo_st.pred_missing++;
            return false;
        }
        const size_t k = 2*(size_t) il + 1;
        const int32_t n = mo_n[k], ntok = mo_ntok[k];
        const int32_t * src = mo_wait(k);
        if (src == nullptr || ntok < 1 || ntok > n_tok || (int64_t) n != (int64_t) la_k*ntok) {
            mo_st.pred_missing++;
            return false;
        }
        la_pred.assign(src, src + n);
        la_pred_ntok = ntok;
        mo_st.pred_taken++;
        return true;
    }

    void mo_log(const char * what, bool interval) {
        if (!mo_on && mo_st.disabled == 0) {
            return;
        }
        mo_acc d = mo_st;
        if (interval) {
            const mo_acc & p = mo_prev;
            d.pub_ids -= p.pub_ids; d.pub_pred -= p.pub_pred; d.ids_taken -= p.ids_taken; d.pred_taken -= p.pred_taken;
            d.ready -= p.ready; d.polled -= p.polled; d.poll_ns -= p.poll_ns; d.pred_missing -= p.pred_missing;
            d.no_publish -= p.no_publish; d.miss_batches -= p.miss_batches; d.miss_bytes -= p.miss_bytes;
            d.c_waits -= p.c_waits; d.skipped_waits -= p.skipped_waits; d.dbg_cmp -= p.dbg_cmp; d.dbg_bad -= p.dbg_bad;
            d.dbg_pred_cmp -= p.dbg_pred_cmp; d.dbg_pred_bad -= p.dbg_pred_bad; d.poisoned -= p.poisoned;
            d.flushes -= p.flushes;
            mo_prev = mo_st;
        }
        LLAMA_LOG_INFO("llama_moe_cache: amb_miss_overlap: %s: round %lld: publishes launched = %llu ids + %llu predicted, "
            "taken = %llu ids + %llu predicted, flag ready on arrival = %llu, polled = %llu (%.3f ms total, max %.1f us), "
            "predicted missing = %llu, overlap-size copies without a publish = %llu, miss batches on U = %llu (%.2f MiB), "
            "compute waits = %llu, skipped waits = %llu | debug: compared %llu ids lists (%llu mismatches), %llu predicted "
            "(%llu mismatches), poisoned runs = %llu, long-poll flushes = %llu\n",
            what, (long long) la_round, (unsigned long long) d.pub_ids, (unsigned long long) d.pub_pred,
            (unsigned long long) d.ids_taken, (unsigned long long) d.pred_taken, (unsigned long long) d.ready,
            (unsigned long long) d.polled, d.poll_ns*1e-6, mo_st.poll_max_ns*1e-3, (unsigned long long) d.pred_missing,
            (unsigned long long) d.no_publish, (unsigned long long) d.miss_batches, d.miss_bytes/1024.0/1024.0,
            (unsigned long long) d.c_waits, (unsigned long long) d.skipped_waits, (unsigned long long) d.dbg_cmp,
            (unsigned long long) d.dbg_bad, (unsigned long long) d.dbg_pred_cmp, (unsigned long long) d.dbg_pred_bad,
            (unsigned long long) d.poisoned, (unsigned long long) d.flushes);
        // C24: poll shape, runtime disables, mode 2
        for (int b = 0; b < 5 && interval; ++b) {
            d.poll_hist[b] -= mo_prev_c24.poll_hist[b];
        }
        if (interval) {
            d.flush_released -= mo_prev_c24.flush_released; d.disabled -= mo_prev_c24.disabled;
            d.s2_layers -= mo_prev_c24.s2_layers; d.s2_fallback -= mo_prev_c24.s2_fallback;
            d.s2_selected -= mo_prev_c24.s2_selected; d.s2_pending -= mo_prev_c24.s2_pending;
            d.s2_records -= mo_prev_c24.s2_records; d.s2_skipped -= mo_prev_c24.s2_skipped;
            mo_prev_c24 = mo_st;
        }
        uint64_t gate_waits = 0;
        const double gate_ms = mo2_gate_ms(gate_waits);
        const int64_t rounds = la_round - mo_prev_round; // the gate stamps are read as a delta in both cases
        if (interval) {
            mo_prev_round = la_round;
        }
        LLAMA_LOG_INFO("llama_moe_cache: amb_miss_overlap: %s: C24: polls ending <10us %llu, <50us %llu, <200us %llu, <1ms %llu, "
            ">=1ms %llu; polls released within 20 us of a flush query = %llu; runtime disables = %llu%s | mode %d: split "
            "copies = %llu (synchronized fallback %llu), selected experts = %llu, in the miss phase = %llu (%.1f%%), gate "
            "events recorded = %llu, not recorded (negative control) = %llu, gate wait = %.3f ms per round over %llu waits "
            "(%s)\n",
            what, (unsigned long long) d.poll_hist[0], (unsigned long long) d.poll_hist[1], (unsigned long long) d.poll_hist[2],
            (unsigned long long) d.poll_hist[3], (unsigned long long) d.poll_hist[4], (unsigned long long) d.flush_released,
            (unsigned long long) d.disabled, mo_on ? "" : " (now off)", mo_mode, (unsigned long long) d.s2_layers,
            (unsigned long long) d.s2_fallback, (unsigned long long) d.s2_selected, (unsigned long long) d.s2_pending,
            d.s2_selected > 0 ? 100.0*d.s2_pending/d.s2_selected : 0.0, (unsigned long long) d.s2_records,
            (unsigned long long) d.s2_skipped, rounds > 0 ? gate_ms/rounds : 0.0, (unsigned long long) gate_waits,
            mo2_stamps ? "stamps on" : "stamps off: set AMB_STEP_EVENTS");
        // C25: only when one of its switches is set (the default log is unchanged)
        c25_log(what, interval);
    }

    // the drain reasons are string literals, compared by content
    void la_count_drain(const char * reason) {
        for (auto & r : la_drains) {
            if (strcmp(r.first, reason) == 0) {
                r.second++;
                return;
            }
        }
        la_drains.push_back({ reason, 1 });
    }

    // I6: U finished, nothing in flight, no prefetched-unused flags (they become ordinary cached experts)
    void lookahead_drain(const char * reason) {
        if (!la_on) {
            return;
        }
        bool any = false;
        // C25 AMB_MOE_LA_CHUNK_MIB: the un-issued candidates of a chunked lookahead are forgotten (K2)
        ck_abandon();
        // DEBUG_DEFER: the pending copies first, so that the synchronize below completes them
        while (!la_defer_q.empty()) {
            la_flush_front();
            la_defer_fl_drain++;
        }
        for (device & d : devices) {
            if (d.upload != nullptr && d.la_last > d.la_done) {
                ggml_backend_synchronize(d.upload);
                d.la_done = d.la_last;
                if (d.miss == nullptr) {
                    d.mo_dirty = false; // the miss uploads were on U (C25: with M they are synchronized below)
                }
                any = true;
            }
            // AMB_MOE_MISS_OVERLAP (M5): miss uploads on U complete before any bank write/move/park that follows a drain
            // (not counted as a drain reason: with the switch unset mo_dirty is never set)
            // C25 AMB_MOE_MISS_STREAM (S3): on M when it exists
            if (d.mo_dirty && (d.miss != nullptr || d.upload != nullptr)) {
                ggml_backend_synchronize(d.miss != nullptr ? d.miss : d.upload);
                d.mo_dirty = false;
            }
        }
        for (group & g : groups) {
            if (!g.la_ok) {
                continue;
            }
            moe_cache_lru & lru = g.lru;
            lru.pf_done = devices[g.id].la_done;
            for (int32_t s : lru.la_unused) {
                if (lru.pf_unused[s]) {
                    lru.pf_unused[s] = 0;
                    la_unused_at_drain++;
                    any = true;
                }
            }
            lru.la_unused.clear();
        }
        if (any) {
            la_count_drain(reason);
        }
    }

    // I5: the newest complete batch by event query (non-blocking); U is in order
    void lookahead_ack(int32_t id) {
        device & d = devices[id];
        if (d.upload == nullptr || d.la_last <= d.la_done) {
            return;
        }
        // C18c: only batches whose event record is enqueued (la_flushed == la_last without DEBUG_DEFER)
        for (uint64_t k = d.la_flushed; k > d.la_done && k + (uint64_t) d.la_n_ev > d.la_last; --k) {
            const int q = la_query(d.la_ev[k % (uint64_t) d.la_n_ev]);
            if (q < 0) {
                // C18b (review m1): an error is not "complete"
                GGML_ABORT("amb_moe_lookahead: CUDA error while querying a lookahead upload event");
            }
            if (q > 0) {
                d.la_done = k;
                break;
            }
        }
        for (group & g : groups) {
            if (g.la_ok && g.id == id) {
                g.lru.pf_done = d.la_done;
            }
        }
    }

    // the copies of a batch (batched call, else one copy per run) on the upload stream of d
    void la_enqueue_copies(device & d, void ** dsts, const void ** srcs, const size_t * sizes, ggml_tensor ** banks, size_t n) {
        bool done = false;
        if (batch_fn != nullptr && n >= 2) {
            done = batch_fn(d.upload, dsts, srcs, sizes, n);
        }
        if (!done) {
            for (size_t k = 0; k < n; ++k) {
                ggml_tensor * bank = banks[k];
                ggml_backend_tensor_set_async(d.upload, bank, srcs[k], (size_t) ((const char *) dsts[k] - (const char *) bank->data), sizes[k]);
            }
        }
    }

    // DEBUG_DEFER: enqueue the copies and the event record of the oldest pending batch (U is in order, so in seq order)
    void la_flush_front() {
        la_defer_t & b = la_defer_q.front();
        device & d = devices[b.dev];
        la_enqueue_copies(d, b.dsts.data(), b.srcs.data(), b.sizes.data(), b.banks.data(), b.dsts.size());
        ggml_backend_event_record(d.la_ev[b.seq % (uint64_t) d.la_n_ev], d.upload);
        d.la_flushed = b.seq;
        const int te = la_tev(d.upload);
        if (la_probe && te >= 0) {
            la_probe_end.push_back({ b.seq, te });
        }
        la_defer_q.pop_front();
    }

    // AMB_STEP_EVENTS: a timing event recorded on the stream of backend, its index, -1 if none
    int la_tev(ggml_backend_t backend) {
        if (!la_tev_active) {
            return -1;
        }
        if (la_tev_used == la_tev_pool.size()) {
            void * ev = la_tev_new(devices[0].backend);
            if (ev == nullptr) {
                return -1;
            }
            la_tev_pool.push_back(ev);
        }
        if (!la_tev_record(la_tev_pool[la_tev_used], backend)) {
            return -1;
        }
        return (int) la_tev_used++;
    }

    // the timing events of the last timed round (complete by now, else the round is dropped)
    void lookahead_timing_read() {
        if (!la_tev_active) {
            return;
        }
        la_tev_active = false;
        double up = 0.0, wt = 0.0;
        bool ok = true;
        float ms = 0.0f;
        for (const auto & pr : la_tev_up) {
            if (pr.first < 0 || pr.second < 0) {
                continue;
            }
            if (!la_tev_elapsed(la_tev_pool[pr.first], la_tev_pool[pr.second], &ms)) {
                ok = false;
                break;
            }
            up += ms;
        }
        for (const auto & pr : la_tev_wait) {
            if (!ok || pr.first < 0 || pr.second < 0) {
                continue;
            }
            if (!la_tev_elapsed(la_tev_pool[pr.first], la_tev_pool[pr.second], &ms)) {
                ok = false;
                break;
            }
            wt += ms;
        }
        // ORDER_PROBE: signed times of the C events relative to the end of the needed batch on U
        la_acc pr;
        for (const la_probe_t & p : la_probe_q) {
            if (!ok) {
                break;
            }
            float d0 = 0.0f, d1 = 0.0f;
            if (!la_tev_elapsed(la_tev_pool[p.te], la_tev_pool[p.c0], &d0) || !la_tev_elapsed(la_tev_pool[p.te], la_tev_pool[p.c1], &d1)) {
                ok = false;
                break;
            }
            pr.pr_n++;
            pr.pr_c0_after += d0 >= 0.0f ? 1 : 0;
            pr.pr_c1_after += d1 >= 0.0f ? 1 : 0;
            pr.pr_c1_near  += d1 >= 0.0f && d1 < 0.1f ? 1 : 0;
            pr.pr_d0_ms += d0;
            pr.pr_d1_ms += d1;
        }
        la_acc & a = la_st[la_tev_cls];
        if (ok) {
            a.timed++;
            a.up_ms   += up;
            a.wait_ms += wt;
            a.pr_n        += pr.pr_n;
            a.pr_c0_after += pr.pr_c0_after;
            a.pr_c1_after += pr.pr_c1_after;
            a.pr_c1_near  += pr.pr_c1_near;
            a.pr_d0_ms    += pr.pr_d0_ms;
            a.pr_d1_ms    += pr.pr_d1_ms;
        } else {
            a.timed_lost++;
        }
        la_probe_q.clear();
        la_probe_end.clear();
        la_tev_up.clear();
        la_tev_wait.clear();
        la_tev_used = 0;
    }

    // plan and issue the lookahead of layer il (predicted in la_pred) after the plan of il_real = il - 1 on backend
    void lookahead_issue(ggml_backend_t backend, int32_t il_real, int32_t il, int64_t n_tok, bool record_c) {
        const clk::time_point t0 = clk::now();
        layer & l = layers[il];
        group & g = groups[l.ig];
        device & d = devices[g.id];
        moe_cache_lru & lru = g.lru;
        la_acc & a = la_st[n_tok == 1 ? 0 : 1];
        const int32_t n_expert = (int32_t) l.slot_map->ne[1];
        GGML_UNUSED(d); // C25: the GPU part moved to la_issue_fills

        if ((int64_t) la_pred.size() != la_k*la_pred_ntok || la_pred_ntok > n_tok) {
            a.bad++;
            return;
        }
        // distinct predicted experts, the ones predicted for more tokens first, then (RANK: the ids of each token are in
        // descending router score) by the best rank over the tokens, then by expert index (deterministic)
        la_count.assign(n_expert, 0);
        la_best.assign(n_expert, 0xFF);
        la_cand.clear();
        for (size_t i = 0; i < la_pred.size(); ++i) {
            const int32_t e = la_pred[i];
            if (e < 0 || e >= n_expert) {
                a.bad++;
                return;
            }
            if (la_count[e]++ == 0) {
                la_cand.push_back(e);
            }
            la_best[e] = std::min<uint8_t>(la_best[e], (uint8_t) (i % (size_t) la_k));
        }
        if (la_rank) {
            std::sort(la_cand.begin(), la_cand.end(), [&](int32_t x, int32_t y) {
                if (la_count[x] != la_count[y]) {
                    return la_count[x] > la_count[y];
                }
                return la_best[x] != la_best[y] ? la_best[x] < la_best[y] : x < y;
            });
        } else {
            std::sort(la_cand.begin(), la_cand.end(), [&](int32_t x, int32_t y) {
                return la_count[x] != la_count[y] ? la_count[x] > la_count[y] : x < y;
            });
        }
        a.evals++;
        a.predicted += la_cand.size();
        la_pred_round[il] = la_round;

        size_t expert_bytes = 0;
        for (const ggml_tensor * w : l.experts) {
            expert_bytes += w->nb[2];
        }
        const size_t budget = n_tok == 1 ? la_budget_1 : la_budget_n;
        const size_t max_n  = expert_bytes > 0 ? budget/expert_bytes : 0;

        if (++lru.la_mark_gen == 0) {
            std::fill(lru.la_mark.begin(), lru.la_mark.end(), 0);
            lru.la_mark_gen = 1;
        }
        const int32_t * slots = lru.slot_map[il];
        // C18b (review m3, PROTECT=1): the predicted experts that are already cached are not victims of this batch
        if (la_protect) {
            for (int32_t e : la_cand) {
                if (slots[e] < lru.n_slots) {
                    lru.la_mark[slots[e]] = lru.la_mark_gen;
                }
            }
        }
        // C25 AMB_MOE_LA_CHUNK_MIB: the same candidates in the same order and budget, admitted and uploaded chunk by chunk
        // (the first now, the next ones by ck_feed during the poll for the publish of layer il); K1..K3
        if (ck_on && !la_dry) {
            ck_abandon();
            ck.cand         = la_cand;
            ck.pos          = 0;
            ck.il           = il;
            ck.il_real      = il_real;
            ck.n_tok        = n_tok;
            ck.round        = la_round;
            ck.ig           = l.ig;
            ck.max_n        = max_n;
            ck.n_issued     = 0;
            ck.expert_bytes = expert_bytes;
            ck.n_chunk      = expert_bytes > 0 ? std::max<size_t>(1, ck_bytes/expert_bytes) : 1;
            ck.t_chunk_ns   = (int64_t) ((double) (ck.n_chunk*expert_bytes)/ck_gbps); // bytes / (GB/s) = ns
            ck.t_iss        = t0;
            ck.stopped      = false;
            ck.active       = true;
            ck.timing       = true;
            c25_st.ck_layers++;
            ck_issue_chunk(backend, record_c, true, t0); // a.ns_issue from the start of lookahead_issue, as before
            return;
        }
        la_fills.clear();
        const uint64_t la_nv0 = a.no_victim, la_cp0 = a.capped; // C21 AMB_MOE_LOOKAHEAD_SMALL counters
        for (int32_t e : la_cand) {
            if (slots[e] < lru.n_slots) {
                a.cached++; // cached, or in flight from an earlier batch
                continue;
            }
            if (la_fills.size() >= max_n) {
                a.capped++;
                continue;
            }
            // I2: not selected by plan(il_real), not in flight, not recently used, not chosen for this batch
            const int32_t s = lru.victim(il_real, true);
            if (s < 0) {
                a.no_victim++;
                continue;
            }
            lru.la_mark[s] = lru.la_mark_gen;
            la_fills.push_back({ e, s });
        }
        if (g.la_small) {
            g.la_small_no_victim += a.no_victim - la_nv0;
            g.la_small_capped    += a.capped - la_cp0;
        }

        if (la_dry) {
            // nothing changes: remember what would have been uploaded for the useful count in copy() of layer il
            std::vector<int32_t> & ds = la_dry_set[il];
            ds.clear();
            for (const auto & f : la_fills) {
                ds.push_back(f.expert);
            }
            a.issued += la_fills.size();
            a.bytes  += la_fills.size()*expert_bytes;
            a.ns_issue += ns_between(t0, clk::now());
            return;
        }
        if (la_fills.empty()) {
            a.ns_issue += ns_between(t0, clk::now());
            return;
        }
        la_issue_fills(backend, record_c, il, n_tok, t0);
    }

    // the GPU part of a lookahead batch of layer il, la_fills planned by the caller: copy list, admission, ordering, copies,
    // event record, counters (a.ns_issue from t0). C25: the code below is the unchanged tail of lookahead_issue, moved into
    // its own function so that the chunks of AMB_MOE_LA_CHUNK_MIB use it as well (one batch = one seq per call)
    void la_issue_fills(ggml_backend_t backend, bool record_c, int32_t il, int64_t n_tok, clk::time_point t0) {
        layer & l = layers[il];
        group & g = groups[l.ig];
        device & d = devices[g.id];
        moe_cache_lru & lru = g.lru;
        la_acc & a = la_st[n_tok == 1 ? 0 : 1];

        // C18b (review m2): the copy list is built first (no state change yet); admission and the GPU calls follow, and an
        // exception between them finishes U and empties the admitted slots, so no slot can later be taken as complete
        // without its data
        std::sort(la_fills.begin(), la_fills.end(), [](const moe_cache_lru::fill & x, const moe_cache_lru::fill & y) { return x.expert < y.expert; });
        la_dsts.clear();
        la_srcs.clear();
        la_sizes.clear();
        la_banks.clear();
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t es = w->nb[2];
            for (size_t i = 0; i < la_fills.size();) {
                size_t n = 1;
                while (i + n < la_fills.size() && la_fills[i + n].expert == la_fills[i].expert + (int32_t) n &&
                       la_fills[i + n].slot == la_fills[i].slot + (int32_t) n) {
                    n++;
                }
                la_dsts.push_back((char *) bank->data + (size_t) la_fills[i].slot*es);
                la_srcs.push_back((const uint8_t *) w->data + (size_t) la_fills[i].expert*es);
                la_sizes.push_back(n*es);
                la_banks.push_back(bank);
                bytes += n*es;
                i += n;
            }
        }
        lru.la_unused.reserve(lru.la_unused.size() + la_fills.size());

        const uint64_t seq = d.la_last + 1;
        size_t n_admitted = 0;
        try {
            // admission (host slot maps and LRU, as a real fill), all slots of the batch are in flight from now on
            for (const auto & f : la_fills) {
                lru.pf_admit(il, f.expert, f.slot, seq);
                n_admitted++;
            }
            if (lru.la_unused.size() > 4096) {
                // drop stale entries
                size_t k = 0;
                for (int32_t s : lru.la_unused) {
                    if (lru.pf_unused[s]) {
                        lru.la_unused[k++] = s;
                    }
                }
                lru.la_unused.resize(k);
            }

            // ordering (LINK_WAIT, default on): U after the blocking copies and the slot map upload of copy(il_real), so the
            // link serves them first; la_ev_c is recorded here, or already in copy() when the issue is deferred (ISSUE=post).
            // Not needed for safety: U writes only victims, which C does not touch before copy(il) (I1, I2)
            if (la_link_wait) {
                if (record_c) {
                    ggml_backend_event_record(d.la_ev_c, backend);
                }
                ggml_backend_event_wait(d.upload, d.la_ev_c);
            }
            const int tb = la_tev(d.upload);

            if (la_debug) {
                // debug: poison the victims, then hold U back, so a missing wait of C shows up as NaN / id-0 tokens
                for (size_t k = 0; k < la_dsts.size(); ++k) {
                    if (!la_memset(d.upload, la_dsts[k], la_sizes[k], 0xFF, la_sizes[k], 1)) {
                        GGML_ABORT("amb_moe_lookahead: AMB_MOE_LOOKAHEAD_DEBUG: poisoning a victim slot failed");
                    }
                }
                if (la_spin_ns > 0 && !la_spin(d.upload, la_spin_ns)) {
                    GGML_ABORT("amb_moe_lookahead: AMB_MOE_LOOKAHEAD_DEBUG: the delay kernel failed");
                }
            }
            if (la_defer) {
                // DEBUG_DEFER: the copies and the event record wait on the host (la_flush_front); no up-time pair
                la_defer_t b;
                b.seq   = seq;
                b.dev   = g.id;
                b.call  = la_copy_calls;
                b.dsts  = la_dsts;
                b.srcs  = la_srcs;
                b.sizes = la_sizes;
                b.banks = la_banks;
                la_defer_q.push_back(std::move(b));
                d.la_last = seq;
                la_deferred++;
            } else {
                la_enqueue_copies(d, la_dsts.data(), la_srcs.data(), la_sizes.data(), la_banks.data(), la_dsts.size());
                ggml_backend_event_record(d.la_ev[seq % (uint64_t) d.la_n_ev], d.upload);
                d.la_last    = seq;
                d.la_flushed = seq;
                const int te = la_tev(d.upload);
                if (tb >= 0 && te >= 0) {
                    la_tev_up.push_back({ tb, te });
                }
                if (la_probe && te >= 0) {
                    la_probe_end.push_back({ seq, te });
                }
            }
        } catch (...) {
            while (!la_defer_q.empty()) { // DEBUG_DEFER: earlier batches are complete after the synchronize
                la_flush_front();
                la_defer_fl_drain++;
            }
            ggml_backend_synchronize(d.upload);
            for (size_t k = 0; k < n_admitted; ++k) {
                lru.la_revert(la_fills[k].slot);
            }
            if (d.la_last < seq) {
                d.la_done = d.la_last; // nothing of this batch is recorded; U is finished
                lru.pf_done = d.la_done;
            }
            throw;
        }

        a.issued  += la_fills.size();
        a.bytes   += bytes;
        a.batches += 1;
        if (g.la_small) {
            g.la_small_issued += la_fills.size();
        }
        a.ns_issue += ns_between(t0, clk::now());
    }

    //
    // C25 AMB_MOE_LA_CHUNK_MIB (see the member comment, K1..K3)
    //

    // the chunked lookahead will not reach its copy(): its un-issued candidates are forgotten (K2: nothing to undo)
    void ck_abandon() {
        if (ck.active) {
            c25_st.ck_abandoned++;
        }
        ck.active = false;
        ck.timing = false;
    }

    // plan the next chunk from ck.cand[ck.pos..] with the victim rule of lookahead_issue (I2, K1) and issue it as a batch
    void ck_issue_chunk(ggml_backend_t backend, bool record_c, bool first, clk::time_point t0 = clk::now()) {
        layer & l = layers[ck.il];
        group & g = groups[ck.ig];
        moe_cache_lru & lru = g.lru;
        la_acc & a = la_st[ck.n_tok == 1 ? 0 : 1];
        const int32_t * slots = lru.slot_map[ck.il];
        GGML_ASSERT(l.ig == ck.ig);
        const uint64_t la_nv0 = a.no_victim, la_cp0 = a.capped;
        la_fills.clear();
        while (ck.pos < ck.cand.size() && la_fills.size() < ck.n_chunk) {
            const int32_t e = ck.cand[ck.pos++];
            if (slots[e] < lru.n_slots) {
                a.cached++; // cached, or in flight from an earlier batch (or chunk)
                continue;
            }
            if (ck.n_issued + la_fills.size() >= ck.max_n) {
                a.capped++;
                continue;
            }
            const int32_t s = lru.victim(ck.il_real, true);
            if (s < 0) {
                a.no_victim++;
                continue;
            }
            lru.la_mark[s] = lru.la_mark_gen;
            la_fills.push_back({ e, s });
        }
        if (ck.n_issued + la_fills.size() >= ck.max_n) {
            // the budget is used up: the rest are counted as lookahead_issue counts them, nothing is left to feed
            for (; ck.pos < ck.cand.size(); ++ck.pos) {
                (slots[ck.cand[ck.pos]] < lru.n_slots ? a.cached : a.capped)++;
            }
        }
        if (g.la_small) {
            g.la_small_no_victim += a.no_victim - la_nv0;
            g.la_small_capped    += a.capped - la_cp0;
        }
        if (ck.pos >= ck.cand.size()) {
            ck.active = false; // ck.timing stays: the publish time of this layer is still measured
        }
        if (la_fills.empty()) {
            a.ns_issue += ns_between(t0, clk::now());
            return;
        }
        const size_t n = la_fills.size();
        la_issue_fills(backend, record_c, ck.il, ck.n_tok, t0);
        ck.n_issued += n;
        if (first) {
            c25_st.ck_first++;
        } else {
            c25_st.ck_fed++;
            c25_st.ck_fed_bytes += n*ck.expert_bytes;
        }
    }

    // called by mo_wait(k) while it polls: the next chunk if the publish k is the one of the chunked layer, U has nothing
    // of this lookahead left (one chunk at a time, so a miss waits behind at most one), and (DEADLINE) the chunk is
    // predicted to finish before the publish
    void ck_feed(size_t k) {
        if (!ck.active || ck.stopped || k != 2*(size_t) ck.il || ck.round != la_round) {
            return;
        }
        if (ck_deadline) {
            const int64_t d_est = ck_d_ns[ck.il];
            const int64_t el    = ns_between(ck.t_iss, clk::now());
            if (d_est <= 0 || el + ck.t_chunk_ns + ck_slack_ns > d_est) {
                ck.stopped = true; // the elapsed time only grows: no later chunk of this layer would pass either
                c25_st.ck_stop_deadline++;
                return;
            }
        }
        device & d = devices[groups[ck.ig].id];
        if (d.la_last > d.la_done) {
            lookahead_ack(groups[ck.ig].id);
            c25_st.ck_queries++;
            if (d.la_last > d.la_done) {
                return;
            }
        }
        ck_issue_chunk(d.backend, false, false);
    }

    // mo_wait(k) saw the publish after polling: the host time from the first chunk to it, for the deadline of the layer
    void ck_publish_seen(size_t k) {
        if (!ck.timing || k != 2*(size_t) ck.il || ck.round != la_round) {
            return;
        }
        const int64_t obs = ns_between(ck.t_iss, clk::now());
        int64_t & e = ck_d_ns[ck.il];
        e = e <= 0 ? obs : e + (obs - e)/8;
    }

    // copy(il) has its ids (before the plan): the chunked lookahead of il ends; the candidates it did not upload (within
    // the budget) are kept for ck_count_left after the plan. Any other chunked lookahead is abandoned
    void ck_close(int32_t il) {
        ck_left.clear();
        ck_left_il = -1;
        if (!ck.timing) {
            return;
        }
        if (ck.il != il || ck.round != la_round) {
            ck_abandon();
            return;
        }
        if (ck.active) {
            const moe_cache_lru & lru = groups[ck.ig].lru;
            const int32_t * slots = lru.slot_map[il];
            const size_t cap = ck.max_n > ck.n_issued ? ck.max_n - ck.n_issued : 0;
            for (size_t i = ck.pos; i < ck.cand.size() && ck_left.size() < cap; ++i) {
                if (slots[ck.cand[i]] >= lru.n_slots) {
                    ck_left.push_back(ck.cand[i]);
                }
            }
            ck_left_il = il;
        }
        ck.active = false;
        ck.timing = false;
    }

    // after plan(il): which of the candidates left were selected (now misses of this plan, "promoted") or not ("dropped")
    void ck_count_left(int32_t il, const moe_cache_lru & lru) {
        if (ck_left_il != il) {
            return;
        }
        size_t expert_bytes = 0;
        for (const ggml_tensor * w : layers[il].experts) {
            expert_bytes += w->nb[2];
        }
        for (int32_t e : ck_left) {
            c25_st.ck_left++;
            if (lru.seen[e] == lru.seen_gen) {
                c25_st.ck_promoted++;
            } else {
                c25_st.ck_dropped++;
                c25_st.ck_saved_bytes += expert_bytes;
            }
        }
        ck_left.clear();
        ck_left_il = -1;
    }

    void c25_log(const char * what, bool interval) {
        if (!ck_on && !ms_req && !c25_stats && ck_mib <= 0.0) {
            return;
        }
        c25_acc t = c25_st;
        if (interval) {
            const c25_acc & p = c25_prev;
            t.ck_layers -= p.ck_layers; t.ck_first -= p.ck_first; t.ck_fed -= p.ck_fed; t.ck_fed_bytes -= p.ck_fed_bytes;
            t.ck_stop_deadline -= p.ck_stop_deadline; t.ck_queries -= p.ck_queries; t.ck_left -= p.ck_left;
            t.ck_promoted -= p.ck_promoted; t.ck_dropped -= p.ck_dropped; t.ck_saved_bytes -= p.ck_saved_bytes;
            t.ck_abandoned -= p.ck_abandoned; t.ms_batches -= p.ms_batches; t.ms_bytes -= p.ms_bytes;
            t.ms_round_syncs -= p.ms_round_syncs; t.st_miss -= p.st_miss; t.st_behind -= p.st_behind;
            c25_prev = c25_st;
        }
        double d_sum = 0.0;
        int    d_n   = 0;
        for (int64_t v : ck_d_ns) {
            if (v > 0) {
                d_sum += (double) v;
                d_n++;
            }
        }
        bool ms_on = false;
        for (const device & d : devices) {
            ms_on = ms_on || d.miss != nullptr;
        }
        LLAMA_LOG_INFO("llama_moe_cache: amb_c25: %s: round %lld: LA_CHUNK %s (%.1f MiB, deadline %s, %.1f GB/s, slack %lld us): "
            "chunked lookaheads = %llu, chunks first = %llu, fed during the poll = %llu (%.2f MiB), feeding stopped by the "
            "deadline = %llu, feeder queries = %llu, candidates left at the publish = %llu (promoted to misses %llu, dropped "
            "%llu = %.2f MiB not sent), abandoned = %llu, mean publish deadline = %.1f us over %d layers | MISS_STREAM %s: "
            "miss batches on M = %llu (%.2f MiB), M round-start syncs = %llu | C25_STATS %s: miss batches checked = %llu, "
            "issued while the newest lookahead batch was on U = %llu (%.1f%%)\n",
            what, (long long) la_round, ck_on ? "on" : (ck_mib > 0.0 ? "refused" : "off"), ck_bytes/1024.0/1024.0,
            ck_deadline ? "on" : "off", ck_gbps, (long long) (ck_slack_ns/1000),
            (unsigned long long) t.ck_layers, (unsigned long long) t.ck_first, (unsigned long long) t.ck_fed,
            t.ck_fed_bytes/1024.0/1024.0, (unsigned long long) t.ck_stop_deadline, (unsigned long long) t.ck_queries,
            (unsigned long long) t.ck_left, (unsigned long long) t.ck_promoted, (unsigned long long) t.ck_dropped,
            t.ck_saved_bytes/1024.0/1024.0, (unsigned long long) t.ck_abandoned, d_n > 0 ? d_sum/d_n*1e-3 : 0.0, d_n,
            ms_on ? "on" : (ms_req ? "refused" : "off"), (unsigned long long) t.ms_batches, t.ms_bytes/1024.0/1024.0,
            (unsigned long long) t.ms_round_syncs, c25_stats ? "on" : "off", (unsigned long long) t.st_miss,
            (unsigned long long) t.st_behind, t.st_miss > 0 ? 100.0*t.st_behind/t.st_miss : 0.0);
    }

    // one line per class, for the interval since the previous periodic line or cumulative
    void log_lookahead(const char * what, bool interval) {
        for (int c = 0; c < 2; ++c) {
            la_acc t = la_st[c];
            if (interval) {
                const la_acc & p = la_prev[c];
                t.rounds -= p.rounds; t.evals -= p.evals; t.predicted -= p.predicted; t.cached -= p.cached;
                t.issued -= p.issued; t.bytes -= p.bytes; t.batches -= p.batches; t.capped -= p.capped;
                t.no_victim -= p.no_victim; t.bad -= p.bad; t.no_readback -= p.no_readback; t.useful_done -= p.useful_done; t.useful_wait -= p.useful_wait;
                t.dry_useful -= p.dry_useful; t.blocking -= p.blocking; t.covered -= p.covered; t.waits -= p.waits;
                t.ns_issue -= p.ns_issue; t.ns_other -= p.ns_other; t.timed -= p.timed; t.timed_lost -= p.timed_lost;
                t.up_ms -= p.up_ms; t.wait_ms -= p.wait_ms;
                t.pr_n -= p.pr_n; t.pr_c0_after -= p.pr_c0_after; t.pr_c1_after -= p.pr_c1_after; t.pr_c1_near -= p.pr_c1_near;
                t.pr_d0_ms -= p.pr_d0_ms; t.pr_d1_ms -= p.pr_d1_ms;
            }
            if (t.rounds == 0) {
                continue;
            }
            const double r = (double) t.rounds;
            const uint64_t useful = la_dry ? t.dry_useful : t.useful_done + t.useful_wait;
            // share of the misses of the covered layers caught: useful / (useful + blocking misses left there); in DRY nothing
            // is caught for real, the covered blocking misses are the would-be misses
            const double denom = la_dry ? (double) t.covered : (double) (useful + t.covered);
            char timing[160] = "";
            if (la_tev_every > 0) {
                snprintf(timing, sizeof(timing), ", GPU ms/round: upload stream busy = %.3f, compute waiting on uploads = %.3f "
                    "(timed rounds = %llu, dropped = %llu)", t.timed > 0 ? t.up_ms/t.timed : 0.0, t.timed > 0 ? t.wait_ms/t.timed : 0.0,
                    (unsigned long long) t.timed, (unsigned long long) t.timed_lost);
            }
            LLAMA_LOG_INFO("llama_moe_cache: amb_moe_lookahead: round %lld: %s: %s: P%d k=%d%s: rounds = %llu, layer-evals = %llu, "
                "predicted = %llu, already cached = %llu, %s = %llu experts / %.2f MiB / %llu batches, budget-capped = %llu, "
                "no victim = %llu, bad = %llu, no readback = %llu, useful = %llu (complete %llu, waited-on %llu), useful share of covered misses = %.1f%%, "
                "compute waits = %llu, host ms/round issuing = %.3f (+ %.3f queries/waits) | per round: blocking misses = %.2f "
                "(in covered layers %.2f), prefetched useful = %.2f, issued = %.2f, experts on the link = %.2f (blocking + issued)%s\n",
                (long long) la_round, what, c == 0 ? "ubatch=1" : "ubatch=2..max", la_req, la_k, la_dry ? " DRY" : "",
                (unsigned long long) t.rounds, (unsigned long long) t.evals, (unsigned long long) t.predicted,
                (unsigned long long) t.cached, la_dry ? "would issue" : "issued", (unsigned long long) t.issued,
                t.bytes/1024.0/1024.0, (unsigned long long) t.batches, (unsigned long long) t.capped,
                (unsigned long long) t.no_victim, (unsigned long long) t.bad, (unsigned long long) t.no_readback,
                (unsigned long long) useful,
                (unsigned long long) (la_dry ? t.dry_useful : t.useful_done), (unsigned long long) (la_dry ? 0 : t.useful_wait),
                denom > 0 ? 100.0*useful/denom : 0.0, (unsigned long long) t.waits, t.ns_issue*1e-6/r, t.ns_other*1e-6/r,
                t.blocking/r, t.covered/r, useful/r, t.issued/r, la_dry ? t.blocking/r : (t.blocking + t.issued)/r, timing);
            if (la_probe) {
                // C18c: signed GPU times of the compute stream relative to the end of the batch an I4 copy() needs on U.
                // c0 = before the blocking copies (C idle), c1 = after the slot map upload. With SKIP_WAIT: c0 after U end
                // -> the whole compute stream queues behind U; c0 before but c1 after (near) U end -> C's H2D copies queue
                // behind U's copies; both before -> C is not ordered behind U on the GPU (look at the host)
                const double pn = t.pr_n > 0 ? (double) t.pr_n : 1.0;
                LLAMA_LOG_INFO("llama_moe_cache: amb_moe_lookahead: round %lld: %s: %s: order probe: I4 copies probed = %llu, "
                    "C idle point after U batch end = %llu, C slot map upload done after U batch end = %llu (within 0.1 ms: %llu), "
                    "mean (C idle point - U end) = %.3f ms, mean (C slot map done - U end) = %.3f ms, unmatched (cumulative) = %llu\n",
                    (long long) la_round, what, c == 0 ? "ubatch=1" : "ubatch=2..max", (unsigned long long) t.pr_n,
                    (unsigned long long) t.pr_c0_after, (unsigned long long) t.pr_c1_after, (unsigned long long) t.pr_c1_near,
                    t.pr_d0_ms/pn, t.pr_d1_ms/pn, (unsigned long long) la_probe_unmatched);
            }
        }
        if (la_defer) {
            LLAMA_LOG_INFO("llama_moe_cache: amb_moe_lookahead: round %lld: %s: DEBUG_DEFER (cumulative): batches deferred = %llu, "
                "copies enqueued before an I4 wait = %llu, two copy() calls later = %llu, at a drain = %llu, pending now = %zu\n",
                (long long) la_round, what, (unsigned long long) la_deferred, (unsigned long long) la_defer_fl_wait,
                (unsigned long long) la_defer_fl_late, (unsigned long long) la_defer_fl_drain, la_defer_q.size());
        }
        uint64_t wasted = 0, first_victims = 0;
        size_t unused_now = 0;
        for (const group & g : groups) {
            if (!g.la_ok) {
                continue;
            }
            wasted        += g.lru.pf_wasted;
            first_victims += g.lru.la_first_victims;
            // C18b (review m4): over the slots, la_unused can hold a slot twice
            for (uint8_t u : g.lru.pf_unused) {
                unused_now += u;
            }
        }
        std::string drains;
        for (const auto & r : la_drains) {
            char buf[160];
            snprintf(buf, sizeof(buf), "%s%s = %llu", drains.empty() ? "" : ", ", r.first, (unsigned long long) r.second);
            drains += buf;
        }
        // C18b (stats 7): interval values (since the previous periodic line) next to the cumulative ones; per round = over
        // the lookahead rounds of the interval (both classes)
        const uint64_t i_wasted = wasted - la_prev_wasted;
        const uint64_t i_drain  = la_unused_at_drain - la_prev_unused_at_drain;
        const uint64_t i_first  = first_victims - la_prev_first_victims;
        const uint64_t i_rounds = (la_st[0].rounds - la_prev[0].rounds) + (la_st[1].rounds - la_prev[1].rounds);
        const double   ir = i_rounds > 0 ? (double) i_rounds : 1.0;
        const uint64_t dry_wasted = la_dry ? (la_st[0].issued + la_st[1].issued) - (la_st[0].dry_useful + la_st[1].dry_useful) : 0;
        if (interval) {
            LLAMA_LOG_INFO("llama_moe_cache: amb_moe_lookahead: round %lld: %s: interval (%llu rounds): wasted = %llu (%.2f per round; "
                "evicted unused %llu, unused at a drain %llu), real misses placed in a wrong prediction's slot = %llu (%.2f per round)\n",
                (long long) la_round, what, (unsigned long long) i_rounds, (unsigned long long) (i_wasted + i_drain), (i_wasted + i_drain)/ir,
                (unsigned long long) i_wasted, (unsigned long long) i_drain, (unsigned long long) i_first, i_first/ir);
        }
        LLAMA_LOG_INFO("llama_moe_cache: amb_moe_lookahead: round %lld: %s: CUMULATIVE since start: wasted = %llu (evicted unused %llu, "
            "unused at a drain %llu%s), unused now = %zu, real misses placed in a wrong prediction's slot = %llu, "
            "post-hook issues = %llu (dropped %llu), skipped waits (DEBUG_SKIP_WAIT) = %llu, drains: %s\n",
            (long long) la_round, what, (unsigned long long) (la_dry ? dry_wasted : wasted + la_unused_at_drain),
            (unsigned long long) wasted, (unsigned long long) la_unused_at_drain, la_dry ? ", DRY: would-be uploads not useful" : "",
            unused_now, (unsigned long long) first_victims, (unsigned long long) la_post_issued, (unsigned long long) la_post_dropped,
            (unsigned long long) la_skipped_waits, drains.empty() ? "none" : drains.c_str());
        // C21 AMB_MOE_LOOKAHEAD_SMALL: cumulative, per relaxed group (the drains of its real misses included)
        for (const group & g : groups) {
            if (!g.la_small) {
                continue;
            }
            LLAMA_LOG_INFO("llama_moe_cache: amb_moe_lookahead: round %lld: %s: SMALL group (first layer %d, %zu layers, %d own / %d "
                "active slots) CUMULATIVE: issued = %llu experts, no victim = %llu, budget-capped = %llu, real-miss drains = %llu, "
                "wasted = %llu\n", (long long) la_round, what, g.layers.empty() ? -1 : g.layers[0], g.layers.size(), g.n_base,
                g.lru.n_active, (unsigned long long) g.la_small_issued, (unsigned long long) g.la_small_no_victim,
                (unsigned long long) g.la_small_capped, (unsigned long long) g.lru.n_drain_calls, (unsigned long long) g.lru.pf_wasted);
        }
        if (interval) {
            la_prev[0] = la_st[0];
            la_prev[1] = la_st[1];
            la_prev_wasted          = wasted;
            la_prev_unused_at_drain = la_unused_at_drain;
            la_prev_first_victims   = first_victims;
        }
    }

    // AMB_MOE_LOOP_DEDUP_CHECK=1: after plan(ip), every visit il of the physical block ip reads the physical mapping:
    //  D1 lru.slot_map[il] is the array of ip for every visit (so the slot map copy() uploads for il is the current one)
    //  D2 an entry e < n_slots of that array holds key (ip, e) in its slot, and every slot keyed (ip, e) is that entry
    //  D3 no slot holds the key of a non-representative visit (keys are physical)
    void loop_dedup_verify(const group & g, int32_t ip) {
        const moe_cache_lru & lru = g.lru;
        const int32_t * map = lru.slot_map[ip];
        GGML_ASSERT(map != nullptr);
        for (int32_t il : g.layers) {
            if (rep[il] == ip && lru.slot_map[il] != map) {
                GGML_ABORT("amb_loop_dedup: check: layer %d does not share the slot map of its physical layer %d", il, ip);
            }
        }
        for (int32_t e = 0; e < lru.n_expert; ++e) {
            const int32_t s = map[e];
            if (s < lru.n_slots && lru.key_of[s] != lru.make_key(ip, e)) {
                GGML_ABORT("amb_loop_dedup: check: layer %d expert %d maps to slot %d holding key %d", ip, e, s, lru.key_of[s]);
            }
        }
        for (int32_t s = 0; s < lru.n_slots; ++s) {
            const int32_t key = lru.key_of[s];
            if (key < 0) {
                continue;
            }
            const int32_t kl = lru.key_layer(key);
            if (rep[kl] != kl) {
                GGML_ABORT("amb_loop_dedup: check: slot %d holds key %d of the non-physical layer %d", s, key, kl);
            }
            if (kl == ip && map[lru.key_expert(key)] != s) {
                GGML_ABORT("amb_loop_dedup: check: slot %d holds expert %d of layer %d, its map says %d", s, lru.key_expert(key), ip, map[lru.key_expert(key)]);
            }
        }
        loop_dedup_checks++;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il      = it->second.first;
        const bool    has_cpu = it->second.second;
        layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // AMB_MOE_HIT_FIRST: the hit-phase map (no ids, no plan); the miss-phase map takes the whole path below
        if (l.slot_map_h != nullptr && src == l.slot_map_h) {
            return hf_copy_h(backend, il, dst);
        }
        const bool is_r = l.slot_map_r != nullptr && src == l.slot_map_r;
        if (is_r) {
            hf_h_il = -1; // (AMB_IDS_ASYNC off: ids_post_split does not clear it)
            GGML_ASSERT(hf_h_last == il && "amb_moe_hit_first: the miss-phase map without the hit-phase map of the same layer");
            hf_st.r_copies++;
            if (hf_check) {
                // CHECK: the stream is idle after this: the hit phase has run with the device copy of map_H
                ggml_backend_synchronize(backend);
                hf_tmp.resize((size_t) l.slot_map->ne[1]);
                ggml_backend_tensor_get(hf_h_dst, hf_tmp.data(), 0, ggml_nbytes(hf_h_dst));
                hf_st.chk_map++;
                if (memcmp(hf_tmp.data(), hf_stage(il, 0), ggml_nbytes(hf_h_dst)) != 0) {
                    hf_st.chk_map_bad++;
                    LLAMA_LOG_ERROR("%s: amb_moe_hit_first: check: layer %d: the device copy of the hit-phase slot map differs from its staging\n", __func__, il);
                }
            }
        }

        // AMB_MOE_STATS: at the start of a round, so the logged rounds are complete
        if (stats_every > 0 && il == stats_layer) {
            if (stats_rounds > 0 && stats_rounds % stats_every == 0) {
                log_stats_periodic();
            }
            stats_rounds++;
        }

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, unless the lookup is the first node of its split
        const ggml_tensor * sel = lookup->src[1];

        // AMB_MOE_PREFETCH: the predicted ids of layer il + 1, computed in an earlier split on this backend (see qwen4exp)
        const ggml_tensor * pf_t = prefetch_ids(il, sel->src[0]->ne[1]);

        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
            if (pf_t != nullptr && node == pf_t) {
                pf_t = nullptr; // not computed before this split: not safe to read now
                pf.bad++;
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));
        GGML_ASSERT(sel->op == GGML_OP_RESHAPE); // of the [n_expert_used, n_tokens] ids, see build_lookup

        const int32_t n_used = (int32_t) sel->src[0]->ne[0];

        // AMB_MOE_LOOKAHEAD: rounds, drains of larger ubatches, completion queries (all before the plan)
        const int64_t n_tok = sel->src[0]->ne[1];
        const bool la_cls_ok = la_on && n_tok >= la_min_tok && n_tok <= la_max_tok; // counted in the lookahead lines
        int64_t la_ns_other = 0;
        la_have = false;
        // AMB_MOE_MISS_OVERLAP: a publish of these ids was launched, so the scheduler did not synchronize C before this
        // call (copy_nosync answers with the same test; nothing changes it in between): C may still run the tail (M1)
        mo_layer = mo_pending(il, 0);
        bool mo_pred_now = false; // DEBUG: the predicted ids were taken (and compared) here already
        if (mo_on && !mo_layer && !has_cpu && n_tok >= mo_min_tok && n_tok <= mo_max_tok) {
            mo_st.no_publish++;
        }
        if (la_on) {
            const clk::time_point tq = clk::now();
            // DEBUG_DEFER: the batches issued two or more copy() calls ago (C, synchronized by the scheduler before this
            // callback, has finished the layer that read them)
            la_copy_calls++;
            while (!la_defer_q.empty() && la_defer_q.front().call <= la_copy_calls - 2) {
                la_flush_front();
                la_defer_fl_late++;
            }
            if (n_tok > la_drain_above) {
                // I6: MMQ batches (and every larger one) never see an upload in flight or a poisoned slot; C18b: the threshold
                // is the smallest MMVQ batch limit of the cached expert types (<= 8)
                lookahead_drain(n_tok > 8 ? "ubatch > 8 tokens" : "ubatch above the MMVQ batch limit of an expert type");
            }
            // a deferred issue (ISSUE=post) whose split never reported its launch: dropped (nothing was admitted)
            if (la_pend.active) {
                la_pend.active = false;
                la_post_dropped++;
            }
            if (il == la_first_layer) {
                // C25 AMB_MOE_MISS_STREAM (S2): the miss uploads of the previous round on M are complete before any slot
                // they wrote can be a lookahead victim (C waited on each of them; normally this returns at once)
                for (device & dd : devices) {
                    if (dd.miss != nullptr && dd.mo_dirty) {
                        ggml_backend_synchronize(dd.miss);
                        dd.mo_dirty = false;
                        c25_st.ms_round_syncs++;
                    }
                }
                lookahead_timing_read();
                la_round++;
                for (group & gg : groups) {
                    if (gg.la_ok) {
                        gg.lru.cur_round = la_round;
                    }
                }
                if (la_cls_ok) {
                    la_acc & a = la_st[n_tok == 1 ? 0 : 1];
                    a.rounds++;
                    if (la_tev_every > 0 && (int64_t) (a.rounds % (uint64_t) la_tev_every) == 0) {
                        la_tev_active = true;
                        la_tev_cls    = n_tok == 1 ? 0 : 1;
                    }
                }
                if (la_round % la_log_every == 0) {
                    log_lookahead("interval  ", true);
                    mo_log("interval  ", true);
                }
            }
            if (g.la_ok) {
                lookahead_ack(g.id);
            }
            la_ns_other += ns_between(tq, clk::now());
        }
        GGML_ASSERT(!(la_on && n_tok > la_drain_above && devices[g.id].la_last > devices[g.id].la_done) && "lookahead upload in flight in an MMQ batch");

        const bool timed = timing_mode > 0;
        clk::time_point t0, t1, t2, t3, t4, t5, t6;
        if (timed) { t0 = clk::now(); }
        ids.resize(ggml_nelements(sel));
        // AMB_IDS_ASYNC: the readback was enqueued right after the producing split; wait for its event only (it has
        // completed already when the scheduler synchronized the backend before this callback)
        bool ids_done = false;
        const int32_t * pub = nullptr;
        if (mo_layer) {
            const size_t k = 2*(size_t) il;
            if ((size_t) mo_n[k] != ids.size()) {
                GGML_ABORT("amb_miss_overlap: layer %d published %d ids, the lookup reads %zu", il, mo_n[k], ids.size());
            }
            pub = mo_wait(k);
            if (pub == nullptr) {
                // C24 fix b: the switch is off now; the scheduler did not synchronize C before this call, the synchronous
                // readback below does (no async readback was enqueued for this lookup)
                mo_layer = false;
            }
        }
        if (mo_layer) {
            // AMB_MOE_MISS_OVERLAP: the published ids (M1: every C kernel that reads a slot has completed when they are
            // visible); the predicted ids of layer il + 1 follow in the tail, taken at the lookahead issue (mo_take_pred)
            memcpy(ids.data(), pub, ids.size()*sizeof(int32_t));
            ids_done = true;
            mo_st.ids_taken++;
            if (la_on && mo_pending(il + 1, 1)) {
                la_have = true; // filled by mo_take_pred before the issue
            }
            if (mo_cmp) {
                // DEBUG: the old readback of the same tensors (enqueued after the split by ids_post_split), after its event
                const ids_async_state::pending * p = ids_async_take(sel, il);
                if (p != nullptr && p->n == ids.size()) {
                    ggml_backend_event_synchronize(ia.ev);
                    mo_st.dbg_cmp++;
                    if (memcmp(ids.data(), p->dst, ids.size()*sizeof(int32_t)) != 0) {
                        mo_st.dbg_bad++;
                        memcpy(ids.data(), p->dst, ids.size()*sizeof(int32_t)); // the old ids are the reference
                    }
                    if (la_have && p->la_n > 0) {
                        mo_dbg.assign(p->la_dst, p->la_dst + p->la_n);
                        if (mo_take_pred(il + 1, n_tok)) {
                            mo_pred_now = true;
                            mo_st.dbg_pred_cmp++;
                            if (la_pred != mo_dbg || la_pred_ntok != p->la_ntok) {
                                mo_st.dbg_pred_bad++;
                                la_pred      = mo_dbg;
                                la_pred_ntok = p->la_ntok;
                            }
                        } else {
                            la_have = false;
                        }
                    }
                }
            }
        } else if (ia.on) {
            const ids_async_state::pending * p = pf_t == nullptr ? ids_async_take(sel, il) : nullptr;
            if (p != nullptr && p->n == ids.size()) {
                if (is_r && hf_no_wait && hf_on && !hf_sync) {
                    hf_st.no_wait++; // AMB_MOE_HIT_FIRST_DEBUG_NO_WAIT: negative control, the ids below are stale
                } else {
                    ggml_backend_event_synchronize(ia.ev);
                }
                memcpy(ids.data(), p->dst, ids.size()*sizeof(int32_t));
                ids_done = true;
                ia.hits++;
                if (is_r) {
                    hf_st.r_async++;
                    if (hf_check) {
                        // CHECK: the same tensor read back behind a synchronize
                        hf_tmp.resize(ids.size());
                        ggml_backend_tensor_get_async(backend, sel, hf_tmp.data(), 0, ggml_nbytes(sel));
                        ggml_backend_synchronize(backend);
                        hf_st.chk_ids++;
                        if (memcmp(hf_tmp.data(), ids.data(), ids.size()*sizeof(int32_t)) != 0) {
                            hf_st.chk_ids_bad++;
                            LLAMA_LOG_ERROR("%s: amb_moe_hit_first: check: layer %d: the ids read by event differ from the synchronized readback (the latter are used)\n", __func__, il);
                            memcpy(ids.data(), hf_tmp.data(), ids.size()*sizeof(int32_t));
                        }
                    }
                }
                // AMB_MOE_LOOKAHEAD: the predicted ids of layer il + 1, read under the same event
                if (la_on && p->la_n > 0 && p->la_ntok <= n_tok) {
                    la_pred.assign(p->la_dst, p->la_dst + p->la_n);
                    la_pred_ntok = p->la_ntok;
                    la_have = true;
                }
            } else {
                if (pf_t != nullptr) {
                    ia.fb_prefetch++;
                } else if (ids.size() > ia.cap) {
                    ia.fb_size++;
                } else {
                    ia.fb_no_pending++;
                }
                if (!ia.warned) {
                    ia.warned = true;
                    LLAMA_LOG_WARN("%s: amb_ids_async: no async readback for layer %d (%zu ids), reading the ids back as before "
                        "(counted in the amb_ids_async line; warned once)\n", __func__, il, ids.size());
                }
            }
        }
        if (!ids_done && is_r) {
            hf_st.r_sync++;
        }
        if (!ids_done) {
            ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
            if (pf_t != nullptr) {
                // the same readback: no extra synchronization point
                ggml_backend_tensor_get_async(backend, pf_t, pf_pred.data(), 0, pf_pred.size()*sizeof(int32_t));
            }
            ggml_backend_synchronize(backend);
        }
        // C25 AMB_MOE_LA_CHUNK_MIB: the ids of this layer are here, no more chunks for it (K1: before the plan)
        if (ck_on) {
            ck_close(il);
        }
        if (timed) { t1 = clk::now(); }
        if (pf_mode != 0) {
            prefetch_ack(g.id);
        }

        if (trace != nullptr) {
            write_trace(il, n_used);
        }

        // AMB_MOE_PREDICT: the selection of a single token and the cache state before the plan, for predict_add
        if (predict && pred_gen > 0 && sel->src[0]->ne[1] == 1) {
            pred_rec & r = pred_recs[il];
            const int32_t * slots    = g.lru.slot_map[il];
            const int64_t   n_expert = l.slot_map->ne[1];
            r.gen = pred_gen;
            r.real.assign(ids.begin(), ids.end());
            r.cached.resize(n_expert);
            for (int64_t e = 0; e < n_expert; ++e) {
                r.cached[e] = slots[e] < g.n_slots ? 1 : 0;
            }
        }

        // part of the misses is computed on the CPU once the cache is full, the carry keeps the share right on average
        const uint32_t share_q = has_cpu && g.lru.n_filled == g.lru.n_active ? cpu_share_q : 0;
        auto n_cpu = [&](size_t n_miss) {
            const uint64_t acc = (uint64_t) n_miss*share_q + l.cpu_acc;
            l.cpu_acc = acc & 0xffff;
            return (size_t) (acc >> 16);
        };
        size_t n_hit = 0;
        if (timed) { t2 = clk::now(); }
        const uint64_t la_hd0 = g.lru.pf_hit_done, la_hw0 = g.lru.pf_hit_wait;
        // AMB_MOE_LOOP_DEDUP: plan the physical layer (rep[il] == il with the switch off: the old call); the clock of a
        // looped block advances with its first visit only
        const int32_t il_p = rep[il];
        if (!g.lru.plan(il_p, ids.data(), ids.size(), sel->src[0]->ne[1], n_used, n_zero, n_cpu, fills, cpu, n_hit, il_p == il)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d (%d active slots, %d own, %d counted on%s)", il,
                g.lru.n_active, g.n_base, g.n_graph, el_floor_bytes > 0 ? "; AMB_ELASTIC_FLOOR_MIB: the compute floor was not lent" : "");
        }
        if (loop_dedup_check) {
            loop_dedup_verify(g, il_p);
        }
        el_verify(g, "plan");
        if (ck_on) {
            ck_count_left(il, g.lru); // C25: promoted / dropped candidates of the chunked lookahead of this layer
        }
        // AMB_MOE_LOOKAHEAD: what the lookahead planned for this layer in this round achieved
        if (la_cls_ok && g.la_ok) {
            la_acc & a = la_st[n_tok == 1 ? 0 : 1];
            a.blocking += fills.size();
            a.useful_done += g.lru.pf_hit_done - la_hd0;
            a.useful_wait += g.lru.pf_hit_wait - la_hw0;
            if (la_pred_round[il] == la_round) {
                a.covered += fills.size();
                if (la_dry) {
                    const std::vector<int32_t> & ds = la_dry_set[il];
                    for (const auto & f : fills) {
                        a.dry_useful += std::find(ds.begin(), ds.end(), f.expert) != ds.end() ? 1 : 0;
                    }
                }
            }
        }
        if (has_cpu) {
            int32_t * cpu_map = (int32_t *) l.cpu_map->data;
            for (int32_t e : l.cpu_experts) {
                cpu_map[e] = -1;
            }
            l.cpu_experts.clear();
            for (const auto & f : cpu) {
                cpu_map[f.expert] = f.expert;
                l.cpu_experts.push_back(f.expert);
            }
        }

        // AMB_MOE_HIT_FIRST: the miss-phase map: the slots this plan filled, -1 for every other expert; H2 for every
        // selected expert against the hit-phase snapshot (always on: a violation is a wrong result, never continue)
        if (is_r) {
            GGML_ASSERT(cpu.empty());
            int32_t * st_r = hf_stage(il, 1);
            const int32_t * st_h  = hf_stage(il, 0);
            const int32_t * slots = g.lru.slot_map[il];
            const int64_t n_e = l.slot_map->ne[1];
            for (int64_t e = 0; e < n_e; ++e) {
                st_r[e] = -1;
            }
            for (const auto & f : fills) {
                st_r[f.expert] = f.slot;
            }
            for (int32_t e : g.lru.uniq) {
                const bool in_h = st_h[e] >= 0;
                const bool in_r = st_r[e] >= 0;
                if (in_h == in_r || (in_h && st_h[e] != slots[e]) || (in_r && st_r[e] != slots[e])) {
                    GGML_ABORT("amb_moe_hit_first: layer %d expert %d: hit-phase slot %d, miss-phase slot %d, slot after the plan %d "
                        "(each selected expert must be in exactly one phase%s)", il, e, st_h[e], st_r[e], slots[e],
                        hf_no_wait ? "; expected with AMB_MOE_HIT_FIRST_DEBUG_NO_WAIT=1: the plan used stale ids" : "");
                }
            }
            (fills.empty() ? hf_st.r_allhit : hf_st.r_miss)++;
            if (hf_skip_r && !fills.empty()) {
                // AMB_MOE_HIT_FIRST_DEBUG_SKIP_R: negative control, no phase computes the filled experts
                for (const auto & f : fills) {
                    st_r[f.expert] = -1;
                }
                hf_st.skipped_r++;
            }
        }

        // AMB_MOE_MISS_OVERLAP=2 (C24): the split-phase slot map of this lookup, from the host slot map after the plan.
        // Every expert has its slot in row 0 (hit phase) and -1 in row 1 (miss phase), except, with the published ids, the
        // experts filled by this plan and the selected ones whose lookahead upload may still run: their slot moves to row 1
        // and row 0 holds -1 (M7: each selected expert in exactly one row; M8: the hit phase reads only complete slots).
        // On the synchronized path (switch off at run time) every expert stays in row 0: C waits for the uploads before the
        // split as without the switch, and the miss phase computes nothing
        const bool is2 = l.slot_map2 != nullptr && src == l.slot_map2;
        bool s2_any = false;
        if (is2) {
            GGML_ASSERT(il < (int32_t) mo2_stage.size() && mo2_stage[il] != nullptr);
            int32_t * st = mo2_stage[il];
            const int32_t * slots = g.lru.slot_map[il];
            const int64_t n_e = l.slot_map->ne[1];
            for (int64_t e = 0; e < n_e; ++e) {
                st[2*e]     = slots[e];
                st[2*e + 1] = -1;
            }
            if (mo_layer) {
                size_t n_pend = 0;
                for (const auto & f : fills) {
                    if (st[2*f.expert] >= 0) {
                        st[2*f.expert + 1] = st[2*f.expert];
                        st[2*f.expert]     = -1;
                        n_pend++;
                    }
                }
                for (int32_t e : g.lru.uniq) {
                    const int32_t s = slots[e];
                    if (st[2*e] >= 0 && s >= 0 && s < g.lru.n_slots && g.lru.in_flight(s)) {
                        st[2*e + 1] = s;
                        st[2*e]     = -1;
                        n_pend++;
                    }
                }
                s2_any = n_pend > 0;
                mo_st.s2_layers++;
                mo_st.s2_selected += g.lru.uniq.size();
                mo_st.s2_pending  += n_pend;
            } else {
                mo_st.s2_fallback++;
            }
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        if (timed) { t3 = clk::now(); }
        // C18c ORDER_PROBE: C idle point (the scheduler synchronized C before this callback)
        const int la_pc0 = la_probe && la_on && g.la_ok ? la_tev(backend) : -1;
        uint64_t la_pseq = 0; // the batch the I4 branch needs
        // AMB_MOE_MISS_OVERLAP: the miss uploads go on U (M3), C keeps running the tail of the previous split
        // C25 AMB_MOE_MISS_STREAM: on M instead of U (S1..S3)
        ggml_backend_t up = mo_layer ? (devices[g.id].miss != nullptr ? devices[g.id].miss : devices[g.id].upload) : backend;
        if (mo_layer && c25_stats && !fills.empty()) {
            // C25 AMB_MOE_C25_STATS (diagnostic, one event query): is the newest lookahead batch still on U now; read only,
            // la_done is not advanced here (no change to the waits below)
            device & d = devices[g.id];
            c25_st.st_miss++;
            if (d.la_last > d.la_done && d.la_flushed == d.la_last && la_query(d.la_ev[d.la_last % (uint64_t) d.la_n_ev]) == 0) {
                c25_st.st_behind++;
            }
        }
        if (copy_mark_fn != nullptr) {
            copy_mark_fn(copy_mark_ud, up, mo_layer ? 2 : 0); // AMB_STEP_EVENTS: before the miss uploads
        }
        size_t bytes = 0;
        // AMB_MOE_BATCH_COPY: collect the copies of the layer and issue them with one batched call
        if (batch_copy && !batch_resolved) {
            batch_resolved = true;
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
            batch_fn = reg ? (batch_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_memcpy_batch_async") : nullptr;
            LLAMA_LOG_INFO("%s: AMB_MOE_BATCH_COPY=1: batched miss fills %s\n", __func__, batch_fn ? "enabled" : "unavailable (not a CUDA backend)");
        }
        const bool batching = batch_copy && batch_fn != nullptr;
        if (batching) {
            bc_items.clear();
        }
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                if (mo_layer && mo_spin_ns > 0) {
                    // AMB_MOE_MISS_OVERLAP_DEBUG_SPIN_US: the fill is 0xFF until its upload (U, before the delay below)
                    if (!la_memset(up, (char *) bank->data + fills[i].slot*expert_size, n*expert_size, 0xFF, n*expert_size, 1)) {
                        GGML_ABORT("amb_miss_overlap: DEBUG_SPIN_US: poisoning a fill slot failed");
                    }
                    mo_st.poisoned++;
                }
                if (batching) {
                    bc_items.push_back({bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size});
                } else {
                    ggml_backend_tensor_set_async(up, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                }
                bytes += n*expert_size;
                i += n;
            }
        }
        if (mo_layer && mo_spin_ns > 0 && !fills.empty()) {
            // AMB_MOE_MISS_OVERLAP_DEBUG_SPIN_US: U is held back after the poison, so a missing C wait reads 0xFF slots
            if (!la_spin(up, mo_spin_ns)) {
                GGML_ABORT("amb_miss_overlap: DEBUG_SPIN_US: the delay kernel failed");
            }
        }
        if (batching && !bc_items.empty()) {
            bool done = false;
            if (bc_items.size() >= 2) {
                bc_dsts.resize(bc_items.size());
                bc_srcs.resize(bc_items.size());
                bc_sizes.resize(bc_items.size());
                size_t bc_bytes = 0;
                for (size_t k = 0; k < bc_items.size(); ++k) {
                    // the address ggml_backend_tensor_set_async(backend, bank, src, offset, size) writes to
                    bc_dsts[k]  = (char *) bc_items[k].bank->data + bc_items[k].offset;
                    bc_srcs[k]  = bc_items[k].src;
                    bc_sizes[k] = bc_items[k].size;
                    bc_bytes   += bc_items[k].size;
                }
                done = batch_fn(up, bc_dsts.data(), bc_srcs.data(), bc_sizes.data(), bc_items.size());
                if (done) {
                    bc_calls++;
                    bc_copies += bc_items.size();
                    bc_bytes_total += bc_bytes;
                } else {
                    bc_fallbacks++;
                }
            }
            if (!done) {
                for (const bc_item & it : bc_items) {
                    ggml_backend_tensor_set_async(up, it.bank, it.src, it.offset, it.size);
                }
            }
        }
        if (copy_mark_fn != nullptr) {
            copy_mark_fn(copy_mark_ud, up, mo_layer ? 3 : 1); // AMB_STEP_EVENTS: after the miss uploads
        }
        // AMB_MOE_MISS_OVERLAP (M3): the event C waits on below, recorded on U after every miss upload of this layer
        if (mo_layer && !fills.empty()) {
            device & d = devices[g.id];
            // recorded in mode 2 too (C24 review S5): nothing waits on it there, but mo_wait queries it as a driver flush
            ggml_backend_event_record(d.mo_ev, up);
            d.mo_dirty = true;
            mo_st.miss_batches++;
            mo_st.miss_bytes += bytes;
            if (d.miss != nullptr) {
                c25_st.ms_batches++; // C25: on M
                c25_st.ms_bytes += bytes;
            }
        }
        // AMB_MOE_MISS_OVERLAP=2 (M8): the gate event of this layer, recorded on U after its miss uploads (and, U being in
        // order, after every lookahead batch issued before them); the negative control records nothing, so the gate of
        // this layer waits for an old record only
        if (is2 && mo_layer && s2_any) {
            if (!mo_skip_wait) {
                ggml_backend_event_record(mo2_ev[il], up);
                mo_st.s2_records++;
            } else {
                mo_st.s2_skipped++;
            }
        }

        // AMB_MOE_LOOKAHEAD, I4: the plan selected slots whose lookahead upload may still run: C waits (GPU side, the host
        // does not block) for the newest of their batches, after the blocking copies were enqueued and before the kernels
        // (a drain inside the plan, no victim, may have completed the batch already: then no wait); la_ev[seq % 4] holds the
        // newest batch of that residue, >= seq, so waiting on it covers seq
        // (C24 mode 2 with the published ids: the in-flight slots are in the miss phase, behind the gate, M8: no wait here)
        if (la_on && g.la_ok && g.lru.pf_need_wait && g.lru.pf_wait_seq > devices[g.id].la_done && !(is2 && mo_layer)) {
            const clk::time_point tq = clk::now();
            device & d = devices[g.id];
            GGML_ASSERT(g.lru.pf_wait_seq <= d.la_last);
            la_pseq = g.lru.pf_wait_seq;
            if (!la_skip_wait) {
                // DEBUG_DEFER: the copies of the needed batches are enqueued (and recorded) before the wait
                while (!la_defer_q.empty() && la_defer_q.front().seq <= g.lru.pf_wait_seq) {
                    la_flush_front();
                    la_defer_fl_wait++;
                }
            }
            const int ta = la_tev(backend);
            if (!la_skip_wait) {
                ggml_backend_event_wait(backend, d.la_ev[g.lru.pf_wait_seq % (uint64_t) d.la_n_ev]);
            } else {
                la_skipped_waits++; // AMB_MOE_LOOKAHEAD_DEBUG_SKIP_WAIT: negative control
            }
            const int tb = la_tev(backend);
            if (ta >= 0 && tb >= 0) {
                la_tev_wait.push_back({ ta, tb });
            }
            if (la_cls_ok) {
                la_st[n_tok == 1 ? 0 : 1].waits++;
            }
            la_ns_other += ns_between(tq, clk::now());
        }

        if (timed) {
            t4 = clk::now();
            if (timing_mode == 2 && !fills.empty()) {
                // mode 2: the stream was synchronized above, so this waits for the transfer of the missed experts alone
                ggml_backend_synchronize(backend);
            }
            t5 = clk::now();
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size() + cpu.size();
        st.cpu    += cpu.size();
        st.bytes  += bytes;
        st.routes += ids.size();
        st.calls  += 1;

        // AMB_MOE_PREFETCH: order C after the uploads the kernels of this split read, then prefetch layer il + 1
        // before the slot map upload: a prefetch victim can change a (not selected) entry of this slot map
        int64_t pf_ns = 0;
        if (pf_mode != 0) {
            const clk::time_point tp = timed ? clk::now() : clk::time_point{};
            prefetch_order(backend, g.id, g.lru.pf_need_wait);
            if (sel->src[0]->ne[1] == 1) {
                pf.steps    += il == pf_first_layer ? 1 : 0;
                pf.blocking += fills.size();
            }
            if (pf_t != nullptr) {
                prefetch(il, il + 1);
            }
            if (timed) {
                pf_ns = ns_between(tp, clk::now());
                pf.ns += pf_ns;
            }
        }

        // the next copy synchronizes the backend before it changes the slot map again
        // AMB_IDS_ASYNC: from a pinned staging copy (a truly async upload). The staging copy of layer il is written again
        // only by the next copy() of layer il, which runs after the scheduler's synchronize before the callback, or (pipeline
        // parallel scheduler) after waiting on an ids event recorded later on this stream: the upload has executed by then
        // AMB_MOE_MISS_OVERLAP (M2): on C, so it is ordered after the tail still running on C; the next copy() of layer il
        // writes the staging copy only after it saw a later publish of layer il on C, i.e. after this upload executed
        if (is_r) {
            // AMB_MOE_HIT_FIRST: the miss-phase map, on C after the fills of this layer and before its split (H4)
            ggml_backend_tensor_set_async(backend, dst, hf_stage(il, 1), 0, ggml_nbytes(dst));
        } else if (is2) {
            // AMB_MOE_MISS_OVERLAP=2: the split-phase slot map, from its pinned staging (M2 holds as for the plain map)
            ggml_backend_tensor_set_async(backend, dst, mo2_stage[il], 0, ggml_nbytes(dst));
            ia.stage_uploads++;
        } else if (ia.on && il < (int32_t) ia.stage.size() && ia.stage[il] != nullptr) {
            // AMB_MOE_LOOP_DEDUP: lru.slot_map[il] is the shared map of the physical layer (== l.slot_map->data when off)
            memcpy(ia.stage[il], g.lru.slot_map[il], ggml_nbytes(l.slot_map));
            ggml_backend_tensor_set_async(backend, dst, ia.stage[il], 0, ggml_nbytes(l.slot_map));
            ia.stage_uploads++;
        } else {
            ggml_backend_tensor_set_async(backend, dst, g.lru.slot_map[il], 0, ggml_nbytes(l.slot_map));
        }
        // AMB_MOE_MISS_OVERLAP, wait site of M3: no kernel of this layer may read a fill before its upload on U has
        // completed; C waits (GPU side, the host does not block) on the event recorded on U after the miss uploads, after the
        // slot map upload and before the split's kernels. Layers without misses have nothing on U to wait for (the
        // lookahead slots they select are covered by the I4 wait above)
        if (mo_layer && !fills.empty() && !is2) {
            if (!mo_skip_wait) {
                ggml_backend_event_wait(backend, devices[g.id].mo_ev);
                mo_st.c_waits++;
            } else {
                mo_st.skipped_waits++; // AMB_MOE_MISS_OVERLAP_DEBUG_SKIP_WAIT: negative control
            }
        }
        // C18c ORDER_PROBE: after the last H2D copy of C before the kernels of this layer
        if (la_pc0 >= 0 && la_pseq > 0) {
            const int la_pc1 = la_tev(backend);
            int te = -1;
            for (const auto & pe : la_probe_end) {
                if (pe.first == la_pseq) {
                    te = pe.second;
                }
            }
            if (te >= 0 && la_pc1 >= 0) {
                la_probe_q.push_back({ te, la_pc0, la_pc1 });
            } else {
                la_probe_unmatched++;
            }
        }

        // AMB_MOE_LOOKAHEAD: plan and issue layer il + 1 now that the blocking copies and the slot map upload of layer il are
        // enqueued on C (the staging copy of this slot map is taken: admission may change its unselected entries freely)
        int64_t la_ns = 0;
        if (la_on) {
            const clk::time_point tl = clk::now();
            const bool want = la_cls_ok && lookahead_mode(il + 1, n_tok) != 0;
            if (la_have && want) {
                if (la_post && !la_dry) {
                    // C18b ISSUE=post: planned and issued by the post-launch hook of this split (ids_post_split); the
                    // compute-side ordering event is recorded now, right after this copy's blocking copies and slot map upload
                    la_pend.active   = true;
                    la_pend.graph    = graph;
                    la_pend.backend  = backend;
                    la_pend.il_real  = il;
                    la_pend.il       = il + 1;
                    la_pend.n_tok    = n_tok;
                    la_pend.recorded = la_link_wait;
                    la_pend.mo       = mo_layer && !mo_pred_now; // AMB_MOE_MISS_OVERLAP: take the publish at the issue
                    if (la_link_wait) {
                        ggml_backend_event_record(devices[groups[layers[il + 1].ig].id].la_ev_c, backend);
                    }
                } else if (!mo_layer || mo_pred_now || mo_take_pred(il + 1, n_tok)) {
                    lookahead_issue(backend, il, il + 1, n_tok, true);
                }
            } else if (want) {
                la_st[n_tok == 1 ? 0 : 1].no_readback++; // the graph should have the predictor: its ids were not read back
            } else if (la_have && la_cls_ok) {
                la_st[n_tok == 1 ? 0 : 1].bad++;
            }
            la_have = false;
            la_ns = ns_between(tl, clk::now());
            if (la_cls_ok) {
                la_st[n_tok == 1 ? 0 : 1].ns_other += la_ns_other;
            }
        }

        if (timed) {
            t6 = clk::now();
            tm.wait    += ns_between(t0, t1);
            tm.plan    += ns_between(t2, t3);
            tm.enqueue += ns_between(t3, t4) + ns_between(t5, t6) - pf_ns - la_ns; // the (look-ahead) prefetch time is logged on its own
            tm.upload  += ns_between(t4, t5);
            tm.copies++;
            tm.miss_experts += (int64_t) fills.size();
            tm.miss_bytes   += (int64_t) bytes;
            graph_used = true;
        }

        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu (%zu on the CPU), hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, st.cpu, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
    }

    // AMB_MOE_STATS: cumulative statistics, and the hit rate and uploads since the previous periodic log
    void log_stats_periodic() {
        auto log = [&](const char * name, const stats & st, const stats & prev) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            const size_t d_hits  = st.hits   - prev.hits;
            const size_t d_n     = n - (prev.hits + prev.misses);
            const size_t d_calls = st.calls  - prev.calls;
            LLAMA_LOG_INFO("llama_moe_cache: amb_moe_stats: round %lld: %s: hits = %zu, misses = %zu (%zu on the CPU), hit rate = %.2f%%, "
                "uploaded = %.2f MiB, lookups = %zu, routes = %zu, distinct experts/lookup = %.2f | interval: hit rate = %.2f%%, "
                "uploaded = %.2f MiB, lookups = %zu, distinct experts/lookup = %.2f\n",
                (long long) stats_rounds, name, st.hits, st.misses, st.cpu, 100.0*st.hits/n, st.bytes/1024.0/1024.0,
                st.calls, st.routes, st.calls > 0 ? (double) n/st.calls : 0.0,
                d_n > 0 ? 100.0*d_hits/d_n : 0.0, (st.bytes - prev.bytes)/1024.0/1024.0, d_calls, d_calls > 0 ? (double) d_n/d_calls : 0.0);
        };
        log("ubatch <= 8", stats_small, stats_prev_small);
        log("ubatch  > 8", stats_large, stats_prev_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: amb_moe_stats: large batches: %zu experts copied from the cache, %.2f MiB\n",
                stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
        if (batch_copy) {
            LLAMA_LOG_INFO("llama_moe_cache: amb_moe_batchcopy: round %lld: batch_calls = %zu, copies = %zu, MiB = %.2f, fallbacks = %zu\n",
                (long long) stats_rounds, bc_calls, bc_copies, bc_bytes_total/1024.0/1024.0, bc_fallbacks);
        }
        log_ids_async("periodic");
        hf_log("periodic");
        if (timing_mode > 0) {
            log_timing_periodic();
        }
        if (pf_mode != 0) {
            const pf_stats t = pf_totals();
            log_prefetch("interval  ", t, pf_prev, stats_rounds);
            log_prefetch("cumulative", t, pf_stats{}, stats_rounds);
            if (timing_mode > 0) {
                const double r_i = (double) (tm.graphs - tm_prev_pf_graphs);
                const double r_c = (double) tm.graphs;
                LLAMA_LOG_INFO("llama_moe_cache: amb_moe_timing: prefetch host ms per round (planning + enqueue on the upload stream, "
                    "part of the remainder above): interval = %.3f, cumulative = %.3f\n",
                    r_i > 0 ? (t.ns - pf_prev.ns)*1e-6/r_i : 0.0, r_c > 0 ? t.ns*1e-6/r_c : 0.0);
                tm_prev_pf_graphs = tm.graphs;
            }
            pf_prev = t;
        }
        if (predict) {
            log_predict("interval  ", true);
            log_predict("cumulative", false);
            pred_prev = pred_cum;
        }
        stats_prev_small = stats_small;
        stats_prev_large = stats_large;
    }

    // AMB_MOE_TIMING: per round (graph evaluation) averages, in ms
    void log_timing_periodic() {
        auto line = [&](const char * what, const timing & t) {
            if (t.graphs == 0) {
                return;
            }
            const double r   = (double) t.graphs;
            const double ms  = 1e-6;
            const double sum = (double) (t.wait + t.plan + t.enqueue + t.upload);
            LLAMA_LOG_INFO("llama_moe_cache: amb_moe_timing: %s: rounds=%lld, per-round ms: total=%.3f, readback_wait=%.3f, plan=%.3f, enqueue=%.3f, "
                "upload_wait(mode2)=%.3f, remainder=%.3f, miss_experts=%.2f, miss_MiB=%.3f, copies/round=%.2f | mode=%d%s\n",
                what, (long long) t.graphs, t.graph*ms/r, t.wait*ms/r, t.plan*ms/r, t.enqueue*ms/r, t.upload*ms/r, ((double) t.graph - sum)*ms/r,
                t.miss_experts/r, t.miss_bytes/1024.0/1024.0/r, t.copies/r, timing_mode,
                timing_mode == 2 ? " (mode 2 perturbs execution: total is not a performance number)" : "");
        };
        timing d;
        d.wait         = tm.wait         - tm_prev.wait;
        d.plan         = tm.plan         - tm_prev.plan;
        d.enqueue      = tm.enqueue      - tm_prev.enqueue;
        d.upload       = tm.upload       - tm_prev.upload;
        d.graph        = tm.graph        - tm_prev.graph;
        d.graphs       = tm.graphs       - tm_prev.graphs;
        d.copies       = tm.copies       - tm_prev.copies;
        d.miss_experts = tm.miss_experts - tm_prev.miss_experts;
        d.miss_bytes   = tm.miss_bytes   - tm_prev.miss_bytes;
        line("interval  ", d);
        line("cumulative", tm);
        tm_prev = tm;
    }

    bool predict_layer(int32_t il, int64_t n_tokens) const {
        return predict && n_tokens == 1 && get_slot_map(il, n_tokens, n_expert_used, false) != nullptr;
    }

    void predict_add(int32_t il, int32_t p, const int32_t * pred) {
        if (!predict || il < 0 || il >= (int32_t) layers.size() || p < 0 || p >= n_pred) {
            return;
        }
        const pred_rec & r = pred_recs[il];
        if (r.gen != pred_gen || r.real.empty()) {
            return; // the copy() of this layer did not run for this evaluation
        }
        const int32_t n_expert = (int32_t) r.cached.size();
        for (int i = 0; i < predict_n_ids; ++i) {
            if (pred[i] < 0 || pred[i] >= n_expert) {
                pred_bad++;
                return;
            }
        }
        pred_acc & a = pred_cum[(size_t) p*layers.size() + il];
        a.evals++;
        a.real += r.real.size();
        for (int32_t e : r.real) {
            a.miss += r.cached[e] ? 0 : 1;
        }
        for (int ik = 0; ik < n_k; ++ik) {
            const int32_t k = pred_k[ik];
            for (int32_t e : r.real) {
                if (std::find(pred, pred + k, e) != pred + k) {
                    a.hit[ik]++;
                    a.miss_hit[ik] += r.cached[e] ? 0 : 1;
                }
            }
            for (int32_t i = 0; i < k; ++i) {
                const int32_t e = pred[i];
                if (!r.cached[e] && std::find(r.real.begin(), r.real.end(), e) == r.real.end()) {
                    a.waste[ik]++;
                }
            }
        }
    }

    // AMB_MOE_PREDICT: one line per predictor, for the interval since the previous periodic log and cumulative
    // recall = selected experts in the predicted set / selected experts, pooled over the steps and layers
    // miss recall = the same over the selected experts that were not cached before the plan
    // waste = predicted experts that were neither selected nor cached, per (step, layer)
    void log_predict(const char * what, bool interval) const {
        auto pct = [](uint64_t a, uint64_t b) { return b > 0 ? 100.0*a/b : 0.0; };
        const size_t n_layer = layers.size();
        for (int p = 0; p < n_pred; ++p) {
            pred_acc t;
            std::vector<std::pair<double, int>> lr[n_k]; // per-layer recall, layer
            for (size_t il = 0; il < n_layer; ++il) {
                const pred_acc & c = pred_cum[(size_t) p*n_layer + il];
                pred_acc d = c;
                if (interval) {
                    const pred_acc & q = pred_prev[(size_t) p*n_layer + il];
                    d.evals -= q.evals;
                    d.real  -= q.real;
                    d.miss  -= q.miss;
                    for (int ik = 0; ik < n_k; ++ik) {
                        d.hit[ik]      -= q.hit[ik];
                        d.miss_hit[ik] -= q.miss_hit[ik];
                        d.waste[ik]    -= q.waste[ik];
                    }
                }
                t.evals += d.evals;
                t.real  += d.real;
                t.miss  += d.miss;
                for (int ik = 0; ik < n_k; ++ik) {
                    t.hit[ik]      += d.hit[ik];
                    t.miss_hit[ik] += d.miss_hit[ik];
                    t.waste[ik]    += d.waste[ik];
                    if (d.real > 0) {
                        lr[ik].emplace_back(pct(d.hit[ik], d.real), (int) il);
                    }
                }
            }
            if (t.evals == 0) {
                continue;
            }
            const double ev = (double) t.evals;
            char per_layer[256] = "";
            if (!interval) {
                // min/median/max over the layers of recall@k for the smallest and the widest set
                int off = 0;
                for (int ik : { 0, n_k - 1 }) {
                    auto v = lr[ik];
                    if (v.empty()) {
                        continue;
                    }
                    std::sort(v.begin(), v.end());
                    off += snprintf(per_layer + off, sizeof(per_layer) - off,
                        " | per-layer recall@%d min/med/max = %.2f (L%d) / %.2f / %.2f (L%d)%%",
                        pred_k[ik], v.front().first, v.front().second, v[v.size()/2].first, v.back().first, v.back().second);
                    if (off < 0 || off >= (int) sizeof(per_layer)) {
                        break;
                    }
                }
            }
            LLAMA_LOG_INFO("llama_moe_cache: amb_moe_predict: round %lld: %s: %s: evals = %llu, "
                "recall@%d/%d/%d = %.2f/%.2f/%.2f%%, miss recall@%d/%d/%d = %.2f/%.2f/%.2f%% (misses = %.2f/eval), "
                "waste/eval@%d/%d/%d = %.2f/%.2f/%.2f%s\n",
                (long long) stats_rounds, what, predict_name(p), (unsigned long long) t.evals,
                pred_k[0], pred_k[1], pred_k[2], pct(t.hit[0], t.real), pct(t.hit[1], t.real), pct(t.hit[2], t.real),
                pred_k[0], pred_k[1], pred_k[2], pct(t.miss_hit[0], t.miss), pct(t.miss_hit[1], t.miss), pct(t.miss_hit[2], t.miss),
                t.miss/ev,
                pred_k[0], pred_k[1], pred_k[2], t.waste[0]/ev, t.waste[1]/ev, t.waste[2]/ev,
                per_layer);
        }
        if (pred_bad > 0) {
            LLAMA_LOG_WARN("llama_moe_cache: amb_moe_predict: %llu predicted id lists out of range were skipped\n", (unsigned long long) pred_bad);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share, bool elastic_layout, size_t elastic_lend_bytes) :
    pimpl(new impl(model, no_alloc, backends, bufts, size, cpu_share, elastic_layout, elastic_lend_bytes)) {
}

llama_elastic_lender * llama_moe_cache::elastic_lender() {
    return pimpl->el_live ? &pimpl->el_lender : nullptr;
}

bool llama_moe_cache::elastic_layout() const {
    return pimpl->el;
}

bool llama_moe_cache::elastic_floor_short() const {
    if (!pimpl->el_live || pimpl->el_floor_bytes == 0) {
        return false;
    }
    for (const auto & g : pimpl->groups) {
        if (g.n_slots > 0 && g.lru.n_active < g.n_graph) {
            return true;
        }
    }
    return false;
}

size_t llama_moe_cache::elastic_floor_bytes() const {
    return pimpl->el_live ? pimpl->el_floor_bytes : 0;
}

int32_t llama_moe_cache::ring_plan(size_t slot_size, int32_t n_want) {
    return pimpl->ring_plan(slot_size, n_want);
}

ggml_backend_buffer_t llama_moe_cache::ring_buffer() const {
    return pimpl->el_live && !pimpl->devices.empty() ? pimpl->devices[0].buf.get() : nullptr;
}

void * llama_moe_cache::ring_ptr(int32_t k) const {
    return k >= 0 && k < (int32_t) pimpl->ring_ptrs.size() ? pimpl->ring_ptrs[k] : nullptr;
}

size_t llama_moe_cache::ring_borrow() {
    return pimpl->ring_borrow();
}

int64_t llama_moe_cache::ring_return() {
    return pimpl->ring_return();
}

int64_t llama_moe_cache::ring_slots_parked() const {
    return pimpl->ring_slots_parked();
}

llama_moe_cache::~llama_moe_cache() = default;

// smallest budget that gives every group of host-resident expert layers on dev the slots for the experts of one token
static size_t llama_moe_cache_min_size_dev(const llama_model & model, ggml_backend_dev_t dev) {
    const int32_t n_expert_used = model.hparams.n_expert_used_max();
    const size_t  alignment     = ggml_backend_buft_get_alignment(ggml_backend_dev_buffer_type(dev));

    // same grouping as the constructor, the budget is split by the size of the experts
    std::vector<std::vector<ggml_tensor *>> refs;
    std::vector<size_t> group_bytes;
    size_t host_bytes = 0;
    // AMB_MOE_LOOP_DEDUP: each physical expert tensor counted once, as in the constructor (no log here)
    const char * why = nullptr;
    const std::vector<int32_t> rep = llama_moe_cache_loop_rep(model, llama_moe_cache_loop_dedup_env(&why) && model.hparams.loop_n_pass > 1);
    for (size_t il = 0; il < model.layers.size(); ++il) {
        auto experts = llama_moe_cache_layer_experts(model.layers[il]);
        if (experts.empty() || model.dev_layer(il) != dev ||
            !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
            continue;
        }
        if (rep[il] != (int32_t) il) {
            const auto r_experts = llama_moe_cache_layer_experts(model.layers[rep[il]]);
            bool same = r_experts.size() == experts.size() && model.dev_layer(rep[il]) == dev &&
                std::all_of(r_experts.begin(), r_experts.end(), llama_moe_cache_is_host_weight) &&
                llama_moe_cache_same_layout(r_experts, experts);
            for (size_t k = 0; same && k < experts.size(); ++k) {
                same = r_experts[k]->data == experts[k]->data;
            }
            if (same) {
                continue; // counted with the representative
            }
        }
        auto it = std::find_if(refs.begin(), refs.end(), [&](const std::vector<ggml_tensor *> & ref) { return llama_moe_cache_same_layout(ref, experts); });
        if (it == refs.end()) {
            refs.push_back(experts);
            group_bytes.push_back(0);
            it = refs.end() - 1;
        }
        for (const ggml_tensor * t : experts) {
            group_bytes[it - refs.begin()] += ggml_nbytes(t);
            host_bytes                     += ggml_nbytes(t);
        }
    }

    size_t res = 0;
    for (size_t ig = 0; ig < refs.size(); ++ig) {
        size_t need = 0; // n_expert_used slots + the extra slot
        for (const ggml_tensor * t : refs[ig]) {
            need += GGML_PAD(t->nb[2]*(n_expert_used + 1), alignment);
        }
        res = std::max(res, (size_t) std::ceil((double) need*host_bytes/group_bytes[ig]) + 1);
    }
    return res;
}

size_t llama_model_moe_cache_min_size(const llama_model * model) {
    if (model->hparams.n_expert == 0 || model->hparams.n_expert_used_max() == 0 || model->devices.size() != 1 ||
            model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        return 0;
    }
    // same devices as the cache itself
    const auto dev_type = ggml_backend_dev_type(model->devices[0].dev);
    if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU && dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
        return 0;
    }
    return llama_moe_cache_min_size_dev(*model, model->devices[0].dev);
}

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

llama_moe_cache_lookup llama_moe_cache::build_lookup(const llm_graph_context & g, const llama_moe_cache_layer & layer, ggml_tensor * selected_experts) const {
    const int32_t il = layer.il;
    ggml_context * ctx0 = g.ctx0;

    // part of the misses may be computed from the host experts, but not with weights before the FFN, LoRAs or rotated experts
    bool cpu_part = !layer.weight_before_ffn && g.loras->empty();
    for (ggml_tensor * w : { layer.up_exps, layer.gate_exps, layer.down_exps, layer.gate_up_exps }) {
        cpu_part = cpu_part && !(w != nullptr && g.hdmd != nullptr && g.hdmd->rot.count(w) > 0);
    }

    ggml_tensor * cpu_map  = cpu_part ? pimpl->get_cpu_map(il, selected_experts->ne[1]) : nullptr;
    ggml_tensor * slot_map = pimpl->get_slot_map(il, selected_experts->ne[1], selected_experts->ne[0], cpu_map != nullptr);
    // AMB_MOE_MISS_OVERLAP=2 (C24): the [2, n_expert] split-phase map instead (never with a CPU part: the switch refuses it)
    // C24 review S1: not with LoRA adapters (build_lora_mm_id would add each delta in both phases); such graphs use the
    // plain map, i.e. the =1 behaviour (copy() keys on the looked-up tensor)
    const bool split2 = slot_map != nullptr && cpu_map == nullptr && g.loras->empty() &&
        miss_overlap_split(il, selected_experts->ne[1]);
    if (split2) {
        slot_map = get_slot_map2(il);
    }
    // AMB_MOE_HIT_FIRST (D2): two lookups, the hit-phase map here and the miss-phase map below (same exclusions as C24)
    const bool hit_first = !split2 && slot_map != nullptr && cpu_map == nullptr && g.loras->empty() &&
        pimpl->hit_first(il, selected_experts->ne[1]);
    if (hit_first) {
        slot_map = pimpl->layers[il].slot_map_h;
    }
    if (slot_map == nullptr) {
        return {};
    }
    for (ggml_tensor * w : { layer.up_exps, layer.gate_exps, layer.down_exps, layer.gate_up_exps }) {
        if (w == nullptr) {
            continue;
        }
        ggml_tensor * experts = pimpl->get_experts(w);
        if (experts == nullptr) {
            return {};
        }
        // the device of the cache must compute the matmul, else the scheduler copies the cache to another backend
        ggml_tensor * b = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, experts->ne[0], w == layer.down_exps ? selected_experts->ne[0] : 1, selected_experts->ne[1]);
        if (!ggml_backend_supports_op(pimpl->backend(il), ggml_mul_mat_id(ctx0, experts, b, selected_experts))) {
            return {};
        }
    }

    ggml_tensor * ids = selected_experts;
    if (!ggml_is_contiguous(ids)) {
        ids = ggml_cont(ctx0, ids);
    }
    ids = ggml_reshape_1d(ctx0, ids, ggml_nelements(ids));

    // the slot map is a host weight, so the scheduler starts a new split here and copies it with the copy callback
    // the callback reads the selected experts, uploads the missing ones and updates the slot map
    ggml_tensor * slots = ggml_get_rows(ctx0, slot_map, ids); // [1, n_expert_used*n_tokens]
    if (!ggml_backend_supports_op(pimpl->backend(il), slots)) {
        return {};
    }
    ggml_backend_sched_set_tensor_backend(g.sched, slots, pimpl->backend(il));
    g.cb(slots, "ffn_moe_slots", il);

    // AMB_MOE_MISS_OVERLAP=1 (C23, default off; see copy()): publish the selected ids to the host now (a kernel that
    // writes them to mapped pinned memory), then let the model expand the work that does not depend on the experts of
    // this layer (g.moe_pre_lookup), all before the lookup: they end the split that precedes the lookup, so they run
    // while the host plans this layer and the missed experts upload. With the switch unset nothing is added here
    if (miss_overlap(il, selected_experts->ne[1])) {
        ggml_tensor * pub = miss_overlap_publish(ctx0, ids, il, 0);
        ggml_backend_sched_set_tensor_backend(g.sched, pub, pimpl->backend(il));
        ggml_build_forward_expand(g.gf, pub);
        if (g.moe_pre_lookup) {
            g.moe_pre_lookup();
        }
    }

    // add the lookup now: the transform of a Hadamard-folded expert reads host weights and would start its split first
    ggml_build_forward_expand(g.gf, slots);

    // the misses computed from the host experts, read after the lookup has picked them
    ggml_tensor * cpu_ids = nullptr;
    if (cpu_map != nullptr) {
        cpu_ids = ggml_get_rows(ctx0, cpu_map, ids); // [1, n_expert_used*n_tokens]
        cpu_ids = ggml_reshape_2d(ctx0, cpu_ids, selected_experts->ne[0], selected_experts->ne[1]);
    }

    if (hit_first) {
        // the miss-phase lookup is not expanded here: build_experts expands the hit chain first, so this lookup (a host
        // weight read: the scheduler starts a split at it) follows the whole hit chain in the graph
        ggml_tensor * slots_r = ggml_get_rows(ctx0, pimpl->layers[il].slot_map_r, ids); // [1, n_expert_used*n_tokens]
        if (!ggml_backend_supports_op(pimpl->backend(il), slots_r)) {
            GGML_ABORT("amb_moe_hit_first: the cache device does not support the miss-phase lookup of layer %d", il);
        }
        ggml_backend_sched_set_tensor_backend(g.sched, slots_r, pimpl->backend(il));
        g.cb(slots_r, "ffn_moe_slots_miss", il);
        ggml_tensor * hit  = ggml_reshape_2d(ctx0, slots,   selected_experts->ne[0], selected_experts->ne[1]);
        ggml_tensor * miss = ggml_reshape_2d(ctx0, slots_r, selected_experts->ne[0], selected_experts->ne[1]);
        hit->flags  |= GGML_TENSOR_FLAG_AMB_MASKED_IDS;
        miss->flags |= GGML_TENSOR_FLAG_AMB_MASKED_IDS;
        llama_moe_cache_lookup res_hf = { il, hit, nullptr };
        res_hf.slots_miss = miss;
        return res_hf;
    }

    if (split2) {
        // row 0 of each looked-up pair: the hit-phase slot, row 1: the miss-phase slot (-1 where the phase skips the pair);
        // the miss-phase slots pass through the gate, so every node that reads them runs after the layer's miss uploads
        const int64_t n = ggml_nelements(ids);
        ggml_tensor * hit  = ggml_cont(ctx0, ggml_view_2d(ctx0, slots, 1, n, slots->nb[1], 0));
        ggml_tensor * miss = ggml_cont(ctx0, ggml_view_2d(ctx0, slots, 1, n, slots->nb[1], ggml_element_size(slots)));
        g.cb(hit, "ffn_moe_slots_hit", il);
        miss = miss_overlap_gate(ctx0, ggml_reshape_1d(ctx0, miss, n), il);
        ggml_backend_sched_set_tensor_backend(g.sched, miss, pimpl->backend(il));
        g.cb(miss, "ffn_moe_slots_miss", il);
        hit  = ggml_reshape_2d(ctx0, hit,  selected_experts->ne[0], selected_experts->ne[1]);
        miss = ggml_reshape_2d(ctx0, miss, selected_experts->ne[0], selected_experts->ne[1]);
        hit->flags  |= GGML_TENSOR_FLAG_AMB_MASKED_IDS;
        miss->flags |= GGML_TENSOR_FLAG_AMB_MASKED_IDS;
        llama_moe_cache_lookup res2 = { il, hit, nullptr };
        res2.slots_miss = miss;
        return res2;
    }

    return { il, ggml_reshape_2d(ctx0, slots, selected_experts->ne[0], selected_experts->ne[1]), cpu_ids };
}

ggml_tensor * llama_moe_cache::build_experts(const llm_graph_context & g, const llama_moe_cache_lookup & lookup,
        ggml_tensor * cur, ggml_tensor * selected_experts, llama_moe_cache_build_experts_t build, void * user_data) const {
    GGML_ASSERT(lookup.slots != nullptr);

    ggml_tensor * experts = build(cur, selected_experts, lookup.slots, true, user_data);
    if (lookup.slots_miss != nullptr) {
        // AMB_MOE_MISS_OVERLAP=2 (C24): the hit phase above runs while the misses upload; the miss phase reads its slots
        // through the gate. Each (token, slot) row is written by exactly one phase, the other phase writes -0.0f there,
        // so the ADD reproduces the single-phase down output bit for bit (x + -0.0f == x) and the weighted sum below is
        // the unchanged one
        ggml_build_forward_expand(g.gf, experts);
        experts = ggml_add(g.ctx0, experts, build(cur, selected_experts, lookup.slots_miss, true, user_data));
        g.cb(experts, "ffn_moe_down_merged", lookup.il);
    }
    if (lookup.cpu_ids != nullptr) {
        // the cached experts and the misses computed from the host experts, each reads the others as zero
        // the cached experts come first in the graph, so that they are computed while the misses are
        ggml_build_forward_expand(g.gf, experts);
        experts = ggml_add(g.ctx0, experts, build(cur, selected_experts, lookup.cpu_ids, false, user_data));
        g.cb(experts, "ffn_moe_down_merged", lookup.il);
    }

    return experts;
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

bool llama_moe_cache::ids_async_enabled() const {
    return pimpl->ia.on;
}

void llama_moe_cache::ids_post_split(const ggml_backend_sched_split_info * info) {
    pimpl->ids_post_split(info);
}

void llama_moe_cache::ids_async_reset() {
    pimpl->ids_async_reset();
    // AMB_MOE_MISS_OVERLAP: before a graph evaluation, a publish launched by an earlier evaluation whose copy() never ran
    // (an aborted evaluation) is not pending anymore, so a later lookup without a publish of its own is synchronized as
    // before. The flags keep counting (M6): the next publish of the slot is waited for by its own sequence number
    if (pimpl->mo_on) {
        pimpl->mo_taken = pimpl->mo_expect;
    }
}

bool llama_moe_cache::miss_overlap(int32_t il, int64_t n_tokens) const {
    return pimpl->miss_overlap(il, n_tokens);
}

ggml_tensor * llama_moe_cache::miss_overlap_publish(ggml_context * ctx, ggml_tensor * ids, int32_t il, int kind) const {
    return pimpl->miss_overlap_publish(ctx, ids, il, kind);
}

bool llama_moe_cache::miss_overlap_enabled() const {
    return pimpl->mo_on;
}

bool llama_moe_cache::hit_first_enabled() const {
    return pimpl->hf_on && !pimpl->hf_sync;
}

bool llama_moe_cache::copy_nosync(const ggml_tensor * src) const {
    return pimpl->copy_nosync(src);
}

void llama_moe_cache::miss_overlap_refuse(const char * reason) {
    pimpl->miss_overlap_refuse(reason);
}

bool llama_moe_cache::miss_overlap_split(int32_t il, int64_t n_tokens) const {
    return pimpl->miss_overlap_split(il, n_tokens);
}

ggml_tensor * llama_moe_cache::get_slot_map2(int32_t il) const {
    return pimpl->get_slot_map2(il);
}

ggml_tensor * llama_moe_cache::miss_overlap_gate(ggml_context * ctx, ggml_tensor * ids, int32_t il) const {
    return pimpl->miss_overlap_gate(ctx, ids, il);
}

void llama_moe_cache::set_copy_mark_hook(void (*fn)(void * ud, ggml_backend_t backend, int which), void * ud) {
    pimpl->copy_mark_fn = fn;
    pimpl->copy_mark_ud = ud;
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

int llama_moe_cache::timing_mode() const {
    return pimpl->timing_mode;
}

void llama_moe_cache::timing_graph_begin() {
    pimpl->graph_used = false;
}

bool llama_moe_cache::timing_graph_used() const {
    return pimpl->graph_used;
}

void llama_moe_cache::timing_graph_end(int64_t ns) {
    pimpl->tm.graph += ns;
    pimpl->tm.graphs++;
}

bool llama_moe_cache::predict_layer(int32_t il, int64_t n_tokens) const {
    return pimpl->predict_layer(il, n_tokens);
}

void llama_moe_cache::predict_begin() {
    if (pimpl->predict) {
        pimpl->pred_gen++;
    }
}

void llama_moe_cache::predict_add(int32_t il, int32_t p, const int32_t * ids) {
    pimpl->predict_add(il, p, ids);
}

int llama_moe_cache::prefetch_mode(int32_t il, int64_t n_tokens) const {
    return pimpl->prefetch_mode(il, n_tokens);
}

int llama_moe_cache::prefetch_k() const {
    return pimpl->pf_k;
}

void llama_moe_cache::prefetch_set_ids(int32_t il, ggml_tensor * ids) {
    if (pimpl->pf_mode != 0 && il >= 0 && il < (int32_t) pimpl->pf_ids_t.size()) {
        pimpl->pf_ids_t[il] = ids;
    }
    // AMB_MOE_LOOKAHEAD: read back by ids_post_split, with the real ids of layer il - 1
    if (pimpl->la_on && il >= 0 && il < (int32_t) pimpl->la_ids_t.size()) {
        pimpl->la_set_cnt++;
        if (pimpl->la_ids_t[il] != nullptr) {
            pimpl->la_set_dup++;
        }
        pimpl->la_ids_t[il] = ids;
    }
}

void llama_moe_cache::prefetch_clear() {
    if (pimpl->la_on && pimpl->la_set_cnt > 0 && pimpl->la_set_cnt != pimpl->la_set_logged) {
        // C19: logged when the count changes (once per graph shape): one predictor per layer means dup = 0
        LLAMA_LOG_INFO("llama_moe_cache: amb_moe_lookahead: C19: predictor tensors registered in the last graph = %d "
            "(layers registered twice = %d)\n", pimpl->la_set_cnt, pimpl->la_set_dup);
        pimpl->la_set_logged = pimpl->la_set_cnt;
    }
    pimpl->la_set_cnt = 0;
    pimpl->la_set_dup = 0;
    std::fill(pimpl->pf_ids_t.begin(), pimpl->pf_ids_t.end(), nullptr);
    std::fill(pimpl->la_ids_t.begin(), pimpl->la_ids_t.end(), nullptr);
}

bool llama_moe_cache::prefetch_ids_wanted() const {
    return pimpl->pf_mode != 0 || pimpl->la_on;
}

int llama_moe_cache::lookahead_mode(int32_t il, int64_t n_tokens) const {
    return pimpl->lookahead_mode(il, n_tokens);
}

int llama_moe_cache::lookahead_k() const {
    return pimpl->la_k;
}

bool llama_moe_cache::lookahead_rank() const {
    return pimpl->la_rank;
}

bool llama_moe_cache::lookahead_p4_fast() const {
    return pimpl->la_p4_fast;
}

void llama_moe_cache::lookahead_drain_ubatch(int64_t n_tokens) {
    if (pimpl->la_on && n_tokens > pimpl->la_drain_above) {
        pimpl->lookahead_drain(n_tokens > 8 ? "ubatch > 8 tokens" : "ubatch above the MMVQ batch limit of an expert type");
    }
}

void llama_moe_cache::lookahead_drain(const char * reason) {
    pimpl->lookahead_drain(reason);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}
