// AMB_H2D_ZC test: the zero-copy H2D path of the CUDA backend (ggml_backend_cuda_memcpy_batch_async and
// ggml_backend_tensor_set_async) must write exactly the bytes a memcpy would, touch nothing outside the
// destinations, take the copy kernel for pinned sources only, fall back for pageable sources and leave
// device-to-host batches to the DMA path.
//
// The test sets AMB_H2D_ZC=1 and AMB_H2D_ZC_STATS=1 itself (one stats line per call, read through the log callback,
// tells which path a call took). Run: test-h2d-zc [device]   (exit code 0 = pass). Optional: AMB_H2D_ZC_CHECK=1 in
// the environment also verifies every zero-copy against a DMA copy on the device.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

// counts of the stats lines seen since the last reset
struct zc_counts {
    unsigned long long zc_calls = 0, not_pinned = 0, skipped = 0, other_fb = 0;
    int lines = 0;
};

static zc_counts g_counts;
static bool      g_verbose = false;

static unsigned long long field(const char * text, const char * key) {
    const char * p = strstr(text, key);
    if (p == nullptr) {
        return 0;
    }
    return strtoull(p + strlen(key), nullptr, 10);
}

static void log_cb(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (strstr(text, "amb_h2d_zc: periodic") != nullptr) {
        g_counts.lines++;
        g_counts.zc_calls   += field(text, "zc_calls = ");
        g_counts.not_pinned += field(text, "not_pinned = ");
        g_counts.skipped    += field(text, "skipped_not_h2d = ");
        g_counts.other_fb   += field(text, "capturing = ") + field(text, "not_compute_stream = ") +
                               field(text, "too_big = ") + field(text, "pending_err = ") + field(text, "launch_err = ");
        if (g_verbose) {
            fputs(text, stderr);
        }
        return;
    }
    if (level >= GGML_LOG_LEVEL_WARN || g_verbose) {
        fputs(text, stderr);
    }
}

enum class expect { zc, not_pinned, skipped, none };

static const char * expect_name(expect e) {
    switch (e) {
        case expect::zc:         return "zero-copy";
        case expect::not_pinned: return "not_pinned fallback";
        case expect::skipped:    return "skipped_not_h2d";
        default:                 return "no call";
    }
}

static bool check_path(expect e) {
    const zc_counts c = g_counts;
    switch (e) {
        case expect::zc:         return c.lines == 1 && c.zc_calls == 1 && c.not_pinned == 0 && c.skipped == 0 && c.other_fb == 0;
        case expect::not_pinned: return c.lines == 1 && c.zc_calls == 0 && c.not_pinned == 1 && c.skipped == 0 && c.other_fb == 0;
        case expect::skipped:    return c.lines == 1 && c.zc_calls == 0 && c.not_pinned == 0 && c.skipped == 1 && c.other_fb == 0;
        default:                 return c.lines == 0;
    }
}

static const size_t SIZES[] = { 0, 1, 3, 15, 16, 17, 4095, (1u << 20) + 7 };
static const size_t N_SIZES = sizeof(SIZES)/sizeof(SIZES[0]);
static const size_t MISALIGN[][2] = { {0, 0}, {1, 3}, {3, 1}, {8, 4}, {5, 0}, {0, 7} }; // {src offset, dst offset}
static const size_t N_MIS   = sizeof(MISALIGN)/sizeof(MISALIGN[0]);

static const size_t BUF = 64u << 20;
static const uint8_t FILL = 0xCD;

struct env {
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t dev_buf = nullptr;
    ggml_tensor *         dev_t   = nullptr; // I8 tensor covering the whole device buffer
    ggml_context *        ctx     = nullptr;
    uint8_t *             dev     = nullptr; // device base
    uint8_t *             pinned  = nullptr; // pinned host (cuda host buffer)
    ggml_backend_buffer_t pinned_buf = nullptr;
    std::vector<uint8_t>  pageable;
    std::vector<uint8_t>  expected;
    std::vector<uint8_t>  got;
    std::mt19937          rng{1234};
};

struct copy_desc {
    size_t src_off, dst_off, size;
};

// lays out count copies with sizes cycled from SIZES starting at size_idx and misalignments cycled from MISALIGN,
// disjoint in source and destination
static std::vector<copy_desc> layout(size_t count, size_t size_idx, size_t mis_idx) {
    std::vector<copy_desc> v;
    size_t so = 0, d = 0;
    for (size_t k = 0; k < count; ++k) {
        const size_t s  = SIZES[(size_idx + k) % N_SIZES];
        const size_t* m = MISALIGN[(mis_idx + k) % N_MIS];
        copy_desc c = { so + m[0], d + m[1], s };
        v.push_back(c);
        so = c.src_off + s + 32;
        d  = c.dst_off + s + 32;
    }
    return v;
}

static int g_fail = 0, g_pass = 0;

static void report(bool ok, const std::string & name, const char * what) {
    if (ok) {
        g_pass++;
    } else {
        g_fail++;
        fprintf(stderr, "FAIL: %s: %s\n", name.c_str(), what);
    }
}

// H2D: batch (or set_tensor_async when use_set) from pinned or pageable source, compared against memcpy
// src_pinned: whether the source is pinned and mapped (the zero-copy path is expected), else a fallback is expected
static void run_h2d(env & e, const std::vector<copy_desc> & cs, uint8_t * src, bool pinned, bool use_set, const std::string & name) {
    size_t src_end = 0, dst_end = 0, total = 0;
    for (const copy_desc & c : cs) {
        src_end = std::max(src_end, c.src_off + c.size);
        dst_end = std::max(dst_end, c.dst_off + c.size);
        total  += c.size;
    }
    for (size_t i = 0; i < src_end; ++i) {
        src[i] = (uint8_t) e.rng();
    }
    ggml_backend_buffer_clear(e.dev_buf, FILL);
    ggml_backend_synchronize(e.backend);

    e.expected.assign(dst_end + 64, FILL);
    for (const copy_desc & c : cs) {
        memcpy(e.expected.data() + c.dst_off, src + c.src_off, c.size);
    }

    g_counts = zc_counts();
    expect ex;
    if (use_set) {
        for (const copy_desc & c : cs) {
            ggml_backend_tensor_set_async(e.backend, e.dev_t, src + c.src_off, c.dst_off, c.size);
        }
        // set_tensor_async does not enter the zero-copy code for size 0
        ex = total == 0 ? expect::none : pinned ? expect::zc : expect::not_pinned;
    } else {
        std::vector<void *> dsts;
        std::vector<const void *> srcs;
        std::vector<size_t> sizes;
        for (const copy_desc & c : cs) {
            dsts.push_back(e.dev + c.dst_off);
            srcs.push_back(src + c.src_off);
            sizes.push_back(c.size);
        }
        if (!ggml_backend_cuda_memcpy_batch_async(e.backend, dsts.data(), srcs.data(), sizes.data(), cs.size())) {
            report(false, name, "ggml_backend_cuda_memcpy_batch_async returned false");
            return;
        }
        // a call whose copies are all empty has no source to classify and takes the kernel path (launching nothing)
        ex = (pinned || total == 0) ? expect::zc : expect::not_pinned;
    }
    ggml_backend_synchronize(e.backend);

    e.got.assign(e.expected.size(), 0);
    ggml_backend_tensor_get(e.dev_t, e.got.data(), 0, e.got.size());
    report(memcmp(e.got.data(), e.expected.data(), e.got.size()) == 0, name, "bytes differ from memcpy (or bytes outside the destinations changed)");
    if (!check_path(ex)) {
        const std::string msg = std::string("expected path ") + expect_name(ex) + ", stats: zc_calls=" + std::to_string(g_counts.zc_calls) +
            " not_pinned=" + std::to_string(g_counts.not_pinned) + " skipped=" + std::to_string(g_counts.skipped) +
            " other_fallbacks=" + std::to_string(g_counts.other_fb) + " lines=" + std::to_string(g_counts.lines);
        report(false, name, msg.c_str());
    } else {
        g_pass++;
    }
}

// D2H batch (device -> pinned or pageable host): must not take the kernel path, bytes must match
static void run_d2h(env & e, const std::vector<copy_desc> & cs, bool pinned, const std::string & name) {
    uint8_t * dst = pinned ? e.pinned : e.pageable.data();
    size_t src_end = 0, dst_end = 0;
    for (const copy_desc & c : cs) {
        src_end = std::max(src_end, c.dst_off + c.size); // on the device the source uses the dst_off layout
        dst_end = std::max(dst_end, c.src_off + c.size);
    }
    std::vector<uint8_t> devdata(src_end);
    for (uint8_t & b : devdata) {
        b = (uint8_t) e.rng();
    }
    ggml_backend_tensor_set(e.dev_t, devdata.data(), 0, devdata.size());
    memset(dst, FILL, dst_end + 64);
    e.expected.assign(dst, dst + dst_end + 64);
    for (const copy_desc & c : cs) {
        memcpy(e.expected.data() + c.src_off, devdata.data() + c.dst_off, c.size);
    }

    std::vector<void *> dsts;
    std::vector<const void *> srcs;
    std::vector<size_t> sizes;
    for (const copy_desc & c : cs) {
        dsts.push_back(dst + c.src_off);
        srcs.push_back(e.dev + c.dst_off);
        sizes.push_back(c.size);
    }
    g_counts = zc_counts();
    const bool issued = ggml_backend_cuda_memcpy_batch_async(e.backend, dsts.data(), srcs.data(), sizes.data(), cs.size());
    ggml_backend_synchronize(e.backend);
    if (!check_path(expect::skipped)) {
        const std::string msg = std::string("expected skipped_not_h2d, stats: zc_calls=") + std::to_string(g_counts.zc_calls) +
            " not_pinned=" + std::to_string(g_counts.not_pinned) + " skipped=" + std::to_string(g_counts.skipped);
        report(false, name, msg.c_str());
    } else {
        g_pass++;
    }
    if (!issued) {
        // the DMA batch itself declined (old behaviour for this kind of call): nothing to compare
        fprintf(stderr, "note: %s: cudaMemcpyBatchAsync declined the D2H batch (DMA path, unchanged behaviour)\n", name.c_str());
        return;
    }
    report(memcmp(dst, e.expected.data(), e.expected.size()) == 0, name, "D2H bytes differ from memcpy");
}

int main(int argc, char ** argv) {
    set_env("AMB_H2D_ZC", "1");
    set_env("AMB_H2D_ZC_STATS", "1");
    g_verbose = getenv("TEST_H2D_ZC_VERBOSE") != nullptr;
    ggml_log_set(log_cb, nullptr);

    const int device = argc > 1 ? atoi(argv[1]) : 0;
    env e;
    e.backend = ggml_backend_cuda_init(device);
    if (e.backend == nullptr) {
        fprintf(stderr, "no CUDA device %d\n", device);
        return 1;
    }

    // a graph has to run first: the zero-copy path is only taken on a compute stream
    {
        ggml_init_params ip = { 4*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * gctx = ggml_init(ip);
        ggml_tensor * a = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 256);
        ggml_tensor * b = ggml_scale(gctx, a, 2.0f);
        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, b);
        ggml_backend_buffer_t gbuf = ggml_backend_alloc_ctx_tensors(gctx, e.backend);
        ggml_backend_buffer_clear(gbuf, 0);
        if (ggml_backend_graph_compute(e.backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed\n");
            return 1;
        }
        ggml_backend_synchronize(e.backend);
        ggml_backend_buffer_free(gbuf);
        ggml_free(gctx);
    }

    e.dev_buf = ggml_backend_alloc_buffer(e.backend, BUF + 4096);
    e.dev     = (uint8_t *) ggml_backend_buffer_get_base(e.dev_buf);
    {
        ggml_init_params ip = { ggml_tensor_overhead(), nullptr, true };
        e.ctx   = ggml_init(ip);
        e.dev_t = ggml_new_tensor_1d(e.ctx, GGML_TYPE_I8, (int64_t) BUF);
        ggml_backend_tensor_alloc(e.dev_buf, e.dev_t, e.dev);
    }
    e.pinned_buf = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), BUF + 4096);
    if (e.pinned_buf == nullptr || !ggml_backend_buffer_is_host(e.pinned_buf)) {
        fprintf(stderr, "pinned host buffer allocation failed\n");
        return 1;
    }
    e.pinned = (uint8_t *) ggml_backend_buffer_get_base(e.pinned_buf);
    e.pageable.resize(BUF + 4096);

    for (int src_kind = 0; src_kind < 2; ++src_kind) {
        const bool pinned = src_kind == 0;
        const char * kn = pinned ? "pinned" : "pageable";
        // single copies: every size x every misalignment, by batch and by set_tensor_async
        for (size_t si = 0; si < N_SIZES; ++si) {
            for (size_t mi = 0; mi < N_MIS; ++mi) {
                const auto cs = layout(1, si, mi);
                char name[128];
                snprintf(name, sizeof(name), "%s batch count=1 size=%zu src+%zu dst+%zu", kn, SIZES[si], MISALIGN[mi][0], MISALIGN[mi][1]);
                run_h2d(e, cs, pinned ? e.pinned : e.pageable.data(), pinned, false, name);
                snprintf(name, sizeof(name), "%s set_tensor_async size=%zu src+%zu dst+%zu", kn, SIZES[si], MISALIGN[mi][0], MISALIGN[mi][1]);
                run_h2d(e, cs, pinned ? e.pinned : e.pageable.data(), pinned, true, name);
            }
        }
        // multi-copy batches: 64 (one launch), 65 and 200 (several launches), mixed sizes and misalignments
        for (size_t count : { (size_t) 64, (size_t) 65, (size_t) 200 }) {
            for (size_t start = 0; start < 3; ++start) {
                const auto cs = layout(count, start, start);
                char name[128];
                snprintf(name, sizeof(name), "%s batch count=%zu start=%zu", kn, count, start);
                run_h2d(e, cs, pinned ? e.pinned : e.pageable.data(), pinned, false, name);
            }
        }
        // D2H readbacks through the batch function
        for (size_t count : { (size_t) 1, (size_t) 65 }) {
            const auto cs = layout(count, 1, 1);
            char name[128];
            snprintf(name, sizeof(name), "D2H batch to %s count=%zu", kn, count);
            run_d2h(e, cs, pinned, name);
        }
    }

    // a source in host memory registered through ggml_backend_cuda_register_host_buffer (mapped when the switch is on)
    {
        std::vector<uint8_t> reg(BUF);
        if (ggml_backend_cuda_register_host_buffer(reg.data(), reg.size())) {
            const auto cs = layout(65, 2, 3);
            run_h2d(e, cs, reg.data(), true, false, "registered host batch count=65");
            run_h2d(e, layout(1, 7, 1), reg.data(), true, true, "registered host set_tensor_async size=1MiB+7");
            ggml_backend_cuda_unregister_host_buffer(reg.data());
        } else {
            fprintf(stderr, "note: host buffer registration disabled or failed, registered-source case skipped\n");
        }
    }

    ggml_backend_buffer_free(e.pinned_buf);
    ggml_backend_buffer_free(e.dev_buf);
    ggml_free(e.ctx);
    ggml_backend_free(e.backend);

    printf("test-h2d-zc: %d checks passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
