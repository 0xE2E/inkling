"""Replay an LLAMA_MOE_CACHE_TRACE (MCT1) file through cache policies.
Compares keys by effective layer (stock: il, e) against keys by physical block (dedup of looped passes: loop_layer(il), e)
at the same slot budget. Policies: LRU and Belady (best possible eviction order, an upper bound).

  python mct_sim.py <trace> <cache MiB> [core_start core_end n_pass]   (defaults 6 18 3 for Inkling-2-Mini)
"""
import sys, struct, collections, heapq

def read(path):
    b = open(path, "rb").read(); o = 0
    def u32():
        nonlocal o; v = struct.unpack_from("<I", b, o)[0]; o += 4; return v
    def u64():
        nonlocal o; v = struct.unpack_from("<Q", b, o)[0]; o += 8; return v
    def f64():
        nonlocal o; v = struct.unpack_from("<d", b, o)[0]; o += 8; return v
    assert b[:4] == b"MCT1"; o = 4
    n_layers, n_used, n_groups = u32(), u32(), u32(); groups = []
    for _ in range(n_groups):
        gid, n_exp, nl = u32(), u32(), u32(); ls = [u32() for _ in range(nl)]
        split, ssum, gh, dh, align = f64(), f64(), u64(), u64(), u64()
        nr = u32(); nb2 = [u64() for _ in range(nr)]
        groups.append(dict(id=gid, n_exp=n_exp, layers=ls, host=gh, dev_host=dh, slot_bytes=sum(nb2), split=split, ssum=ssum))
    recs = []
    while o < len(b):
        il, nu, n = struct.unpack_from("<iii", b, o); o += 12
        ids = struct.unpack_from("<%di" % n, b, o); o += 4 * n
        recs.append((il, nu, ids))
    return n_layers, n_used, groups, recs

def phys(il, cs, ce, npass):
    nc = ce - cs
    if il < cs: return il
    if il < cs + nc * npass: return cs + (il - cs) % nc
    return il - nc * (npass - 1)

def sim(stream, slots):
    # stream: list of sets of keys per lookup (a lookup's keys are all needed at once)
    lru = collections.OrderedDict(); hit = miss = 0
    for ks in stream:
        for k in ks:
            if k in lru: hit += 1; lru.move_to_end(k)
            else:
                miss += 1; lru[k] = 1
        while len(lru) > slots:
            # never evict a key of the current lookup
            for k in lru:
                if k not in ks: del lru[k]; break
            else: break
    # Belady
    nxt = {}; future = [None] * len(stream)
    for t in range(len(stream) - 1, -1, -1):
        future[t] = {k: nxt.get(k, 1 << 60) for k in stream[t]}
        for k in stream[t]: nxt[k] = t
    cache = {}; heap = []; bh = bm = 0
    for t, ks in enumerate(stream):
        for k in ks:
            if k in cache: bh += 1
            else: bm += 1
            cache[k] = future[t][k]; heapq.heappush(heap, (-future[t][k], k))
        keep = []
        while len(cache) > slots and heap:
            nu, k = heapq.heappop(heap)
            if k not in cache or cache[k] != -nu: continue          # stale entry
            if k in ks: keep.append((nu, k)); continue              # needed by this lookup
            del cache[k]
        for x in keep: heapq.heappush(heap, x)
    return hit, miss, bh, bm

def main():
    path, mib = sys.argv[1], float(sys.argv[2])
    cs, ce, npass = (int(x) for x in sys.argv[3:6]) if len(sys.argv) > 5 else (6, 18, 3)
    n_layers, n_used, groups, recs = read(path)
    print(f"layers {n_layers} n_used {n_used} lookups {len(recs)} groups {len(groups)}")
    for g in groups:
        budget = mib * 2**20 * g["split"] / g["ssum"] * g["host"] / g["dev_host"]
        slots = int(budget // g["slot_bytes"]); L = set(g["layers"])
        P = sorted({phys(il, cs, ce, npass) for il in L})
        print(f"\ngroup {g['id']}: {len(L)} layers ({len(P)} physical), slot {g['slot_bytes']/2**20:.2f} MiB, slots {slots} "
              f"(stock max {len(L)*g['n_exp']}, dedup max {len(P)*g['n_exp']})")
        for name, key in (("stock (il,e)", lambda il, e: il * 1000 + e), ("dedup (phys,e)", lambda il, e: phys(il, cs, ce, npass) * 1000 + e)):
            stream = [frozenset(key(il, e) for e in ids if e >= 0) for il, nu, ids in recs if il in L]
            h, m, bh, bm = sim(stream, slots)
            print(f"  {name:15s} LRU hit {h/(h+m)*100:5.1f}%  misses {m:8d} ({m*g['slot_bytes']/2**30:7.2f} GiB)   "
                  f"Belady hit {bh/(bh+bm)*100:5.1f}%  misses {bm:8d}")
    # how often a looped pass routes to experts the previous pass of the same token used
    nc = ce - cs; last = {}; ov = collections.Counter(); tot = collections.Counter()
    for il, nu, ids in recs:
        last[il] = ids
        if cs + nc <= il < cs + nc * npass and len(ids) == nu:          # single-token lookups of pass 2 or 3
            prev = last.get(il - nc)
            if prev is not None and len(prev) == nu:
                p = (il - cs) // nc + 1
                ov[p] += len(set(ids) & set(prev)); tot[p] += nu
    for p in sorted(tot):
        print(f"pass {p}: share of experts also used by pass {p-1} of the same token: {ov[p]/tot[p]*100:.1f}%")
    print("(lookups per layer)", collections.Counter(il for il, _, _ in recs).most_common(3))

if __name__ == "__main__":
    main()
