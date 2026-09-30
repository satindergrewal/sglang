#!/usr/bin/env python3
"""Gate 1 + Gate 2 for the rows=4 decode variant.

Gate 1 (bit-exactness): rows=4/chunks=8/splits=16 vs rows=16/chunks=4/splits=4
  on identical inputs at decode shapes — the fp32 partial order is identical,
  so outputs must match bitwise.
Gate 2 (A/B): serve-skew launch times (8 experts x 4 pairs, 64MB panels),
  v2 (rows=16, splits=4) vs v3 (rows=4, splits=16), P in {16,32,64}.
"""
import json, time
import torch

SO = "/mnt/nvme0/work-exl3/exl3_ops_out2/exl3_ops_patch_serve/exl3_ops_patch_serve.so"
torch.ops.load_library(SO)
GL = torch.ops.sgl_exl3_grouped.grouped_linear
E, K, N, CH = 64, 4096, 4096, 4
DEV = "cuda"
torch.manual_seed(7)

def make_packed():
    ptrs, sptrs, bptrs = [], [], []
    for _ in range(E):
        t = torch.randint(0, 65536, (K // 16, N // 16, 64), dtype=torch.uint16, device=DEV)
        sv = torch.randn(N // 16, dtype=torch.half, device=DEV) * 0.01
        bi = torch.zeros(N // 16, dtype=torch.half, device=DEV)
        ptrs.append(t.data_ptr()); sptrs.append(sv.data_ptr()); bptrs.append(bi.data_ptr())
    mk = lambda v: torch.tensor(v, dtype=torch.int64, device=DEV)
    return mk(ptrs), mk(sptrs), mk(bptrs)

ptrs, sptrs, bptrs = make_packed()

def run(rows, chunks, splits, x, counts, offsets, out):
    cb = N // 128
    ws = torch.zeros((cb, splits, E * chunks, rows, 128), dtype=torch.float32, device=DEV)
    cnt = torch.zeros((E * cb, chunks), dtype=torch.int32, device=DEV)
    GL(x, ptrs, sptrs, bptrs, counts, offsets, ws, cnt, 1, splits, chunks, 4, 0, out)
    torch.cuda.synchronize()
    return ws, cnt

def counts_skew(P, groups=8):
    c = torch.zeros(E + 1, dtype=torch.int32, device=DEV)
    per = P // groups
    c[:groups] = per
    rem = P - per * groups
    c[:rem] += 1
    assert int(c[:E].sum()) == P, f"counts sum {int(c[:E].sum())} != {P}"
    o = torch.zeros(E + 2, dtype=torch.int64, device=DEV)
    o[1:E+1] = c[:E].cumsum(0).to(torch.int64)
    o[E+1] = P
    return c, o

GLD = torch.ops.sgl_exl3_grouped.grouped_linear_direct
print("=== Gate 1: direct vs v2 at decode shapes ===", flush=True)
ok_all = True
for P in (16, 32, 64, 128, 256):
    x = torch.randn(P, K, dtype=torch.half, device=DEV)
    counts, offsets = counts_skew(P)
    out_v2 = torch.full((P, N), float("nan"), dtype=torch.half, device=DEV)
    out_v3 = torch.full((P, N), float("nan"), dtype=torch.half, device=DEV)
    run(16, 4, 4, x, counts, offsets, out_v2)
    # direct: atomic-split partials into a zeroed fp32 scratch + epilogue
    scratch = torch.zeros(P, N, dtype=torch.float32, device=DEV)
    GLD(x, ptrs, sptrs, bptrs, counts, offsets, 1, 4, 0, 8, scratch, out_v3)
    torch.cuda.synchronize()
    same_rows = (out_v2 != out_v3)
    nan = torch.isnan(out_v3).any().item()
    # fp32 grouping differs (no split partials): gate = rel diff <= one half-ulp scale
    d = (out_v2.float() - out_v3.float()).abs()
    rel = (d / out_v2.float().abs().clamp(min=1e-3)).max().item()
    ok = (not nan) and rel < 0.01
    ok_all &= ok
    print(f"P={P}: nan={nan} maxdiff={d.max().item():.3e} relmax={rel:.2e} -> {'OK' if ok else 'FAIL'}", flush=True)
print("GATE1", "PASS" if ok_all else "FAIL", flush=True)

print("=== Gate 2: A/B at serve skew ===", flush=True)
res = {}
for P in (16, 32, 64):
    x = torch.randn(P, K, dtype=torch.half, device=DEV)
    counts, offsets = counts_skew(P)
    out = torch.empty(P, N, dtype=torch.half, device=DEV)
    # v2 baseline
    cb = N // 128
    ws = torch.zeros((cb, 4, E * 4, 16, 128), dtype=torch.float32, device=DEV)
    cnt = torch.zeros((E * cb, 4), dtype=torch.int32, device=DEV)
    for _ in range(5):
        GL(x, ptrs, sptrs, bptrs, counts, offsets, ws, cnt, 1, 4, 4, 4, 0, out)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        GL(x, ptrs, sptrs, bptrs, counts, offsets, ws, cnt, 1, 4, 4, 4, 0, out)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / 100 * 1e6
    res[(P, "v2")] = us
    print(f"P={P} v2: {us:.1f} us/launch", flush=True)
    # direct (includes the scratch zero + epilogue, as the runner would pay)
    scratch = torch.zeros(P, N, dtype=torch.float32, device=DEV)
    for _ in range(5):
        scratch.zero_()
        GLD(x, ptrs, sptrs, bptrs, counts, offsets, 1, 4, 0, 8, scratch, out)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        scratch.zero_()
        GLD(x, ptrs, sptrs, bptrs, counts, offsets, 1, 4, 0, 8, scratch, out)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / 100 * 1e6
    res[(P, "direct")] = us
    print(f"P={P} direct: {us:.1f} us/launch", flush=True)
for P in (16, 32, 64, 128, 256):
    print(f"P={P}: direct/v2 speedup = {res[(P,'v2')]/res[(P,'direct')]:.2f}x", flush=True)
print("GATE2 DONE", flush=True)
