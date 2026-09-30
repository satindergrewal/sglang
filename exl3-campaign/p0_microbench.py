#!/usr/bin/env python3
"""P0 per-kernel microbench: exact 3.75-shape grouped GEMM timings.

Measures exl3_grouped.grouped_linear and exl3 exl3_gemm (via the served
ops patch) at the 3.75 artifact's decode shapes, producing the per-kernel
table + effective weight-stream bandwidth that the attribution arm cites.
Run inside a serve image with /work/exl3_ops/exl3_ops_patch.so loaded
(EXL3_MOE_GROUPED=1) or load it explicitly below.

Shapes (3.75 artifact, TP2-EP2 local): E=64 local experts, K=4096 hidden,
gate/up N=4096 (2*I=2*2048), down N=4096 (out=hidden), I=2048.
P = pairs = bs*topk swept 16..512. bits: gate/up 4 (wb=64), down 3.5 (wb=56
half-rate family -> bits=3, half_k=1 convention per the runner).
"""
import json, sys, time
import torch

torch.ops.load_library("/work/exl3_ops/exl3_ops_patch.so")
GL = torch.ops.sgl_exl3_grouped.grouped_linear
E, K, NG, ND = 64, 4096, 4096, 4096
DEV = "cuda"
CH, SPLITS = 4, 4

def make_packed(n_out, bits, half_k):
    k16, n16 = K // 16, n_out // 16
    wb = bits * 16 if not half_k else bits * 16 + 8  # half-rate family sizing
    panels, svhs, biases, ptrs, sptrs, bptrs = [], [], [], [], [], []
    for _ in range(E):
        t = torch.randint(0, 65536, (k16, n16, wb), dtype=torch.uint16, device=DEV)
        sv = torch.randn(n16, dtype=torch.half, device=DEV) * 0.01
        bi = torch.zeros(n16, dtype=torch.half, device=DEV)
        panels.append(t); svhs.append(sv); biases.append(bi)
        ptrs.append(t.data_ptr()); sptrs.append(sv.data_ptr()); bptrs.append(bi.data_ptr())
    mk = lambda v: torch.tensor(v, dtype=torch.int64, device=DEV)
    return panels, mk(ptrs), mk(sptrs), mk(bptrs)

def ws_for(n_out):
    cb = n_out // 128
    return (torch.zeros((cb, SPLITS, E * CH, 16, 128), dtype=torch.float32, device=DEV),
            torch.zeros((E * cb, CH), dtype=torch.int32, device=DEV))

def run_case(P, bits, half_k, n_out, tag, iters=50):
    xp = torch.randn(P, K, dtype=torch.half, device=DEV)
    panels, ptrs, sptrs, bptrs = make_packed(n_out, bits, half_k)
    ws, cnt = ws_for(n_out)
    counts = torch.zeros(E + 1, dtype=torch.int32, device=DEV)
    if P >= E:
        counts[:E] = P // E
        counts[: P % E] += 1
    else:
        counts[:P] = 1
    offsets = torch.zeros(E + 2, dtype=torch.int64, device=DEV)
    offsets[1:] = counts.cumsum(0).to(torch.int64)
    out = torch.empty(P, n_out, dtype=torch.half, device=DEV)
    cb_code = 1 if bits == 4 else 0
    for _ in range(5):
        GL(xp, ptrs, sptrs, bptrs, counts, offsets, ws, cnt, cb_code, SPLITS, CH, bits, int(half_k), out)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        GL(xp, ptrs, sptrs, bptrs, counts, offsets, ws, cnt, cb_code, SPLITS, CH, bits, int(half_k), out)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / iters * 1e6
    weight_bytes = E * K * n_out * bits / 8
    gbps = weight_bytes / (us * 1e-6) / 1e9
    row = {"case": tag, "P": P, "bits": bits, "N": n_out, "us_per_launch": round(us, 2),
           "weight_MB": round(weight_bytes / 1e6, 1), "eff_GBps": round(gbps, 1)}
    print(json.dumps(row), flush=True)
    return row

def run_dense(bits, iters=50):
    # exl3 dense path: use the sgl_exl3_linear op via the primary patch if present
    try:
        LIN = torch.ops.sgl_kernel.sgl_exl3_linear
    except AttributeError:
        print(json.dumps({"case": "dense", "skipped": "op not loaded"}), flush=True)
        return
    k16, n16 = K // 16, NG // 16
    wb = bits * 16
    t = torch.randint(0, 65536, (k16, n16, wb), dtype=torch.uint16, device=DEV)
    sv = torch.randn(n16, dtype=torch.half, device=DEV) * 0.01
    bias = torch.zeros(n16, dtype=torch.half, device=DEV)
    suh = torch.randn(K, dtype=torch.half, device=DEV) * 0.01
    rows = [64, 256, 512]
    for M in rows:
        x = torch.randn(M, K, dtype=torch.half, device=DEV)
        xh = torch.empty_like(x)
        out = torch.empty(M, NG, dtype=torch.half, device=DEV)
        torch.ops.sgl_kernel.sgl_exl3_had_in(x, suh, xh)
        for _ in range(5):
            LIN(xh, t, sv, bias, bits, out)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            LIN(xh, t, sv, bias, bits, out)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / iters * 1e6
        wb_bytes = K * NG * bits / 8
        print(json.dumps({"case": "dense", "M": M, "bits": bits, "us": round(us, 2),
                          "weight_MB": round(wb_bytes / 1e6, 1),
                          "eff_GBps": round(wb_bytes / (us * 1e-6) / 1e9, 1)}), flush=True)

rows = []
for P in (16, 32, 64, 128, 256, 512):
    rows.append(run_case(P, 4, False, NG, "gate-4bit"))
for P in (16, 32, 64, 128, 256, 512):
    rows.append(run_case(P, 3, True, ND, "down-3bit-half"))
run_dense(5)
json.dump(rows, open("/tmp/p0_micro.json", "w"), indent=1)
print("P0 MICROBENCH DONE")
