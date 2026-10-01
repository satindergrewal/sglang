# P3 prefill sweep: grouped v2 at large pair counts (the serve's prefill
# shapes: M tokens x topk8 pairs, ws rows=16 chunks=4), splits sweep, plus
# the dense v3 shape at M=256/2048 for reference.
import json, os, time, torch
from safetensors import safe_open
torch.ops.load_library("/mnt/nvme0/work-exl3/build_g38/exl3_ops_patch_g38/exl3_ops_patch_g38.so")
GL = torch.ops.sgl_exl3_grouped.grouped_linear
ART = "/mnt/nvme0/work-exl3/mimo375/out375"
wm = json.load(open(ART + "/model.safetensors.index.json"))["weight_map"]
dev = "cuda"
name0 = "model.layers.47.mlp.experts.0.up_proj.trellis"
sv0 = name0.rsplit(".", 1)[0] + ".svh"
panel = safe_open(ART + "/" + wm[name0], framework="pt", device=dev).get_tensor(name0).contiguous()
svh0 = safe_open(ART + "/" + wm[sv0], framework="pt", device=dev).get_tensor(sv0).contiguous().to(torch.half)
k16, n16, wb = panel.shape
K, N = k16 * 16, n16 * 16
E = 8
ptrs, sptrs, bptrs = [panel.data_ptr()], [svh0.data_ptr()], [0]
for _ in range(E - 1):
    t = torch.randint(0, 65536, (k16, n16, wb), dtype=torch.uint16, device=dev)
    sv = torch.randn(n16, dtype=torch.half, device=dev) * 0.01
    ptrs.append(t.data_ptr()); sptrs.append(sv.data_ptr()); bptrs.append(0)
mk = lambda v: torch.tensor(v, dtype=torch.int64, device=dev)
ptrs, sptrs, bptrs = mk(ptrs), mk(sptrs), mk(bptrs)
cb_blk = N // 128
CH = 4

for P in (256, 1024, 2048):
    per = P // E
    counts = torch.zeros(E + 1, dtype=torch.int32, device=dev)
    counts[:E] = per
    offsets = torch.zeros(E + 2, dtype=torch.int64, device=dev)
    offsets[1:E+1] = torch.arange(1, E+1, dtype=torch.int64, device=dev) * per
    offsets[E + 1] = P
    x = torch.randn(P, K, dtype=torch.half, device=dev)
    out = torch.empty(P, N, dtype=torch.half, device=dev)
    for splits in (1, 2, 4, 8):
        if (k16 + splits - 1) // splits < 8:
            continue
        ws = torch.zeros((cb_blk, splits, E * CH, 16, 128), dtype=torch.float32, device=dev)
        cnt = torch.zeros((E * cb_blk, CH), dtype=torch.int32, device=dev)
        for _ in range(3):
            GL(x, ptrs, sptrs, bptrs, counts, offsets, ws, cnt, 2, splits, CH, wb // 16, 0, out)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(20):
            GL(x, ptrs, sptrs, bptrs, counts, offsets, ws, cnt, 2, splits, CH, wb // 16, 0, out)
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) / 20 * 1e6
        print(f"P={P} splits={splits}: {t:.0f}us ({K*N*wb*2*E/(t*1e-6)/1e12:.2f} TB/s panel traffic)")
print("P3 GROUPED PREFILL SWEEP DONE")
