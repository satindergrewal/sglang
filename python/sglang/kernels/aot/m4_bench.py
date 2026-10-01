# M4-decisive microbench at decode skew (8 experts x 4 pairs, real 4-bit panel):
#   (a) direct_b (the captured-decode champion, atomic scratch, no ws)
#   (b) v2 ws rows=16 chunks=4 splits=4 (the eager default)
#   (c) v2 ws rows=4 chunks=16 splits=8/16/32 (the M4 form: (4,128) slabs,
#       deep splits; capacity 64 pairs/expert covers worst-case skew)
import json, os, time, torch
from safetensors import safe_open
torch.ops.load_library("/mnt/nvme0/work-exl3/build_g38/exl3_ops_patch_g38/exl3_ops_patch_g38.so")
GL = torch.ops.sgl_exl3_grouped.grouped_linear
GLD_B = torch.ops.sgl_exl3_grouped.grouped_linear_direct_b
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

counts = torch.zeros(E + 1, dtype=torch.int32, device=dev)
counts[:E] = 4
offsets = torch.zeros(E + 2, dtype=torch.int64, device=dev)
offsets[1:E+1] = torch.arange(1, E+1, dtype=torch.int64, device=dev) * 4
offsets[E + 1] = 32
x = torch.randn(32, K, dtype=torch.half, device=dev)
out = torch.empty(32, N, dtype=torch.half, device=dev)

def bench(fn, reps=100):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e6

# (a) direct_b, splits=8
os.environ["EXL3_PANEL_LAYOUT"] = "B"
scratch = torch.zeros(32, N, dtype=torch.float32, device=dev)
def f_direct():
    scratch.zero_()
    GLD_B(x, ptrs, sptrs, bptrs, counts, offsets, 2, wb // 16, 0, 8, scratch, out)
print(f"direct_b s=8: {bench(f_direct):.1f}us")

# (b) v2 rows=16 chunks=4 splits=4
ws16 = torch.zeros((cb_blk, 4, E * 4, 16, 128), dtype=torch.float32, device=dev)
cnt16 = torch.zeros((E * cb_blk, 4), dtype=torch.int32, device=dev)
def f_v2_16():
    GL(x, ptrs, sptrs, bptrs, counts, offsets, ws16, cnt16, 2, 4, 4, wb // 16, 0, out)
print(f"v2 r16 c4 s4: {bench(f_v2_16):.1f}us")

# (c) v2 rows=4 chunks=16, splits sweep
for s in (8, 16, 32):
    ws4 = torch.zeros((cb_blk, s, E * 16, 4, 128), dtype=torch.float32, device=dev)
    cnt4 = torch.zeros((E * cb_blk, 16), dtype=torch.int32, device=dev)
    def f_v2_4():
        GL(x, ptrs, sptrs, bptrs, counts, offsets, ws4, cnt4, 2, s, 16, wb // 16, 0, out)
    print(f"v2 r4  c16 s{s}: {bench(f_v2_4):.1f}us")
    del ws4, cnt4
# prefill pair counts on the eager ws path: rows=16 c4 vs rows=4 c16
for P in (1024, 2048):
    per = P // E
    c2 = torch.zeros(E + 1, dtype=torch.int32, device=dev)
    c2[:E] = per
    o2 = torch.zeros(E + 2, dtype=torch.int64, device=dev)
    o2[1:E+1] = torch.arange(1, E+1, dtype=torch.int64, device=dev) * per
    o2[E + 1] = P
    xp = torch.randn(P, K, dtype=torch.half, device=dev)
    outp = torch.empty(P, N, dtype=torch.half, device=dev)
    ws16p = torch.zeros((cb_blk, 4, E * 4, 16, 128), dtype=torch.float32, device=dev)
    cnt16p = torch.zeros((E * cb_blk, 4), dtype=torch.int32, device=dev)
    def f16():
        GL(xp, ptrs, sptrs, bptrs, c2, o2, ws16p, cnt16p, 2, 4, 4, wb // 16, 0, outp)
    r16 = bench(f16, 20)
    ws4p = torch.zeros((cb_blk, 8, E * 16, 4, 128), dtype=torch.float32, device=dev)
    cnt4p = torch.zeros((E * cb_blk, 16), dtype=torch.int32, device=dev)
    def f4():
        GL(xp, ptrs, sptrs, bptrs, c2, o2, ws4p, cnt4p, 2, 8, 16, wb // 16, 0, outp)
    r4 = bench(f4, 20)
    print(f"P={P}: r16/c4/s4 {r16:.0f}us  vs  r4/c16/s8 {r4:.0f}us")
    del ws16p, cnt16p, ws4p, cnt4p
os.environ["EXL3_PANEL_LAYOUT"] = "A"
print("M4 DECISIVE BENCH DONE")
