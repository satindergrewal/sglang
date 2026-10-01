import torch, json, time
from safetensors import safe_open
torch.ops.load_library("/mnt/nvme0/work-exl3/exl3_ops_out2/exl3_ops_patch_serve/exl3_ops_patch_serve.so")
GLD  = torch.ops.sgl_exl3_grouped.grouped_linear_direct       # Phase-1 kernel (separate decode)
GLF  = torch.ops.sgl_exl3_grouped.grouped_linear_direct_f     # fused decode-into-fragment
E, K, N = 64, 4096, 2048
dev = "cuda"
wm = json.load(open("/art/model.safetensors.index.json"))["weight_map"]
name = "model.layers.47.mlp.experts.0.up_proj.trellis"
panel = safe_open("/art/" + wm[name], framework="pt", device=dev).get_tensor(name).contiguous()
svh = safe_open("/art/" + wm["model.layers.47.mlp.experts.0.up_proj.svh"], framework="pt", device=dev).get_tensor("model.layers.47.mlp.experts.0.up_proj.svh").contiguous().to(torch.half)
ptrs = []; sptrs = []; bptrs = []
for _ in range(E):
    bi = torch.zeros(N, dtype=torch.half, device=dev)
    ptrs.append(panel.data_ptr()); sptrs.append(svh.data_ptr()); bptrs.append(bi.data_ptr())
mk = lambda v: torch.tensor(v, dtype=torch.int64, device=dev)

print("=== Gate 1: bit-exactness — fused vs Phase-1 kernel, real panels ===")
ok_all = True
for P in (16, 32, 64, 128):
    x = torch.randn(P, K, dtype=torch.half, device=dev)
    counts = torch.zeros(E + 1, dtype=torch.int32, device=dev)
    per = P // 8; counts[:8] = per; counts[:P - per * 8] += 1
    offsets = torch.zeros(E + 2, dtype=torch.int64, device=dev)
    offsets[1:E + 1] = counts[:E].cumsum(0).to(torch.int64); offsets[E + 1] = P
    out_a = torch.empty(P, N, dtype=torch.half, device=dev)
    out_f = torch.empty(P, N, dtype=torch.half, device=dev)
    # Phase-1 kernel needs svh/bias pointers for its epilogue
    GLD(x, mk(ptrs), mk(sptrs), mk(bptrs), counts, offsets, 1, 4, 0, 16, torch.zeros(P, N, dtype=torch.float32, device=dev), out_a)
    GLF(x, mk(ptrs), mk(sptrs), mk(bptrs), counts, offsets, 1, 4, 0, 16, torch.zeros(P, N, dtype=torch.float32, device=dev), out_f)
    torch.cuda.synchronize()
    bit = torch.equal(out_a, out_f)
    d = (out_a.float() - out_f.float()).abs().max().item()
    ok_all &= bit
    print(f"P={P}: bitwise={bit} maxdiff={d:.3e}", flush=True)
print("GATE1", "PASS" if ok_all else "FAIL", flush=True)

print("=== Gate 2: microbench P=32/64 splits=16 ===", flush=True)
for P in (32, 64):
    x = torch.randn(P, K, dtype=torch.half, device=dev)
    counts = torch.zeros(E + 1, dtype=torch.int32, device=dev)
    per = P // 8; counts[:8] = per; counts[:P - per * 8] += 1
    offsets = torch.zeros(E + 2, dtype=torch.int64, device=dev)
    offsets[1:E + 1] = counts[:E].cumsum(0).to(torch.int64); offsets[E + 1] = P
    res = {}
    for tag, op in (("phase1", GLD), ("fused", GLF)):
        scratch = torch.zeros(P, N, dtype=torch.float32, device=dev)
        out = torch.empty(P, N, dtype=torch.half, device=dev)
        for _ in range(5):
            scratch.zero_(); op(x, mk(ptrs), mk(sptrs), mk(bptrs), counts, offsets, 1, 4, 0, 16, scratch, out)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(200):
            scratch.zero_(); op(x, mk(ptrs), mk(sptrs), mk(bptrs), counts, offsets, 1, 4, 0, 16, scratch, out)
        torch.cuda.synchronize(); t = (time.perf_counter() - t0) / 200 * 1e6
        res[tag] = t
    print(f"P={P}: phase1={res['phase1']:.1f}us fused={res['fused']:.1f}us speedup={res['phase1']/res['fused']:.2f}x", flush=True)
print("GATES DONE", flush=True)
