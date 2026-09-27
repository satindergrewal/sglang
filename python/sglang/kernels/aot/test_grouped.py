import json, torch, sys
sys.path.insert(0, '/home/satinder/Documents/Github/sglang-vendorport/python')
from safetensors import safe_open
from sglang.srt.layers.quantization.exl3 import dequant_matrix_orig

torch.ops.load_library('/home/satinder/.cache/torch_extensions/py312_cu130/exl3_ops_patch/exl3_ops_patch.so')

B = '/home/satinder/models/mimo40-trim6'
wm = json.load(open(f'{B}/model.safetensors.index.json'))['weight_map']
def g(n):
    return safe_open(f'{B}/{wm[n]}', framework='pt', device='cuda').get_tensor(n)

torch.manual_seed(0)
dev = 'cuda'

# --- Gate 1: dense kernel vs python dequant on a real expert matrix ---
name = 'model.layers.4.mlp.experts.0.gate_proj'
p = name + '.'
trellis = g(p+'trellis'); suh = g(p+'suh'); svh = g(p+'svh')
W = dequant_matrix_orig(trellis, suh, svh, 'mul1').float()
x = torch.randn(16, W.shape[0], device=dev, dtype=torch.float32).half()
ref = (x.float() @ W)

xh = torch.empty(x.shape, dtype=torch.float16, device=dev)
torch.ops.sgl_kernel.sgl_exl3_had_in(x, suh, xh)
out = torch.empty((x.shape[0], svh.shape[0]), dtype=torch.float16, device=dev)
torch.ops.sgl_kernel.sgl_exl3_linear(xh, trellis, svh, None, 2, out)
d = (out.float() - ref).abs()
rel = d.max().item() / ref.abs().max().item()
cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()
print(f'GATE1 dense: maxrel {rel:.5f} cos {cos:.6f}')
assert cos > 0.999, 'dense kernel gate FAILED'

# --- Gate 2: grouped vs per-expert loop ---
E = 4
counts = torch.tensor([4, 3, 6, 3], dtype=torch.int32, device=dev)
P = int(counts.sum().item())
offsets = torch.zeros(E, dtype=torch.int64, device=dev)
offsets[1:] = counts[:-1].cumsum(0).to(torch.int64)
print('counts', counts.tolist(), 'offsets', offsets.tolist())

# pair row -> expert map, sorted by expert (stable) like the runner
tok_map = torch.argsort(torch.repeat_interleave(torch.arange(E, device=dev), counts), stable=True)
x_pairs = x[tok_map].contiguous()
x_orig = x_pairs.clone()  # pre-had_in rows for the reference loop
experts = [0, 1, 2, 3]

packed_ptrs = torch.zeros(E, dtype=torch.int64, device=dev)
svh_ptrs = torch.zeros(E, dtype=torch.int64, device=dev)
bias_ptrs = torch.zeros(E, dtype=torch.int64, device=dev)
Ws = []
keepalive = []  # pointers handed to the kernel must stay live
for i, e in enumerate(experts):
    pp = f'model.layers.4.mlp.experts.{e}.gate_proj.'
    t = g(pp+'trellis'); s_ = g(pp+'suh'); v_ = g(pp+'svh')
    keepalive.extend([t, v_])
    packed_ptrs[i] = t.data_ptr(); svh_ptrs[i] = v_.data_ptr(); bias_ptrs[i] = 0
    Ws.append(dequant_matrix_orig(t, s_, v_, 'mul1').float())

# Per-expert had_in on the pair rows (each expert has its own suh), exactly
# what the runner does before the grouped GEMM.
for i, e in enumerate(experts):
    s0 = int(offsets[i]); c = int(counts[i])
    suh_e = g(f'model.layers.4.mlp.experts.{e}.gate_proj.suh')
    xh_p = torch.empty(x_pairs[s0:s0+c].shape, dtype=torch.float16, device=dev)
    torch.ops.sgl_kernel.sgl_exl3_had_in(x_pairs[s0:s0+c].contiguous(), suh_e, xh_p)
    x_pairs[s0:s0+c] = xh_p

col_blocks = svh.shape[0] // 128
chunks_cap = 2
splits = 4
n = svh.shape[0]
ws = torch.zeros((col_blocks, splits, E * chunks_cap, 16, 128), dtype=torch.float32, device=dev)
cnt_buf = torch.zeros((E * col_blocks, chunks_cap), dtype=torch.int32, device=dev)
out_g = torch.empty((P, n), dtype=torch.float16, device=dev)
torch.ops.sgl_exl3_grouped.grouped_linear(
    x_pairs, packed_ptrs, svh_ptrs, bias_ptrs, counts, offsets,
    ws, cnt_buf, 2, splits, chunks_cap, 4, 0, out_g)

# reference: per-expert loop with the dense op
ref_g = torch.empty_like(out_g)
for i, e in enumerate(experts):
    s0 = int(offsets[i]); c = int(counts[i])
    xe = x_orig[s0:s0+c]
    xh_e = torch.empty(xe.shape, dtype=torch.float16, device=dev)
    torch.ops.sgl_kernel.sgl_exl3_had_in(xe.contiguous(), g(f'model.layers.4.mlp.experts.{e}.gate_proj.suh'), xh_e)
    o = torch.empty((c, n), dtype=torch.float16, device=dev)
    torch.ops.sgl_kernel.sgl_exl3_linear(xh_e, g(f'model.layers.4.mlp.experts.{e}.gate_proj.trellis'), g(f'model.layers.4.mlp.experts.{e}.gate_proj.svh'), None, 2, o)
    ref_g[s0:s0+c] = o

for r in range(P):
    c_r = torch.nn.functional.cosine_similarity(out_g[r].float(), ref_g[r].float(), dim=0).item()
    print(f'  row {r} (expert {tok_map[r].item()}->…): cos {c_r:.5f} out_abs {out_g[r].float().abs().max().item():.4f} ref_abs {ref_g[r].float().abs().max().item():.4f}')
d2 = (out_g.float() - ref_g.float()).abs()
mx = d2.max().item()
denom = ref_g.float().abs().max().item()
print(f'GATE2 grouped-vs-loop: maxdiff {mx:.6f} (scale {denom:.3f}) rel {mx/denom:.6f}')
ok = (d2.max().item() / denom) < 0.02
print('GATE2', 'PASS' if ok else 'FAIL')
assert ok
print('ALL GATES PASS')
