import os, sys, json, torch, time
sys.path.insert(0, '/home/satinder/Documents/Github/sglang-vendorport/python')
os.environ.setdefault('EXL3_MOE_GROUPED', '1')
from safetensors import safe_open
import torch.nn.functional as F

torch.ops.load_library('/home/satinder/.cache/torch_extensions/py312_cu130/exl3_ops_patch_serve/exl3_ops_patch_serve.so')

B = '/home/satinder/models/mimo40-trim6'
wm = json.load(open(f'{B}/model.safetensors.index.json'))['weight_map']
def g(n):
    return safe_open(f'{B}/{wm[n]}', framework='pt', device='cuda').get_tensor(n)

dev = 'cuda'
L = '4'
E = 256          # real expert count per layer
HID = 4096
INT = 2048
TOPK = 8

# pack experts 0..255 of L4 (gate/up/down) — trellis + suh/svh
def build(proj):
    lst = []
    for e in range(E):
        pp = f'model.layers.{L}.mlp.experts.{e}.{proj}_proj'
        t = g(pp + '.trellis'); suh = g(pp + '.suh'); svh = g(pp + '.svh')
        lst.append((t, suh, svh))
    return lst

gate = build('gate'); up = build('up'); down = build('down')
keep = [t for l in (gate, up, down) for (t, s, v) in l]

def ptrs(lst, attr_idx=0):
    p = torch.zeros(E + 1, dtype=torch.int64, device=dev)
    for i, entry in enumerate(lst):
        p[i] = entry[attr_idx].data_ptr()
    return p

gate_ptrs, up_ptrs, down_ptrs = ptrs(gate), ptrs(up), ptrs(down)
gate_svh, up_svh, down_svh = ptrs(gate, 2), ptrs(up, 2), ptrs(down, 2)
gate_suh, up_suh, down_suh = ptrs(gate, 1), ptrs(up, 1), ptrs(down, 1)
zero_b = torch.zeros(E + 1, dtype=torch.int64, device=dev)
gate_bits = gate[0][0].shape[2] // 16
half_k = bool(gate[0][0].shape[2] % 16)
CB = 2

CH, SPLITS = 4, 4
def ws_make(n_out):
    cb_n = n_out // 128
    ws = torch.zeros((cb_n, SPLITS, (E + 1) * CH, 16, 128), dtype=torch.float32, device=dev)
    cnt = torch.zeros(((E + 1) * cb_n, CH), dtype=torch.int32, device=dev)
    return ws, cnt

ws_g, cnt_g = ws_make(INT)
ws_d, cnt_d = ws_make(HID)

def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us

print(f"{'rows':>5} {'route':>6} | {'route+sort':>10} {'had_in':>9} {'gemm_gu':>9} {'silu':>8} {'had_mid':>9} {'gemm_d':>9} {'scatter':>9} {'CHAIN':>9} | {'eff GB/s':>9} {'TFLOPs':>8}")
for rows_tokens in (4, 16, 64, 256):
    x = torch.randn(rows_tokens, HID, dtype=torch.float16, device=dev)
    ids = torch.randint(0, E, (rows_tokens, TOPK), device=dev, dtype=torch.int64)
    wts = torch.rand(rows_tokens, TOPK, device=dev, dtype=torch.float32) * 0.1

    # --- routing + gather (as the runner does it) ---
    def do_route():
        flat = ids.reshape(-1).to(torch.long)
        key = torch.where(flat >= 0, flat, torch.full_like(flat, E))
        order = torch.argsort(key, stable=True)
        sids = key[order]
        counts = torch.histc(sids.float(), bins=E, min=0, max=E - 1).to(torch.int32)
        counts = torch.cat([counts, torch.zeros(1, dtype=torch.int32, device=dev)])
        offsets = torch.zeros(E + 2, dtype=torch.int64, device=dev)
        offsets[1:] = counts.cumsum(0).to(torch.int64)
        xp = x[order // TOPK].contiguous()
        return counts, offsets, xp

    t_route = bench(lambda: do_route())
    counts, offsets, x_pairs = do_route()
    nd = int((counts[:E] > 0).sum().item())
    npairs = int(counts[:E].sum().item())

    xh_g = torch.empty_like(x_pairs); xh_u = torch.empty_like(x_pairs)
    def do_hadin():
        torch.ops.sgl_exl3_grouped.grouped_had_in(x_pairs, gate_suh, counts, offsets, 64, xh_g)
        torch.ops.sgl_exl3_grouped.grouped_had_in(x_pairs, up_suh, counts, offsets, 64, xh_u)
    t_had = bench(do_hadin)
    do_hadin()

    def gemm_pair(xh):
        o_g = torch.empty((xh.shape[0], INT), dtype=torch.float16, device=dev)
        torch.ops.sgl_exl3_grouped.grouped_linear(xh, gate_ptrs, gate_svh, zero_b, counts, offsets, ws_g, cnt_g, CB, SPLITS, CH, gate_bits, half_k, o_g)
        o_u = torch.empty((xh.shape[0], INT), dtype=torch.float16, device=dev)
        torch.ops.sgl_exl3_grouped.grouped_linear(xh, up_ptrs, up_svh, zero_b, counts, offsets, ws_g, cnt_g, CB, SPLITS, CH, gate_bits, half_k, o_u)
        return o_g, o_u
    o_g, o_u = gemm_pair(xh_g)
    t_gemm_gu = bench(lambda: gemm_pair(xh_g))

    def do_silu():
        return (F.silu(o_g.float()) * o_u.float()).to(torch.float16)
    mid = do_silu()
    t_silu = bench(do_silu)

    mid_t = torch.empty_like(mid)
    def do_hadmid():
        torch.ops.sgl_exl3_grouped.grouped_had_in(mid, down_suh, counts, offsets, 64, mid_t)
    do_hadmid()
    t_hadmid = bench(do_hadmid)

    def gemm_down():
        o_d = torch.zeros((mid.shape[0], HID), dtype=torch.float16, device=dev)
        torch.ops.sgl_exl3_grouped.grouped_linear(mid_t, down_ptrs, down_svh, zero_b, counts, offsets, ws_d, cnt_d, CB, SPLITS, CH, gate_bits, half_k, o_d)
        return o_d
    o_d = gemm_down()
    t_gemm_d = bench(gemm_down)

    tok = (torch.arange(rows_tokens, device=dev).repeat_interleave(TOPK))
    order = torch.argsort(ids.reshape(-1).to(torch.long), stable=True)
    tok_sorted = tok[order]
    pw = torch.rand(npairs if npairs else 1, device=dev, dtype=torch.float32) * 0.1
    out = torch.zeros((rows_tokens, HID), dtype=torch.float32, device=dev)
    def do_scatter():
        out.index_add_(0, tok_sorted[:o_d.shape[0]], o_d.float())
    t_scatter = bench(do_scatter)

    chain = t_route + t_had + t_gemm_gu + t_silu + t_hadmid + t_gemm_d + t_scatter
    # bandwidth model: distinct experts' weights streamed once per step
    wbytes = nd * 3 * HID * INT * (gate_bits / 2.0)  # bits/2 = bytes per param at 2b granularity? bits/8*2? use bits/8
    wbytes = nd * 3 * HID * INT * gate_bits / 8
    eff = wbytes / (chain * 1e-6) / 1e9
    flops = npairs * 2 * (2 * HID * INT + INT * HID)
    tf = flops / (chain * 1e-6) / 1e12
    print(f"{rows_tokens:>5} {nd:>4}d{npairs:>5}p | {t_route:>10.1f} {t_had:>9.1f} {t_gemm_gu:>9.1f} {t_silu:>8.1f} {t_hadmid:>9.1f} {t_gemm_d:>9.1f} {t_scatter:>9.1f} {chain:>9.1f} | {eff:>9.0f} {tf:>8.2f}")
