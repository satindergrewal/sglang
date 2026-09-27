import os, sys, json, torch
sys.path.insert(0, '/home/satinder/Documents/Github/sglang-vendorport/python')
os.environ.setdefault('SGLANG_EXL3_MOE_PACKED', '1')
os.environ.setdefault('EXL3_MOE_GROUPED', '1')

# Load the trim's L4 experts through the real runner pieces: build the packed
# state via ExL3MoEMethod internals is heavy; instead drive the grouped path
# directly with tensors from the trim + the kernel, then compare vs the loop
# path of the same runner class at the batch shapes decode actually uses.
from safetensors import safe_open
from sglang.srt.layers.quantization.exl3 import (dequant_matrix_orig,
                                                 MoEGroupedOverflow)
import torch.nn.functional as F

torch.ops.load_library('/home/satinder/.cache/torch_extensions/py312_cu130/exl3_ops_patch/exl3_ops_patch.so')

B = '/home/satinder/models/mimo40-trim6'
wm = json.load(open(f'{B}/model.safetensors.index.json'))['weight_map']
def g(n):
    return safe_open(f'{B}/{wm[n]}', framework='pt', device='cuda').get_tensor(n)

torch.manual_seed(1)
dev = 'cuda'
L = '4'
E = 8  # small expert set for the harness
slots = 8  # topk

# build fake packed dict for experts 0..7 of L4 gate/up/down
packed = {}
keep = []
for proj, wname in [('gate', 'w13w1'), ('up', 'w13w3'), ('down', 'w2w2')]:
    lst = []
    for e in range(E):
        pp = f'model.layers.{L}.mlp.experts.{e}.'
        if proj == 'gate':
            name = pp + 'gate_proj'
        elif proj == 'up':
            name = pp + 'up_proj'
        else:
            name = pp + 'down_proj'
        t = g(name + '.trellis'); s_ = g(name + '.suh'); v_ = g(name + '.svh')
        mul1 = g(name + '.mul1')
        keep.extend([t, s_, v_, mul1])
        lst.append(('trellis', t, s_, v_, None, 2))
    packed[proj] = lst

# rebuild pointer metadata exactly as the runner does
dev = 'cuda'
def _ptr_arrays(proj):
    n_exp = len(packed[proj])
    pk = torch.zeros(n_exp + 1, dtype=torch.int64, device=dev)
    sv = torch.zeros(n_exp + 1, dtype=torch.int64, device=dev)
    bi = torch.zeros(n_exp + 1, dtype=torch.int64, device=dev)
    for i, mat in enumerate(packed[proj]):
        t, s_, v_, b_, c_ = mat[1], mat[2], mat[3], mat[4], mat[5]
        pk[i] = t.data_ptr(); sv[i] = v_.data_ptr(); bi[i] = 0
    packed[f'{proj}_ptrs'] = pk
    packed[f'{proj}_svh_ptrs'] = sv
    packed[f'{proj}_bias_ptrs'] = bi
    packed[f'{proj}_bits'] = mat[1].shape[2] // 16
    packed[f'{proj}_half_k'] = 1 if mat[1].shape[2] % 16 else 0
    packed[f'{proj}_cb'] = 2
    packed[f'{proj}_suh_keep'] = [m[2] for m in packed[proj]]
    packed[f'{proj}_n_out'] = v_.shape[0]
for proj in ('gate', 'up', 'down'):
    _ptr_arrays(proj)
packed['gate_n'] = packed['gate_n_out']
packed['down_n'] = packed['down_n_out']

H = packed['gate_suh_keep'][0].shape[0]
I = packed['down_suh_keep'][0].shape[0]
print(f'H={H} I={I}')

class RC:
    activation = 'silu'
    routed_scaling_factor = None

# decode-shaped batch: bs tokens x topk slots
for bs in (1, 2, 4, 8):
    T = bs
    x = torch.randn(T, H, device=dev).half()
    if os.environ.get('ONE_EXPERT') == '1':
        ids = torch.zeros(T, slots, device=dev, dtype=torch.int32)
    elif os.environ.get('REMOTE_PAIRS') == '1':
        ids = torch.randint(0, E, (T, slots), device=dev, dtype=torch.int32)
        ids[:, ::2] = -1  # half the pairs remote (sentinel)
    else:
        ids = torch.randint(0, E, (T, slots), device=dev, dtype=torch.int32)
    wts = torch.rand(T, slots, device=dev, dtype=torch.float32)
    wts = wts / wts.sum(-1, keepdim=True)

    # reference: per-expert loop (fp32 dequant math)
    flat = ids.reshape(-1).long()
    w_ref = wts.reshape(-1)
    ref = torch.zeros(T, packed['down_n'], device=dev, dtype=torch.float32)
    cache = {}
    real = flat >= 0
    for i in range(E):
        idx = ((flat == i) & real).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        rows = idx // slots
        xe = x[rows]
        ge = dequant_matrix_orig(packed['gate'][i][1], packed['gate'][i][2], packed['gate'][i][3], 'mul1').float()
        ue = dequant_matrix_orig(packed['up'][i][1], packed['up'][i][2], packed['up'][i][3], 'mul1').float()
        de = dequant_matrix_orig(packed['down'][i][1], packed['down'][i][2], packed['down'][i][3], 'mul1').float()
        mid = (F.silu(xe.float() @ ge) * (xe.float() @ ue))
        od = (mid @ de)
        ref.index_add_(0, rows, od * w_ref[idx][:, None])
        del ge, ue, de
        torch.cuda.empty_cache()

    # grouped path
    from sglang.srt.layers.quantization.exl3 import ExL3MoEMethod
    method = ExL3MoEMethod.__new__(ExL3MoEMethod)
    method._interm_comp = 1.0
    x_pairs_src = x
    out = method._run_packed_moe_grouped(
        x, ids, wts, packed, torch.float16, RC(), E)
    d = (out.float() - ref).abs().max().item()
    denom = ref.abs().max().item()
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()
    print(f'bs={bs}: maxdiff {d:.5f} scale {denom:.3f} rel {d/denom:.5f} cos {cos:.6f}')
    nz = (out.float().abs().sum(-1) == 0)
    if nz.any():
        print('  zero rows:', nz.nonzero().flatten().tolist())
    for r in range(min(T, 4)):
        print(f'  row {r}: cos', round(torch.nn.functional.cosine_similarity(out[r].float(), ref[r], dim=0).item(), 5))
    assert cos > 0.999, f'bs={bs} FAILED'
print('RUNNER-PATH GATES PASS')
