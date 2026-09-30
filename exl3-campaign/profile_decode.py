#!/usr/bin/env python3
"""Decode-step profile decomposition: native vs EXL3 (111-protocol decode load).

Runs against a LIVE serve (any of the matrix configs):
  1. warms up with a short bench burst,
  2. captures a torch-profiler window around a steady decode load via
     /start_profile + /stop_profile,
  3. aggregates CUDA kernel time into components
     (attention / dense-gemm / moe / nccl / sampling / other),
  4. prints per-step totals and the component split.

Usage:
  python3 profile_decode.py --base http://127.0.0.1:8015 --label exl3-375-nodraft
  python3 profile_decode.py --base http://127.0.0.1:8016 --label native-nodraft
Compare the two labels' tables. Run each serve in the same 111 shape:
16 prompts in 256 out 64, max-concurrency 4.
"""
import argparse, json, time, urllib.request, collections, os, glob

def http_post(url, payload=None):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload or {}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read() or b"{}")

def http_get(url):
    with urllib.request.urlopen(url, timeout=600) as r:
        return json.loads(r.read())

PROMPTS = [
    "Write a short paragraph about the sea.",
    "Explain how a bicycle works.",
    "Describe a mountain sunrise.",
    "What is the capital of France?",
] * 4  # 16 requests, concurrency 4 via bench loop

def chat(payload_text):
    return {
        "model": "default",
        "messages": [{"role": "user", "content": payload_text}],
        "max_tokens": 64,
        "temperature": 0.0,
    }

def run_decode_load(base, n=16, conc=4):
    import threading
    sem = threading.Semaphore(conc)
    errs = []

    def one(p):
        with sem:
            try:
                http_post(base + "/v1/chat/completions", chat(p))
            except Exception as e:  # noqa: BLE001
                errs.append(str(e))

    threads = [threading.Thread(target=one, args=(p,)) for p in PROMPTS[:n]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errs

def classify_kernel(name):
    n = name.lower()
    if "nccl" in n or "allreduce" in n or "all_reduce" in n or "cross_device" in n or "ncclDevKernel" in name:
        return "nccl"
    if any(k in n for k in ("flashinfer", "batch_prefill", "batch_decode", "paged_kv", "mha_fwd", "fmha", "attention", "flash::", "fmha_")):
        return "attention"
    if any(k in n for k in ("exl3", "trellis", "gemm", "matmul", "cutlass", "nvjet", "s16816", "wgmma", "sm100", "cublas")):
        # split moe out of dense by kernel-name hints
        if any(k in n for k in ("grouped", "moe", "expert", "route", "had_in", "had_out", "silu", "topk", "sort")):
            return "moe"
        return "dense_gemm"
    if any(k in n for k in ("sampling", "softmax", "argmax", "topk", "multinomial", "penal")):
        return "sampling"
    if any(k in n for k in ("quant", "fp4", "fp8", "dequant", "convert", "cast", "cvt", "elementwise", "vectorized", "reduce_kernel")):
        return "quant_elementwise"
    return "other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", default="/tmp/decode_profile")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tr_dir = os.path.join(args.out_dir, args.label)
    os.makedirs(tr_dir, exist_ok=True)

    # warmup burst
    print("warmup...", flush=True)
    run_decode_load(args.base, n=8, conc=4)

    info = http_get(args.base + "/get_server_info")
    print("server:", {k: info.get(k) for k in ("max_total_num_tokens",) if k in info})

    print("profiling window...", flush=True)
    http_post(args.base + "/start_profile", {
        "output_dir": tr_dir,
        "num_steps": 200,
        "activities": ["CPU", "GPU"],
    })
    t0 = time.time()
    errs = run_decode_load(args.base, n=16, conc=4)
    dt = time.time() - t0
    print(f"decode load wall: {dt:.2f}s errors={len(errs)}", flush=True)
    time.sleep(3)
    try:
        http_post(args.base + "/stop_profile", {})
    except Exception:
        pass
    time.sleep(5)

    traces = sorted(glob.glob(os.path.join(tr_dir, "*.trace.json.gz")) + glob.glob(os.path.join(tr_dir, "*.trace.json")))
    assert traces, f"no trace produced in {tr_dir}"
    import gzip
    opener = gzip.open if traces[0].endswith(".gz") else open
    with opener(traces[0], "rt") as f:
        ev = json.load(f)["traceEvents"]

    kern = [e for e in ev if e.get("ph") == "X" and e.get("cat", "").lower() in ("kernel", "gpu_kernel")]
    agg = collections.defaultdict(float)
    cnt = collections.Counter()
    total = 0.0
    for e in kern:
        d = e.get("dur", 0) / 1000.0  # us -> ms
        c = classify_kernel(e.get("name", ""))
        agg[c] += d
        cnt[c] += 1
        total += d
    print(f"\n== {args.label} ==  GPU kernel total {total:.1f} ms over {len(kern)} launches")
    for c in sorted(agg, key=agg.get, reverse=True):
        print(f"  {c:16s} {agg[c]:9.1f} ms  ({100*agg[c]/max(total,1e-9):5.1f}%)  n={cnt[c]}")

if __name__ == "__main__":
    main()
