#!/bin/bash
# Native-weights fp8-KV serve on 8015: the working long-context daily.
# NVFP4-KV verdict (2026-10-02): boots and generates plain text but chat is
# degenerate at any temperature (4-bit e2m1 KV crushes special-token framing).
# 1M pool unlock: --swa-full-tokens-ratio 0.02 (model SWA window = 128 tokens;
# default ratio 0.8 wasted ~22 KB/token on the SWA pool). EXL3 restorable via
# boot_lb6.sh; NVFP4-KV config preserved in boot_nvfp4_8015.sh.
docker rm -f exl3-375 exl-native 2>/dev/null
docker run -d --name exl-native --gpus all --shm-size=32g --ipc=host \
  -e NCCL_P2P_DISABLE=1 -e NCCL_P2P_LEVEL=LOC -e NCCL_CUMEM_ENABLE=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v /mnt/nvme0/bigmodels/MiMo-V2.6-Flash-RL:/model:ro \
  -p 8015:8015 \
  satgeze/sglang-exl3:exl3-grouped-20260928v \
  python3 -m sglang.launch_server \
    --model-path /model \
    --served-model-name mimo \
    --tool-call-parser mimo \
    --tp-size 2 --ep-size 2 \
    --mem-fraction-static 0.95 \
    --max-total-tokens 1048576 \
    --swa-full-tokens-ratio 0.02 \
    --kv-cache-dtype fp8_e4m3 --attention-backend flashinfer \
    --moe-runner-backend flashinfer_mxfp4 \
    --trust-remote-code \
    --reasoning-parser deepseek-r1 \
    --disable-custom-all-reduce \
    --disable-prefill-cuda-graph \
    --json-model-override-args '{"enable_multimodal": false}' \
    --cuda-graph-max-bs-decode 48 --host 0.0.0.0 --port 8015 > /mnt/nvme0/work-exl3/native8015.log 2>&1
