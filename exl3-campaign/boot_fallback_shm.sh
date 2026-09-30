#!/bin/bash
# FALLBACK DAILY: force SHM transport (no GPU-to-GPU P2P) — used only if the
# box's P2P path is still wedged after the cold cycle. docker-run env flags
# belong BEFORE the image name.
docker rm -f exl3-375 2>/dev/null
docker run -d --name exl3-375 --gpus all --shm-size=32g --ipc=host \
  -e NCCL_P2P_DISABLE=1 -e NCCL_P2P_LEVEL=LOC -e NCCL_CUMEM_ENABLE=0 \
  -v /mnt/nvme0/work-exl3/mimo375/out375:/model:ro \
  -p 8015:8015 \
  satgeze/sglang-exl3:exl3-grouped-20260928n \
  python3 -m sglang.launch_server \
    --model-path /model \
    --tp-size 2 --ep-size 2 \
    --mem-fraction-static 0.88 \
    --kv-cache-dtype fp8_e4m3 \
    --trust-remote-code \
    --reasoning-parser deepseek-r1 \
    --disable-custom-all-reduce \
    --disable-prefill-cuda-graph \
    --json-model-override-args '{"enable_multimodal": false}' \
    --port 8015 > /mnt/nvme0/work-exl3/daily375.log 2>&1
