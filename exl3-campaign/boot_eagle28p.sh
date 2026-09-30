#!/bin/bash
# EAGLE radix-ON fix test (image 28p = 28n + draft-extend prefix-zero fix)
docker rm -f exl3-eagle28p 2>/dev/null
docker run -d --name exl3-eagle28p --gpus all --shm-size=32g \
  -v /mnt/nvme0/work-exl3/mimo375/out375:/model:ro \
  -v /mnt/nvme0/bigmodels/MiMo-V2.6-Flash-RL:/checkpoint:ro \
  -p 8015:8015 \
  satgeze/sglang-exl3:exl3-grouped-20260928p \
  python3 -m sglang.launch_server \
    --model-path /model \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path /checkpoint \
    --speculative-draft-attention-backend flashinfer \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --tp-size 2 --ep-size 2 \
    --mem-fraction-static 0.85 \
    --kv-cache-dtype fp8_e4m3 \
    --trust-remote-code \
    --reasoning-parser deepseek-r1 \
    --disable-custom-all-reduce \
    --disable-prefill-cuda-graph \
    --json-model-override-args '{"enable_multimodal": false}' \
    --port 8015 > /mnt/nvme0/work-exl3/eagle28p.log 2>&1
