#!/bin/bash
# THE DAILY: 3.75bpw EXL3 MiMo-V2.6-Flash-RL no-draft serve on 8015
# (multimodal off: the 3.75 artifact ships no audio tower)
docker rm -f exl3-375 2>/dev/null
docker run -d --name exl3-375 --gpus all --shm-size=32g \
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
