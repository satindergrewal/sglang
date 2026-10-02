#!/bin/bash
# Native-weights NVFP4-KV serve on 8015 (user daily swap; EXL3 restorable via
# boot_lb6.sh). Reference config = boot_nvfp4_native_nd.sh (matrix cell) with:
# port 8015, --served-model-name mimo, --tool-call-parser mimo, full 1M window.
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
    --mem-fraction-static 0.89 \
    --kv-cache-dtype nvfp4 --attention-backend flashinfer \
    --moe-runner-backend flashinfer_mxfp4 \
    --trust-remote-code \
    --reasoning-parser deepseek-r1 \
    --disable-custom-all-reduce \
    --disable-prefill-cuda-graph \
    --json-model-override-args '{"enable_multimodal": false}' \
    --cuda-graph-max-bs-decode 48 --host 0.0.0.0 --port 8015 > /mnt/nvme0/work-exl3/native8015.log 2>&1
