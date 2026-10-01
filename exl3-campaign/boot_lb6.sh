#!/bin/bash
# Daily + tool-call parsing + served alias: layout-B serve, in-loader panel
# reorder, fixed dense exl3_patch_lb.so (stride-unit fix + layout_b), grouped
# g38 with direct_b. New: --tool-call-parser mimo (MiMoDetector matches the
# chat template's tool format) and --served-model-name mimo (v1/models id).
docker rm -f exl3-375 2>/dev/null
docker run -d --name exl3-375 --gpus all --shm-size=32g -e SGLANG_EXL3_MOE_PACKED=1 -e EXL3_MOE_GROUPED=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e EXL3_PANEL_LAYOUT=B --ipc=host \
  -e NCCL_P2P_DISABLE=1 -e NCCL_P2P_LEVEL=LOC -e NCCL_CUMEM_ENABLE=0 \
  -v /mnt/nvme0/work-exl3/mimo375/out375:/model:ro \
  -v /mnt/nvme0/work-exl3/wops2/build_primary/exl3_patch_lb.so:/work/exl3_patch/exl3_patch.so:ro \
  -v /mnt/nvme0/work-exl3/build_g38/exl3_ops_patch_g38/exl3_ops_patch_g38.so:/work/exl3_ops/exl3_ops_patch.so:ro \
  -v /mnt/nvme0/work-exl3/wops2/image_exl3_runner_lb.py:/sgl-workspace/sglang/python/sglang/srt/layers/quantization/exl3.py:ro \
  -p 8015:8015 \
  satgeze/sglang-exl3:exl3-grouped-20260928x \
  python3 -m sglang.launch_server \
    --model-path /model \
    --served-model-name mimo \
    --tool-call-parser mimo \
    --tp-size 2 --ep-size 2 \
    --mem-fraction-static 0.85 \
    --kv-cache-dtype fp8_e4m3 \
    --trust-remote-code \
    --reasoning-parser deepseek-r1 \
    --disable-custom-all-reduce \
    --disable-prefill-cuda-graph \
    --json-model-override-args '{"enable_multimodal": false}' \
    --host 0.0.0.0 --port 8015 > /mnt/nvme0/work-exl3/daily375.log 2>&1
