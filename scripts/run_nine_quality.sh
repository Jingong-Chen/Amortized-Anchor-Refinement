#!/bin/bash
# 9-scene run at the MAIN QUALITY config (matches \mainboth=24.39: no mirror, HD, 96k) + perceptual metrics.
# Locks the headline result to 9-scene mean/std AND adds SSIM/LPIPS (reviewer critiques #1,#4,#5). ~42min/scene, ~6.3h total.
ROOT=${AAR_ROOT:?set AAR_ROOT to the repo root}; ENV=${AAR_ENV:-}            # optional conda prefix; unset means "use the active env"
export PYTHONPATH=$ROOT/refs/vggt-omega HF_HOME=$ROOT/hf_cache TORCH_HOME=$ROOT/.torch \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} TORCH_CUDA_ARCH_LIST=7.5 \
  PATH=${ENV:+$ENV/bin:}$PATH PYTHONUTF8=1 LANG=C.UTF-8 \
  CC=${CC:-gcc} CXX=${CXX:-g++}
cd $ROOT
for s in walk bear doll newundress opencase passdoll pickupdoll putcomputer stretch; do
  od="experiments/outputs/nine_quality/$s"
  if ls "$od"/result_*.json >/dev/null 2>&1; then echo "=== $s already done, skip ==="; continue; fi
  echo "=== scene $s start $(date +%H:%M:%S) ==="
  CUDA_VISIBLE_DEVICES=0 ${PYTHON:-python} -u experiments/hybrid_amort.py \
    --scene "$s" --n_frames 33 --cpu_data 1 --densify 1 --gt_supervise 1 --all_cam_gt 1 \
    --ssim_w 0.2 --lpips 0.1 --lr_decay 1 --antifloat 1 --tau_mul 1.5 \
    --min_n 200000 --iters 96000 --ff_offload 1 --seed 0 --eval_perceptual 1 \
    --out "$od" 2>&1 | tail -4
done
echo "=== 9-scene QUALITY done $(date +%H:%M:%S) ==="
