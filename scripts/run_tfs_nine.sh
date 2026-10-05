#!/bin/bash
# TFS nine-scene pipeline: per scene  (1) fit the MAIN 33f field (nine_quality recipe + --save_model 1)
#                                     (2) morse_prune budget sweep (topology floor vs importance)
#                                     (3) flow_code package at the 4e4 / 2e4 topo keeps
# Idempotent: every stage skips if its output exists. ~42min fit + ~12min analysis per scene.
set -u
ROOT=${AAR_ROOT:?set AAR_ROOT to the repo root}; ENV=${AAR_ENV:-}            # optional conda prefix; unset means "use the active env"
export PYTHONPATH=$ROOT/refs/vggt-omega HF_HOME=$ROOT/hf_cache TORCH_HOME=$ROOT/.torch \
  PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:256 CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} TORCH_CUDA_ARCH_LIST=7.5 \
  PATH=${ENV:+$ENV/bin:}$PATH PYTHONUTF8=1 LANG=C.UTF-8 \
  CC=${CC:-gcc} CXX=${CXX:-g++} \
  TORCH_EXTENSIONS_DIR=$ROOT/.torch_ext
cd $ROOT
for s in walk bear doll newundress opencase passdoll pickupdoll putcomputer stretch; do
  od="experiments/outputs/nine_models/$s"
  ck=$(ls "$od"/model_*_96000.pt 2>/dev/null | head -1)
  if [ -z "$ck" ]; then
    echo "=== [$s] FIT start $(date +%H:%M:%S) ==="
    CUDA_VISIBLE_DEVICES=0 ${PYTHON:-python} -u experiments/hybrid_amort.py \
      --scene "$s" --n_frames 33 --cpu_data 1 --densify 1 --gt_supervise 1 --all_cam_gt 1 \
      --ssim_w 0.2 --lpips 0.1 --lr_decay 1 --antifloat 1 --tau_mul 1.5 \
      --min_n 200000 --max_n 700000 --iters 96000 --ff_offload 1 --seed 0 --save_model 1 \
      --out "$od" 2>&1 | tail -3
    ck=$(ls "$od"/model_*_96000.pt 2>/dev/null | head -1)
    [ -z "$ck" ] && { echo "=== [$s] FIT FAILED, skip scene ==="; continue; }
  else echo "=== [$s] fit exists, skip ==="; fi
  if [ ! -f "experiments/outputs/morse_$s/morse_prune_$s.json" ]; then
    echo "=== [$s] MORSE start $(date +%H:%M:%S) ==="
    CUDA_VISIBLE_DEVICES=0 ${PYTHON:-python} -u experiments/morse_prune.py --scene "$s" --ckpt "$ck" \
      --targets 80000,40000,20000,10000,5000 2>&1 | grep -E "prune |topo-ref|full|saved" | tail -9
  else echo "=== [$s] morse exists, skip ==="; fi
  for tgt in 40000 20000; do
    if [ ! -f "experiments/outputs/flowcode_${s}_t${tgt}/flowcode_$s.json" ]; then
      echo "=== [$s] FLOW t$tgt $(date +%H:%M:%S) ==="
      CUDA_VISIBLE_DEVICES=0 ${PYTHON:-python} -u experiments/flow_code.py --scene "$s" --ckpt "$ck" \
        --keep "experiments/outputs/morse_$s/keep_topo_${tgt}.pt" --lattices 32 \
        --out "experiments/outputs/flowcode_${s}_t${tgt}" 2>&1 | grep -E "static|q10|saved" | tail -3
    fi
  done
  sleep 5
done
echo "=== TFS NINE done $(date +%H:%M:%S) ==="