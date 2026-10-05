#!/bin/bash
# RetimeGS (reproduced, heavyweight 60k recipe) on remaining scenes, 33f@512, official protocol.
ROOT=${AAR_ROOT:?set AAR_ROOT to the repo root}; ENV=${AAR_ENV:-}            # optional conda prefix; unset means "use the active env"
export PYTHONPATH=$ROOT/refs/vggt-omega HF_HOME=$ROOT/hf_cache TORCH_HOME=$ROOT/.torch \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} TORCH_CUDA_ARCH_LIST=7.5 \
  PATH=${ENV:+$ENV/bin:}$PATH PYTHONUTF8=1 LANG=C.UTF-8 \
  CC=${CC:-gcc} CXX=${CXX:-g++}
cd $ROOT
for s in walk bear doll newundress opencase passdoll pickupdoll putcomputer stretch; do
  od=experiments/outputs/teacher33v2_$s
  if [ -f "$od/res_$s.json" ]; then echo "=== teacher $s already done, skip ==="; continue; fi
  echo "=== teacher train $s start $(date +%H:%M:%S) ==="
  CUDA_VISIBLE_DEVICES=0 ${PYTHON:-python} -u experiments/retimegs_4d.py \
    --root $ROOT/data/sc_mv_all --scene "$s" --nframes 33 --res 512 \
    --iters 60000 --cap 1280000 --flow 20 --cr 1 --vreg 1.0 --lpips 0.1 --lr_decay 1 --triple 0.3 \
    --out "$od" 2>&1 | tail -3
done
echo "=== teacher nine done $(date +%H:%M:%S) ==="
