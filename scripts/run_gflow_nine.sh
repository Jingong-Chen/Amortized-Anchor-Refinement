#!/bin/bash
# GaussianFlow-style baseline (reproduced; official repo releases no full trainer):
# flow-supervised Gaussian dynamics (RAFT optical flow, alpha-composited per-Gaussian screen flow)
# on our pipeline, standard budget, 33f@512, official every-other-frame protocol.
ROOT=${AAR_ROOT:?set AAR_ROOT to the repo root}; ENV=${AAR_ENV:-}            # optional conda prefix; unset means "use the active env"
export PYTHONPATH=$ROOT/refs/vggt-omega HF_HOME=$ROOT/hf_cache TORCH_HOME=$ROOT/.torch \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} TORCH_CUDA_ARCH_LIST=7.5 \
  PATH=${ENV:+$ENV/bin:}$PATH PYTHONUTF8=1 LANG=C.UTF-8 \
  CC=${CC:-gcc} CXX=${CXX:-g++}
cd $ROOT
for s in walk bear doll newundress opencase passdoll pickupdoll putcomputer stretch; do
  od=experiments/outputs/gflow_$s
  if [ -f "$od/res_$s.json" ]; then echo "=== gflow $s already done, skip ==="; continue; fi
  echo "=== gflow train $s start $(date +%H:%M:%S) ==="
  CUDA_VISIBLE_DEVICES=0 ${PYTHON:-python} -u experiments/retimegs_4d.py \
    --root $ROOT/data/sc_mv_all --scene "$s" --nframes 33 --res 512 \
    --iters 30000 --cap 400000 --flow 20 \
    --out "$od" 2>&1 | tail -3
done
echo "=== gflow nine done $(date +%H:%M:%S) ==="
