#!/bin/bash
# Official Deformable-3DGS on ALL 9 scenes, FULL 33-frame action, official every-other-frame protocol,
# -r 2 (512x375, matching our eval res). Train 40k -> render test -> fg PSNR/SSIM/LPIPS eval.
ROOT=${AAR_ROOT:?set AAR_ROOT to the repo root}; ENV=${AAR_ENV:-}            # optional conda prefix; unset means "use the active env"; DG=$ROOT/refs/Deformable-3D-Gaussians
export HF_HOME=$ROOT/hf_cache TORCH_HOME=$ROOT/.torch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} TORCH_CUDA_ARCH_LIST=7.5 PATH=${ENV:+$ENV/bin:}$PATH \
  CC=${CC:-gcc} CXX=${CXX:-g++} \
  TORCH_EXTENSIONS_DIR=$ROOT/.torch_ext SC_NFRAMES=33 PYTHONUTF8=1 LANG=C.UTF-8
for s in walk bear doll newundress opencase passdoll pickupdoll putcomputer stretch; do
  od=$ROOT/experiments/outputs/dgs33_$s
  if [ -f "$od/eval33.json" ]; then echo "=== dgs33 $s already evaluated, skip ==="; continue; fi
  echo "=== DGS33 train $s start $(date +%H:%M:%S) ==="
  rm -rf "$od"
  cd $DG
  CUDA_VISIBLE_DEVICES=0 ${PYTHON:-python} -u train.py -s $ROOT/data/sc_mv_all/$s -m "$od" --eval -r 2 --iterations 40000 2>&1 | tail -2
  echo "=== DGS33 render $s ==="
  CUDA_VISIBLE_DEVICES=0 ${PYTHON:-python} -u render.py -m "$od" --skip_train 2>&1 | tail -2
  echo "=== DGS33 eval $s ==="
  cd $ROOT
  CUDA_VISIBLE_DEVICES=0 ${PYTHON:-python} -u experiments/eval_dgs33.py "$od/test/ours_40000" "$s" 33 2>&1 | tail -1
done
echo "=== DGS33 nine done $(date +%H:%M:%S) ==="
