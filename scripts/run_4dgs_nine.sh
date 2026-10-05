#!/bin/bash
# Official 4DGaussians (Wu, CVPR24) on ALL 9 scenes, 33-frame, official protocol, 512.
ROOT=${AAR_ROOT:?set AAR_ROOT to the repo root}; ENV=${AAR_ENV:-}            # optional conda prefix; unset means "use the active env"; FD=$ROOT/refs/4DGaussians
export HF_HOME=$ROOT/hf_cache TORCH_HOME=$ROOT/.torch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} TORCH_CUDA_ARCH_LIST=7.5 PATH=${ENV:+$ENV/bin:}$PATH \
  CC=${CC:-gcc} CXX=${CXX:-g++} \
  TORCH_EXTENSIONS_DIR=$ROOT/.torch_ext SC_NFRAMES=33 SC_RES=512 PYTHONUTF8=1 LANG=C.UTF-8
ITER=14000
for s in walk bear doll newundress opencase passdoll pickupdoll putcomputer stretch; do
  od=$ROOT/experiments/outputs/fourdgs_$s
  if [ -f "$od/eval33.json" ]; then echo "=== 4dgs $s already evaluated, skip ==="; continue; fi
  echo "=== 4DGS train $s start $(date +%H:%M:%S) ==="
  rm -rf "$od"
  cd $FD
  CUDA_VISIBLE_DEVICES=0 ${PYTHON:-python} -u train.py -s $ROOT/data/sc_mv_all/$s -m "$od" \
    --configs arguments/stagecapture/default.py --expname "sc_$s" 2>&1 | tail -3
  echo "=== 4DGS render $s ==="
  CUDA_VISIBLE_DEVICES=0 ${PYTHON:-python} -u render.py -m "$od" --skip_train --skip_video \
    --configs arguments/stagecapture/default.py 2>&1 | tail -2
  echo "=== 4DGS eval $s ==="
  cd $ROOT
  RD=$(ls -d "$od"/test/ours_* 2>/dev/null | tail -1)
  [ -n "$RD" ] && CUDA_VISIBLE_DEVICES=0 ${PYTHON:-python} -u experiments/eval_dgs33.py "$RD" "$s" 33 2>&1 | tail -1
done
echo "=== 4DGS nine done $(date +%H:%M:%S) ==="
