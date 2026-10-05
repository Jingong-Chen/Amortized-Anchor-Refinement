#!/bin/bash
# Deformable-3DGS and 4DGaussians on the DNA-Rendering scenes, same 33-frame every-other-frame
# protocol and same evaluator as the nine Stage-Capture scenes.
ROOT=${AAR_ROOT:?set AAR_ROOT to the repo root}; ENV=${AAR_ENV:-}            # optional conda prefix; unset means "use the active env"
DG=$ROOT/refs/Deformable-3D-Gaussians; FD=$ROOT/refs/4DGaussians
export HF_HOME=$ROOT/hf_cache TORCH_HOME=$ROOT/.torch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} TORCH_CUDA_ARCH_LIST=7.5 PATH=${ENV:+$ENV/bin:}$PATH \
  CC=${CC:-gcc} CXX=${CXX:-g++} \
  TORCH_EXTENSIONS_DIR=$ROOT/.torch_ext SC_NFRAMES=33 SC_RES=512 PYTHONUTF8=1 LANG=C.UTF-8
SCENES="${1:-0008_01 0019_06 0047_12}"
DR=data/dna_rendering

for s in $SCENES; do
  od=$ROOT/experiments/outputs/dnadgs_$s
  if [ -f "$od/eval33.json" ]; then echo "=== dgs $s done, skip ==="; else
    echo "=== DGS train $s $(date +%H:%M:%S) ==="; rm -rf "$od"; cd $DG
    CUDA_VISIBLE_DEVICES=0 ${PYTHON:-python} -u train.py -s $ROOT/$DR/$s -m "$od" --eval -r 2 --iterations 40000 2>&1 | tail -2
    CUDA_VISIBLE_DEVICES=0 ${PYTHON:-python} -u render.py -m "$od" --skip_train 2>&1 | tail -2
    cd $ROOT
    CUDA_VISIBLE_DEVICES=0 ${PYTHON:-python} -u experiments/eval_dgs33.py "$od/test/ours_40000" "$s" 33 "$DR" 2>&1 | tail -1
  fi
done

for s in $SCENES; do
  od=$ROOT/experiments/outputs/dna4dgs_$s
  if [ -f "$od/eval33.json" ]; then echo "=== 4dgs $s done, skip ==="; else
    echo "=== 4DGS train $s $(date +%H:%M:%S) ==="; rm -rf "$od"; cd $FD
    CUDA_VISIBLE_DEVICES=0 ${PYTHON:-python} -u train.py -s $ROOT/$DR/$s -m "$od" \
      --configs arguments/stagecapture/default.py --expname "dna_$s" 2>&1 | tail -3
    CUDA_VISIBLE_DEVICES=0 ${PYTHON:-python} -u render.py -m "$od" --skip_train --skip_video \
      --configs arguments/stagecapture/default.py 2>&1 | tail -2
    cd $ROOT
    RD=$(ls -d "$od"/test/ours_* 2>/dev/null | tail -1)
    [ -n "$RD" ] && CUDA_VISIBLE_DEVICES=0 ${PYTHON:-python} -u experiments/eval_dgs33.py "$RD" "$s" 33 "$DR" 2>&1 | tail -1
  fi
done
echo "=== DNA baselines done $(date +%H:%M:%S) ==="
