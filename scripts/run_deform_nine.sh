#!/bin/bash
# Independent baseline: Deformable-3DGS-style (MLP deformation motion) on ALL 9 scenes,
# 33-frame, our even/odd protocol, res 512, cap 250k (comparable capacity to ours ~226k).
# Trains via retimegs_4d.py --deform 2, then evaluates both/novel/retiming via eval_deform.py.
ROOT=${AAR_ROOT:?set AAR_ROOT to the repo root}; ENV=${AAR_ENV:-}            # optional conda prefix; unset means "use the active env"
export PYTHONPATH=$ROOT/refs/vggt-omega HF_HOME=$ROOT/hf_cache TORCH_HOME=$ROOT/.torch \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} TORCH_CUDA_ARCH_LIST=7.5 \
  PATH=${ENV:+$ENV/bin:}$PATH PYTHONUTF8=1 LANG=C.UTF-8 \
  CC=${CC:-gcc} CXX=${CXX:-g++}
cd $ROOT
for s in walk bear doll newundress opencase passdoll pickupdoll putcomputer stretch; do
  od="experiments/outputs/deform_nine/$s"
  if [ -f "$od/eval.json" ]; then echo "=== $s already evaluated, skip ==="; continue; fi
  echo "=== train Deform-GS $s (33f, cap250k) start $(date +%H:%M:%S) ==="
  CUDA_VISIBLE_DEVICES=0 ${PYTHON:-python} -u experiments/retimegs_4d.py \
    --root $ROOT/data/sc_mv_all --scene "$s" --nframes 33 --res 512 --deform 2 \
    --iters 20000 --cap 250000 --out "$od" 2>&1 | tail -3
  echo "=== eval Deform-GS $s (our protocol) ==="
  CUDA_VISIBLE_DEVICES=0 ${PYTHON:-python} -u experiments/eval_deform.py "$od" "$s" 33 2>&1 | tee "$od/eval.txt" | tail -2
  # stamp a machine-readable eval.json from eval.txt
  ${PYTHON:-python} -c "
import re,json,sys
t=open('$od/eval.txt').read()
m=re.search(r'both=([\d.]+)\s+novel=([\d.]+)\s+retiming=([\d.]+)\s+train=([\d.]+)',t)
if m: json.dump({'scene':'$s','both':float(m.group(1)),'novel':float(m.group(2)),'retiming':float(m.group(3)),'train':float(m.group(4))},open('$od/eval.json','w'))
"
done
echo "=== Deform-GS 9-scene done $(date +%H:%M:%S) ==="
