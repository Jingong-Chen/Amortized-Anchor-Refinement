#!/bin/bash
# Exp-87/88/89: close the deployment gap of hybrid amortization (walk + bear).
# Variables isolated vs Exp-85d (teacher-back + teacher-sup = 24.25):
#   back-side init : mirror (deployable symmetry prior) vs teacher-proxy vs none
#   supervision    : real sparse capture (gt-sup, deployable) vs teacher renders
ROOT=${AAR_ROOT:?set AAR_ROOT to the repo root}
ENV=${AAR_ENV:-}            # optional conda prefix; unset means "use the active env"
cd $ROOT
export PYTHONPATH=$ROOT/refs/vggt-omega HF_HOME=$ROOT/hf_cache TORCH_HOME=$ROOT/.torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} TORCH_CUDA_ARCH_LIST=7.5
export PATH=${ENV:+$ENV/bin:}$PATH PYTHONUTF8=1 LANG=C.UTF-8
export CC=${CC:-gcc} CXX=${CXX:-g++} TORCH_EXTENSIONS_DIR=$ROOT/.torch_ext
LOG=$ROOT/logs
run() {
  name=$1; shift
  echo "=== [$name] start $(date +%H:%M:%S) ==="
  ${PYTHON:-python} -u experiments/hybrid_amort.py "$@" > $LOG/$name.log 2>&1 \
    && echo "=== [$name] DONE $(grep RESULT $LOG/$name.log) ===" \
    || echo "=== [$name] FAILED (see $LOG/$name.log) ==="
}
# 1) E87a: mirror back + TEACHER sup  (only variable vs 85d: position source)
run e87a_walk_mirror_teachersup --scene walk --teacher repro_4d_clean --add_back 40000 --mirror_init 1 --densify 1 --iters 4000
# 2) E88c: mirror back + GT sup      (FULL DEPLOY config)
run e88c_walk_mirror_gtsup      --scene walk --teacher repro_4d_clean --add_back 40000 --mirror_init 1 --gt_supervise 1 --densify 1 --iters 4000
# 3) E88b: teacher back + GT sup     (only variable vs 85d: supervision source)
run e88b_walk_teacherback_gtsup --scene walk --teacher repro_4d_clean --add_back 40000 --teacher_init 1 --gt_supervise 1 --densify 1 --iters 4000
# 4) E88a: no back + GT sup          (back-side variable under gt-sup; control vs 85c 18.53)
run e88a_walk_noback_gtsup      --scene walk --teacher repro_4d_clean --gt_supervise 1 --densify 1 --iters 4000
# 5) E88d: deploy config, long       (saturation of the deployable config; gt-sup iters are cheap)
run e88d_walk_mirror_gtsup12k   --scene walk --teacher repro_4d_clean --add_back 40000 --mirror_init 1 --gt_supervise 1 --densify 1 --iters 12000
# 6) E89: deploy config on bear      (cross-scene check)
run e89_bear_mirror_gtsup       --scene bear --teacher retime_bear --add_back 40000 --mirror_init 1 --gt_supervise 1 --densify 1 --iters 4000
echo "=== ALL DONE $(date +%H:%M:%S) ==="
