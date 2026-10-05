#!/bin/bash
# Measure PEAK fitting GPU memory (nvidia-smi, per-process) for every method on `walk`,
# under the SAME config that produced the paper's #G/quality. Kills each run once the
# peak plateaus (densification done) so we don't train to convergence. Framework-agnostic:
# sums GPU memory of all compute-apps that appear AFTER launch (i.e. our job + its forks),
# so the pre-existing 6.4GB root job never contaminates the measurement.
set -u
ROOT=${AAR_ROOT:?set AAR_ROOT to the repo root}
ENV=${AAR_ENV:-}            # optional conda prefix; unset means "use the active env"; PY=${PYTHON:-python}
RES=$ROOT/experiments/outputs/fit_vram.txt
LOGD=$ROOT/experiments/outputs/fitvram_logs; mkdir -p "$LOGD"
: > "$RES"
export PYTHONPATH=$ROOT/refs/vggt-omega HF_HOME=$ROOT/hf_cache TORCH_HOME=$ROOT/.torch \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} TORCH_CUDA_ARCH_LIST=7.5 \
  PATH=${ENV:+$ENV/bin:}$PATH PYTHONUTF8=1 LANG=C.UTF-8 \
  CC=${CC:-gcc} CXX=${CXX:-g++} \
  TORCH_EXTENSIONS_DIR=$ROOT/.torch_ext SC_NFRAMES=33

pids_now(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }
mem_of(){ # sum used_memory (MiB) over PID list $1
  local want="$1"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
   | tr -d ' ' | awk -F, -v W="$want" 'BEGIN{s=0} {if(index(","W",", ","$1",")>0) s+=$2} END{print s}'
}

# run <name> <min_runtime_s> <workdir> <cmd...>
measure(){
  local name="$1" minrt="$2" wd="$3"; shift 3
  echo "===== $name start $(date +%H:%M:%S) ====="
  local before; before=",$(pids_now | paste -sd, ),"
  ( cd "$wd" && exec "$@" ) >"$LOGD/$name.log" 2>&1 &
  local lpid=$!
  local peak=0 stable=0 t=0 last=0
  sleep 20   # let CUDA context + model come up
  while kill -0 "$lpid" 2>/dev/null; do
    # PIDs that are new since launch = our job (+forks)
    local cur newl=""
    cur=$(pids_now)
    for p in $cur; do case "$before" in *",$p,"*) ;; *) newl="$newl,$p";; esac; done
    local m; m=$(mem_of "${newl#,}")
    [ -z "$m" ] && m=0
    if [ "$m" -gt "$peak" ]; then peak=$m; fi
    # plateau: peak unchanged for ~120s AND past min runtime
    if [ "$peak" -eq "$last" ]; then stable=$((stable+5)); else stable=0; last=$peak; fi
    t=$((t+5))
    if [ "$t" -ge "$minrt" ] && [ "$stable" -ge 120 ] && [ "$peak" -gt 0 ]; then
      echo "  plateau at ${peak}MiB after ${t}s"; break
    fi
    if [ "$t" -ge 1800 ]; then echo "  hard cap 1800s"; break; fi
    sleep 5
  done
  kill "$lpid" 2>/dev/null; pkill -P "$lpid" 2>/dev/null
  # also kill any leftover new python on GPU
  sleep 8
  local gb; gb=$(awk -v m="$peak" 'BEGIN{printf "%.2f", m/1024}')
  printf "%-10s %6d MiB  = %s GB  (walk, %ds)\n" "$name" "$peak" "$gb" "$t" | tee -a "$RES"
  sleep 12   # let VRAM release before next
}

# 1) OURS 33f — peak is at feed-forward init (early); short run captures it
measure ours33 90 "$ROOT" $PY -u experiments/hybrid_amort.py --scene walk --n_frames 33 \
  --cpu_data 1 --densify 1 --gt_supervise 1 --all_cam_gt 1 --mirror_init 1 --add_back 40000 \
  --antifloat 1 --tau_mul 1.5 --min_n 200000 --ssim_w 0.2 --lpips 0.1 --lr_decay 1 --ff_offload 1 \
  --iters 96000 --seed 0 --out "$LOGD/ours33_out"

# 2) GaussianFlow repro (retimegs_4d.py, cap 1.28M, flow) — matches model's 1.28M
measure gflow 200 "$ROOT" $PY -u experiments/retimegs_4d.py --root $ROOT/data/sc_mv_all --scene walk \
  --nframes 33 --res 512 --iters 30000 --cap 1280000 --flow 20 --out "$LOGD/gflow_out"

# 3) RetimeGS teacher (full: cr/triple/lpips/vreg, cap 1.28M)
measure retimegs 200 "$ROOT" $PY -u experiments/retimegs_4d.py --root $ROOT/data/sc_mv_all --scene walk \
  --nframes 33 --res 512 --iters 60000 --cap 1280000 --flow 20 --cr 1 --vreg 1.0 --lpips 0.1 \
  --lr_decay 1 --triple 0.3 --out "$LOGD/retime_out"

# 4) Deformable-3DGS (official)
measure deform 200 "$ROOT/refs/Deformable-3D-Gaussians" $PY -u train.py \
  -s $ROOT/data/sc_mv_all/walk -m "$LOGD/deform_out" --eval -r 2 --iterations 40000

# 5) 4D-GS (official)
measure fourdgs 200 "$ROOT/refs/4DGaussians" $PY -u train.py \
  -s $ROOT/data/sc_mv_all/walk -m "$LOGD/fourdgs_out" --configs arguments/stagecapture/default.py --expname sc_walk

echo "===== ALL DONE $(date +%H:%M:%S) ====="
cat "$RES"
