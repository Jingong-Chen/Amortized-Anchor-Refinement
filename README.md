# Amortized Anchor Refinement

Code for *Amortized Anchor Refinement for Deployable Continuous-Time 4D Gaussian
Reconstruction*.

A dynamic scene is reconstructed as a continuous-time 4D Gaussian field under a fixed
deployment budget, then compressed into a few-megabyte package that a standalone XR
headset decodes in shader. The pipeline has three stages:

| Stage | What it does | Entry point |
| --- | --- | --- |
| **GFA** — Grounded Feed-forward Anchor | A frozen VGGT-Ω predicts per-view geometry; rays are grounded into Gaussians (K per ray) and the unobserved back side is filled by mirror symmetry | `aar/hybrid_amort.py` |
| **CCR** — Capacity-Constrained Refinement | Short per-scene optimization against the captured views, holding at least `N_min` Gaussians so the densify-and-prune schedule cannot strip the field | `aar/hybrid_amort.py` |
| **PHPC** — Persistent-Homology Pruning and Compression | Training-free pruning with a verified topology floor, then trajectory coding into the streamed package | `aar/morse_prune.py`, `aar/pack_webxr.py` |

## Layout

```
aar/           the method: anchor, refinement, pruning, packaging
data_tools/    convert a capture into the directory layout the pipeline reads
eval/          metrics and result collection, shared by our runs and the baselines
figures/       every figure in the paper, regenerated from run outputs
scripts/       end-to-end runners for the paper's experiments
viewer/        the WebGL/WebXR client that decodes a package on a headset
```

Paths resolve relative to the repository root. Override with `AAR_ROOT` if you keep
datasets and outputs elsewhere:

```bash
export AAR_ROOT=/path/to/your/workspace
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
```

The feed-forward anchor needs [VGGT-Ω](https://github.com/facebookresearch/vggt-omega) on the
import path:

```bash
git clone https://github.com/facebookresearch/vggt-omega refs/vggt-omega
export PYTHONPATH=$PWD/refs/vggt-omega:$PYTHONPATH
```

## Data

No capture data ships with this repository.

**Stage-Capture** is the benchmark released with RetimeGS. Place it as
`data/sc_mv_all/<scene>/<frame>/` with `<cam>.png`, `masks/<cam>.png` and a
`cameras.json` per frame.

**DNA-Rendering** ships as SenseMoCap `.smc` pairs. Convert them with:

```bash
python data_tools/dna_to_stage.py --scene 0008_01 --probe 1    # check the calibration first
python data_tools/dna_to_stage.py --scene 0008_01 --cams 32 --frames 33
```

`--probe` reports, for both readings of the stored extrinsics, how many annotated 3D
joints land inside the person mask. The release stores camera-to-world; the opposite
reading collapses every camera centre onto one point. The converter also blanks each
view outside the radius where its radial distortion polynomial is still monotonic —
past that radius the rectification map folds and duplicates image content into a border
ring, which enters the masks as foreground the reconstruction cannot explain.

## Reconstruct a scene

```bash
python aar/hybrid_amort.py --scene walk \
  --iters 96000 --res 512 --n_frames 33 --cpu_data 1 \
  --gh 64 --gw 88 --K 2 --input_res 256 \
  --densify 1 --min_n 200000 --densify_until 0.7 --antifloat 1 --tau_mul 1.5 \
  --gt_supervise 1 --all_cam_gt 1 --ssim_w 0.2 --lpips 0.1 --lr_decay 1 \
  --ff_dtype fp32 --ff_offload 1 --seed 0 --save_model 1 --eval_perceptual 1 \
  --out outputs/walk
```

`--ff_offload 1` keeps the frozen backbone on the CPU and streams layers to the GPU, which
is what holds the end-to-end footprint inside the deployment budget.

Densification runs over the first 70% of the steps. On scenes whose Gaussian count grows
without settling, use `--densify_until 0.35`; the count alone decides, no held-out frame is
consulted. Without it those scenes diverge rather than degrade.

For a portrait capture such as DNA-Rendering, match the anchor ray grid to the aspect
ratio (`--gh 88 --gw 74`) and point the loader at that dataset root
(`--data_root data/dna_rendering`).

## Compress and stream

```bash
# prune to a ladder of budgets, with the verified topology floor
python aar/morse_prune.py --scene walk --ckpt outputs/walk/model_*.pt \
  --targets 80000,40000,20000,10000,5000 --imp_mode opacity

# the published ranking baselines, on the same field and coder
python aar/abl_prune_criteria.py --scenes walk --budgets 20000

# pick a package per scene by its fidelity to the unpruned field, without ground truth
python aar/select_selfref.py

# pack into the streamed binary (80 bytes/splat, DC-only colour)
python aar/pack_webxr.py --ckpt outputs/walk/model_*.pt \
  --keep outputs/morse_walk/keep_topo_40000.pt --scene walk \
  --maskcull 1 --dilate 25 --out viewer/field.bin
```

Serve `viewer/` over HTTP and open it on the headset. `viewer/launch_quest.sh` sets up the
`adb reverse` tunnel for a Meta Quest. `aar/shoot_viewer.py` drives the same client
headlessly through Chrome DevTools to capture figure frames.

## Reproduce the paper

```bash
export AAR_ROOT=$PWD
bash scripts/run_nine_quality.sh      # ours, nine scenes
bash scripts/run_dgs33_nine.sh        # Deformable-3DGS
bash scripts/run_4dgs_nine.sh         # 4D-GS
bash scripts/run_gflow_nine.sh        # GaussianFlow
bash scripts/run_teacher_nine.sh      # RetimeGS
bash scripts/run_tfs_nine.sh          # pruning and streaming
bash scripts/run_dna_baselines.sh     # DNA-Rendering cross-dataset
python eval/aggregate_nine.py         # per-scene table
python eval/collect_dna.py            # cross-dataset table rows
```

The baseline runners expect the public implementations under `refs/`. Every number comes
from `eval/eval_dgs33.py`, so our runs and the baselines are scored by the same code.

Figures regenerate from run outputs, e.g.
`python figures/fig_appendix_dna.py --cams 16,3`.

## Notes

- Run-to-run variation from nondeterministic CUDA atomics in `gsplat` is within ±0.3 dB
  with a fixed seed.
- `gsplat` compiles CUDA kernels on first import. If `nvcc` rejects your host compiler,
  point `CC`/`CXX` at one the CUDA release accepts before running anything.

## Citation

Paper under review; citation to follow.

## License

Released under the MIT License; see [LICENSE](LICENSE).
