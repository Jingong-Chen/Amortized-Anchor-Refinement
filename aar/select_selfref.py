"""Sender-side package selection without ground truth.

Every candidate ranking (opacity, visibility mass, RadSplat contribution, LightGaussian significance)
is pruned to the same budget. Two quantities are then measured at the sender, using only the fitted
field it already holds:
  - topology preservation rate: fraction of eps-persistent features that survive pruning;
  - self-fidelity: PSNR of the pruned renders against renders of the *full field* (no ground truth).
The package that maximises self-fidelity among the candidates whose preservation rate is within
`--rate_tol` of the best is transmitted. The held-out PSNR against ground truth is reported here only
to check the decision after the fact; it is not available to the selector.

Usage: select_selfref.py --scenes walk,doll --budgets 20000,10000 [--clean 1]
"""
import os, sys, json, math, glob, argparse
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from gsplat import rasterization
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/aar")
from teacher_retime import load_cams

ap = argparse.ArgumentParser()
ap.add_argument("--scenes", default="walk,doll")
ap.add_argument("--budgets", default="20000,10000")
ap.add_argument("--clean", type=int, default=1)           # use the cleaned fields
ap.add_argument("--rate_tol", type=float, default=0.02)   # candidates within this of the best rate stay eligible
ap.add_argument("--cams", type=int, default=6); ap.add_argument("--times", type=int, default=6)
ap.add_argument("--res", type=int, default=512); ap.add_argument("--n_frames", type=int, default=33)
a = ap.parse_args(); dev = "cuda"

def keep_paths(scene, bg):
    """Candidate keep sets, with the verified floor already applied where we produce it."""
    base = f"{ROOT}/experiments/outputs"
    c = {}
    if a.clean:
        c["opacity"] = f"{base}/clean_{scene}/morse/keep_topo_{bg}.pt"
        c["contrib"] = f"{base}/morse_{scene}_clean/keep_contrib_{bg}.pt"
        c["lightgs"] = f"{base}/morse_{scene}_clean/keep_lightgs_{bg}.pt"
    else:
        c["opacity"] = f"{base}/morse_{scene}_opacity/keep_topo_{bg}.pt"
        c["mass"] = f"{base}/morse_{scene}/keep_topo_{bg}.pt"
        c["contrib"] = f"{base}/morse_{scene}/keep_contrib_{bg}.pt"
        c["lightgs"] = f"{base}/morse_{scene}/keep_lightgs_{bg}.pt"
    return {k: v for k, v in c.items() if os.path.exists(v)}

out_all = {}
for scene in a.scenes.split(","):
    ckp = (f"{ROOT}/experiments/outputs/clean_{scene}/model_cleaned_96000.pt" if a.clean
           else sorted(glob.glob(f"{ROOT}/experiments/outputs/nine_models/{scene}/model_*_96000.pt"))[0])
    ck = torch.load(ckp, map_location=dev)
    means, scales, quats = ck["means"], ck["scales"], ck["quats"]
    opac, sh0, shN = ck["opac"], ck["sh0"], ck["shN"]
    mu_t, ltl, ltr = ck["mu_t"], ck["log_tl"], ck["log_tr"]
    v1, v2, v3 = ck["v1"], ck["v2"], ck["v3"]
    N = means.shape[0]

    SD = f"{ROOT}/data/sc_mv_all/{scene}"
    frames = sorted([d for d in os.listdir(SD) if d.isdigit()])[:a.n_frames]; T = len(frames)
    im0 = Image.open(f"{SD}/{frames[0]}/00.png"); W0, H0 = im0.size; sc = a.res / max(W0, H0)
    Wr, Hr = int(round(W0 * sc)), int(round(H0 * sc))
    vms, Ks = load_cams(f"{SD}/{frames[0]}/cameras.json", Wr, Hr, dev); ncam = len(vms)
    held_c = list(range(1, ncam, 2))[::max(1, len(range(1, ncam, 2)) // a.cams)][:a.cams]
    held_t = list(range(1, T, 2))[::max(1, len(range(1, T, 2)) // a.times)][:a.times]
    GAMMA = 0.005

    def render(idx, c, tv):
        tl = torch.exp(ltl[idx]); tr = torch.exp(ltr[idx]); mt = mu_t[idx]
        le = mt - tl; re = mt + tr
        ta = torch.where(le < 0.02, torch.ones_like(le), torch.sigmoid((tv - le) / GAMMA)) * \
             torch.where(re > 0.98, torch.ones_like(re), torch.sigmoid((re - tv) / GAMMA))
        s = tv - 0.5
        pos = means[idx] + v1[idx] * s + v2[idx] * s * s + v3[idx] * s ** 3
        al = (torch.sigmoid(opac[idx]) * ta).clamp(1e-4, 1 - 1e-4)
        col, _, _ = rasterization(pos, F.normalize(quats[idx], dim=-1), torch.exp(scales[idx]).clamp(max=0.06),
                                  al, torch.cat([sh0[idx], shN[idx]], 1), vms[c][None], Ks[c][None],
                                  Wr, Hr, sh_degree=3, render_mode="RGB")
        return col[0].clamp(0, 1)

    full_idx = torch.arange(N, device=dev)
    REF, GT = {}, {}
    with torch.no_grad():
        for c in held_c:
            for t in held_t:
                REF[(c, t)] = render(full_idx, c, t / (T - 1))
                g = torch.tensor(np.asarray(Image.open(f"{SD}/{frames[t]}/{c:02d}.png").convert("RGB").resize((Wr, Hr))) / 255., dtype=torch.float32, device=dev)
                m = torch.tensor((np.asarray(Image.open(f"{SD}/{frames[t]}/masks/{c:02d}.png").convert("L").resize((Wr, Hr), Image.NEAREST)) > 127).astype(np.float32), device=dev)[..., None]
                GT[(c, t)] = (g, m)

    def scores(idx):
        se_ref, se_gt = [], []
        with torch.no_grad():
            for c in held_c:
                for t in held_t:
                    r = render(idx, c, t / (T - 1))
                    se_ref.append(((r - REF[(c, t)]) ** 2).mean().item())
                    g, m = GT[(c, t)]
                    if m.sum() >= 10:
                        se_gt.append((((r - g) ** 2 * m).sum() / (m.sum() * 3 + 1e-6)).item())
        p = lambda v: -10 * math.log10(float(np.mean(v)) + 1e-12)
        return p(se_ref), p(se_gt)

    for bg in [int(x) for x in a.budgets.split(",")]:
        cand = keep_paths(scene, bg)
        if not cand: print(f"[{scene}@{bg}] no candidates"); continue
        rows = {}
        for name, path in cand.items():
            idx = torch.load(path, map_location=dev)
            self_p, gt_p = scores(idx)
            rows[name] = {"self": self_p, "gt": gt_p, "n": int(idx.numel())}
            print(f"[{scene}@{bg}] {name:<8} self-fidelity {self_p:6.2f} | held PSNR {gt_p:6.2f}", flush=True)
        pick = max(rows, key=lambda k: rows[k]["self"])
        best_gt = max(rows, key=lambda k: rows[k]["gt"])
        print(f"[{scene}@{bg}] selector picks {pick} ({rows[pick]['gt']:.2f} dB); best possible {best_gt} "
              f"({rows[best_gt]['gt']:.2f} dB); regret {rows[best_gt]['gt']-rows[pick]['gt']:.2f} dB", flush=True)
        out_all[f"{scene}@{bg}"] = {"rows": rows, "pick": pick, "best": best_gt}
json.dump(out_all, open(f"{ROOT}/experiments/outputs/select_selfref{'_clean' if a.clean else ''}.json", "w"), indent=1)
print("[saved] select_selfref.json")
