"""SOTA pruning-criteria baselines on the fitted nine-scene fields, at a fixed budget.
Criteria (training-free, computed on the SAME field, all downstream stages identical):
  lightgs : LightGaussian-style global significance = sum_views hit-area x opacity x volume^0.5
            (hit count approximated by the projected pixel area from the rasterizer's radii)
  contrib : accumulated blending contribution (RadSplat / Mini-Splatting family) =
            d(sum of rendered alpha)/d(alpha_j) x alpha_j summed over views and ref times
Outputs keep sets + held PSNR + topology preservation rate per scene.
Usage: abl_prune_criteria.py --budget 20000 [--scenes walk,...]"""
import os, json, argparse, math, sys, glob, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from gsplat import rasterization
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/aar")
from teacher_retime import load_cams
import gudhi

ap = argparse.ArgumentParser()
ap.add_argument("--budget", type=int, default=20000)
ap.add_argument("--budgets", default="")
ap.add_argument("--ckpt", default="")                     # override the fitted model (e.g. a cleaned field)
ap.add_argument("--tag", default="")                      # suffix for the per-scene output dir                  # comma list; overrides --budget when set
ap.add_argument("--scenes", default="walk,bear,doll,newundress,opencase,passdoll,pickupdoll,putcomputer,stretch")
ap.add_argument("--grid", type=int, default=128); ap.add_argument("--tref", type=int, default=5)
ap.add_argument("--pers", type=float, default=0.05)
ap.add_argument("--res", type=int, default=512); ap.add_argument("--n_frames", type=int, default=33)
a = ap.parse_args(); dev = "cuda"

def cubical_pairs(dens):
    d = np.asarray(dens, np.float64)
    cc = gudhi.CubicalComplex(top_dimensional_cells=(-d).ravel(order="F"), dimensions=list(d.shape))
    cc.persistence()
    reg, ess = cc.cofaces_of_persistence_pairs()
    outp = []
    for dim, arr in enumerate(reg):
        for b, dth in arr:
            bx = np.unravel_index(int(b), d.shape, order="F"); dx = np.unravel_index(int(dth), d.shape, order="F")
            outp.append((dim, float(d[bx]), float(d[dx]), bx, dx))
    for dim, arr in enumerate(ess):
        for b in arr:
            bx = np.unravel_index(int(b), d.shape, order="F")
            outp.append((dim, float(d[bx]), 0.0, bx, bx))
    return outp
def features_of(d, thr):
    return [(dim, bx, dx, bv - dv) for dim, bv, dv, bx, dx in cubical_pairs(d) if dim <= 1 and (bv - dv) >= thr]

RES = {"budget": a.budget, "scenes": {}}
for scene in a.scenes.split(","):
    ck_path = a.ckpt or sorted(glob.glob(f"{ROOT}/experiments/outputs/nine_models/{scene}/model_*_96000.pt"))[0]
    ck = torch.load(ck_path, map_location=dev)
    means, scales, quats = ck["means"].to(dev), ck["scales"].to(dev), ck["quats"].to(dev)
    opac, sh0, shN = ck["opac"].to(dev), ck["sh0"].to(dev), ck["shN"].to(dev)
    mu_t, log_tl, log_tr = ck["mu_t"].to(dev), ck["log_tl"].to(dev), ck["log_tr"].to(dev)
    v1, v2, v3 = ck["v1"].to(dev), ck["v2"].to(dev), ck["v3"].to(dev)
    N = means.shape[0]

    GAMMA = 0.005
    def temporal_alpha(tv, idx=None):
        tl = torch.exp(log_tl if idx is None else log_tl[idx]); tr = torch.exp(log_tr if idx is None else log_tr[idx])
        mt = mu_t if idx is None else mu_t[idx]
        le = mt - tl; re = mt + tr
        left = torch.where(le < 0.02, torch.ones_like(le), torch.sigmoid((tv - le) / GAMMA))
        right = torch.where(re > 0.98, torch.ones_like(re), torch.sigmoid((re - tv) / GAMMA))
        return left * right
    def pos_at(tv, idx=None):
        s = tv - 0.5
        if idx is None: return means + v1 * s + v2 * s * s + v3 * s ** 3
        return means[idx] + v1[idx] * s + v2[idx] * s * s + v3[idx] * s ** 3

    SD = f"{ROOT}/data/sc_mv_all/{scene}"
    frames = sorted([d for d in os.listdir(SD) if d.isdigit()])[:a.n_frames]; T = len(frames)
    im0 = Image.open(f"{SD}/{frames[0]}/00.png"); W0, H0 = im0.size; sc = a.res / max(W0, H0)
    Wr, Hr = int(round(W0 * sc)), int(round(H0 * sc))
    vms, Ks = load_cams(f"{SD}/{frames[0]}/cameras.json", Wr, Hr, dev); ncam = len(vms)
    held_c = list(range(1, ncam, 2)); held_t = list(range(1, T, 2))
    GT = {}
    for c in held_c:
        for t in held_t:
            r = torch.tensor(np.asarray(Image.open(f"{SD}/{frames[t]}/{c:02d}.png").convert("RGB").resize((Wr, Hr))) / 255., dtype=torch.float32)
            m = torch.tensor((np.asarray(Image.open(f"{SD}/{frames[t]}/masks/{c:02d}.png").convert("L").resize((Wr, Hr), Image.NEAREST)) > 127).astype(np.float32))
            GT[(c, t)] = (r, m)
    def render(idx, c, tv):
        al = (torch.sigmoid(opac[idx]) * temporal_alpha(tv, idx)).clamp(1e-4, 1 - 1e-4)
        col, _, _ = rasterization(pos_at(tv, idx), F.normalize(quats[idx], dim=-1),
                                  torch.exp(scales[idx]).clamp(max=0.06), al,
                                  torch.cat([sh0[idx], shN[idx]], 1), vms[c][None], Ks[c][None],
                                  Wr, Hr, sh_degree=3, render_mode="RGB")
        return col[0].clamp(0, 1)
    def eval_both(idx):
        es = []
        with torch.no_grad():
            for c in held_c:
                for t in held_t:
                    r, m = GT[(c, t)]; m3 = m[..., None].to(dev)
                    if m3.sum() < 10: continue
                    es.append((((render(idx, c, t / (T - 1)) - r.to(dev)) ** 2 * m3).sum() / (m3.sum() * 3 + 1e-6)).item())
        return -10 * math.log10(float(np.mean(es)) + 1e-12)

    w_base = torch.sigmoid(opac)
    gscale = torch.exp(scales).clamp(max=0.06)
    trefs = np.linspace(0.10, 0.90, a.tref)

    # ---- criterion scores over training cams x reference times (cached per scene) ----
    MD = f"{ROOT}/experiments/outputs/morse_{scene}{a.tag}"; os.makedirs(MD, exist_ok=True)
    cached = os.path.exists(f"{MD}/score_lightgs.pt") and os.path.exists(f"{MD}/score_contrib.pt")
    score_lg = torch.zeros(N, device=dev)
    score_ct = torch.zeros(N, device=dev)
    CHUNK = 8
    for tv in ([] if cached else trefs):
        act = temporal_alpha(float(tv)) > 0.3
        aidx = torch.nonzero(act).squeeze(1)
        p = pos_at(float(tv), aidx)
        q = F.normalize(quats[aidx], dim=-1); s = gscale[aidx]
        colsh = torch.cat([sh0[aidx], shN[aidx]], 1)
        for c0 in range(0, ncam, CHUNK):
            cs = list(range(c0, min(c0 + CHUNK, ncam)))
            # lightgs: projected pixel area as hit-count proxy (no grad)
            with torch.no_grad():
                al = (w_base[aidx] * temporal_alpha(float(tv), aidx)).clamp(1e-4, 1 - 1e-4)
                _, _, meta = rasterization(p, q, s, al, colsh, vms[cs], Ks[cs], Wr, Hr,
                                           sh_degree=3, render_mode="RGB", packed=False)
                r2 = meta["radii"].float().clamp(min=0).prod(-1)      # [C,N'] projected area rx*ry
                score_lg[aidx] += r2.sum(0)
            # contrib: d(sum alpha map)/d(alpha_j) * alpha_j  (accumulated blend weight)
            al = (w_base[aidx] * temporal_alpha(float(tv), aidx)).clamp(1e-4, 1 - 1e-4).detach().requires_grad_(True)
            _, amap, _ = rasterization(p, q, s, al, colsh, vms[cs], Ks[cs], Wr, Hr,
                                       sh_degree=3, render_mode="RGB", packed=False)
            amap.sum().backward()
            score_ct[aidx] += (al.grad * al).detach()
    if cached:
        score_lg = torch.load(f"{MD}/score_lightgs.pt", map_location=dev)
        score_ct = torch.load(f"{MD}/score_contrib.pt", map_location=dev)
        print(f"[{scene}] loaded cached criterion scores", flush=True)
    else:
        vol = gscale.prod(-1)
        score_lg = score_lg * w_base * vol.pow(0.5)
        torch.save(score_lg.cpu(), f"{MD}/score_lightgs.pt")
        torch.save(score_ct.cpu(), f"{MD}/score_contrib.pt")
    order_lg = torch.argsort(score_lg, descending=True)
    order_ct = torch.argsort(score_ct, descending=True)

    # ---- topology preservation (identical machinery to morse_prune) ----
    gsm = gscale.prod(-1).pow(1 / 3)
    with torch.no_grad():
        allp = torch.cat([pos_at(float(t)) for t in trefs])
        lo = allp.quantile(0.001, 0) - 0.10; hi = allp.quantile(0.999, 0) + 0.10
    vox = (hi - lo).max().item() / a.grid
    def density_at(tv, keep=None):
        with torch.no_grad():
            act = temporal_alpha(float(tv)) > 0.3
            if keep is not None: act = act & keep
            p = pos_at(float(tv), act); w = w_base[act]
            ijk = ((p - lo) / vox).round().long().clamp(0, a.grid - 1)
            lin = (ijk[:, 0] * a.grid + ijk[:, 1]) * a.grid + ijk[:, 2]
            gg = torch.zeros(a.grid ** 3, device=dev).scatter_add_(0, lin, w).view(a.grid, a.grid, a.grid)
            sig = float((gsm[act].median() / vox).clamp(min=1.0))
            r = int(3 * sig); x = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
            k = torch.exp(-x ** 2 / (2 * sig ** 2)); k /= k.sum()
            gg = gg[None, None]
            gg = F.conv3d(gg, k.view(1, 1, -1, 1, 1), padding=(r, 0, 0))
            gg = F.conv3d(gg, k.view(1, 1, 1, -1, 1), padding=(0, r, 0))
            gg = F.conv3d(gg, k.view(1, 1, 1, 1, -1), padding=(0, 0, r))
            return gg[0, 0], act
    refs = {}
    for tv in trefs:
        dgrid, _ = density_at(tv)
        d = dgrid.cpu().numpy(); thr = a.pers * d.max()
        refs[float(tv)] = (features_of(d, thr), thr)
    MATCH_R = 6.0
    def topo_rate(keep_mask):
        tot = ok = 0
        for tv in trefs:
            fr, thr = refs[float(tv)]
            dgrid, _ = density_at(tv, keep_mask)
            fs = features_of(dgrid.cpu().numpy(), 0.5 * thr)
            for dim, bx, dx, pers in fr:
                hit = any(fdim == dim and max(abs(bx[0] - fbx[0]), abs(bx[1] - fbx[1]), abs(bx[2] - fbx[2])) <= MATCH_R
                          and fp >= 0.5 * pers for fdim, fbx, fdx, fp in fs)
                tot += 1; ok += int(hit)
        return ok / max(tot, 1)

    row = {}
    budgets = [int(x) for x in a.budgets.split(",")] if a.budgets else [a.budget]
    for bg in budgets:
        for name, order in [("lightgs", order_lg), ("contrib", order_ct)]:
            key = name if len(budgets) == 1 else f"{name}_{bg}"
            t0 = time.time()
            kp = order[:bg]
            mask = torch.zeros(N, dtype=torch.bool, device=dev); mask[kp] = True
            p_ = eval_both(kp); tr_ = topo_rate(mask)
            row[key] = {"both": p_, "topo_rate": tr_, "budget": bg}
            torch.save(kp.cpu(), f"{MD}/keep_{name}_{bg}.pt")
            print(f"[{scene}] {name}@{bg}: both={p_:.2f} topo={tr_:.2f} ({time.time()-t0:.0f}s)", flush=True)
    RES["scenes"][scene] = row
    del ck, means, scales, quats, opac, sh0, shN, mu_t, log_tl, log_tr, v1, v2, v3, GT
    torch.cuda.empty_cache()

SC = list(RES["scenes"].keys())
keys = sorted({k for s in SC for k in RES["scenes"][s]})
for name in keys:
    ps = [RES["scenes"][s][name]["both"] for s in SC if name in RES["scenes"][s]]
    tr_ = [RES["scenes"][s][name]["topo_rate"] for s in SC if name in RES["scenes"][s]]
    RES[f"{name}_mean"] = {"both": float(np.mean(ps)), "both_std": float(np.std(ps)), "topo": float(np.mean(tr_))}
    print(f"[MEAN] {name}: both={np.mean(ps):.2f}+-{np.std(ps):.2f} topo={np.mean(tr_):.2f}")
tagf = a.budgets.replace(",", "-") if a.budgets else str(a.budget)
json.dump(RES, open(f"{ROOT}/experiments/outputs/abl_prune_criteria_{tagf}.json", "w"), indent=1)
print("[saved]", f"{ROOT}/experiments/outputs/abl_prune_criteria_{tagf}.json")
