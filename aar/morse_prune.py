"""Morse/persistence-guided Gaussian pruning (the topology-floor half of the streaming module).

Idea: prune the fitted 4D field to a small transmission budget. Pure importance pruning
(opacity x scale) kills THIN STRUCTURES first (wrists, ankles) because they are low-mass --
exactly the topology of the subject. We compute the density field of the fitted Gaussians,
extract its persistence diagram (Morse theory on the superlevel filtration), and mark the
critical cells of high-persistence features (H0 merge saddles = thin bridges like a wrist;
H1 loop births) as a PROTECTED set: pruning may never remove Gaussians supporting them.
Same #G budget, two criteria:  importance-only  vs  importance + topology floor.

Usage: morse_prune.py --scene walk --ckpt <model.pt> --targets 80000,40000,20000,10000
Outputs: <out>/morse_prune_<scene>.json + side-by-side renders at the smallest budget.
"""
import os, json, argparse, math, sys, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from gsplat import rasterization
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/aar")
from teacher_retime import load_cams

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="walk")
ap.add_argument("--ckpt", default="")
ap.add_argument("--targets", default="80000,40000,20000,10000")
ap.add_argument("--grid", type=int, default=128)          # density voxelization resolution
ap.add_argument("--tref", type=int, default=5)            # reference timestamps for (time-varying) topology
ap.add_argument("--pers", type=float, default=0.05)       # persistence threshold, fraction of max density
ap.add_argument("--protect_r", type=float, default=3.0)   # protection radius around critical cells (voxels)
ap.add_argument("--quota", type=int, default=150)         # protect at most top-quota Gaussians per critical cell
                                                          # (a FLOOR: guarantee bridges survive; budget stays with appearance)
ap.add_argument("--score_path", default="")               # precomputed ranking score (defaults to morse_<scene>/score_<mode>.pt)
ap.add_argument("--accept_delta", type=float, default=0.01)  # accept the restored set only if the preservation rate
                                                          # improves by at least this margin; a restoration that buys a
                                                          # negligible rate gain still costs appearance mass -> reject it
ap.add_argument("--restore_frac", type=float, default=0.30)  # cap total restoration at this fraction of the budget
                                                          # (the floor may never displace the appearance mass entirely)
ap.add_argument("--res", type=int, default=512)
ap.add_argument("--n_frames", type=int, default=33)
ap.add_argument("--imp_mode", default="mass", choices=["mass", "opacity", "contrib", "lightgs"])
                                                          # base ranking: opacity x scale (mass), opacity alone,
                                                          # or a precomputed score (morse_<scene>/score_<mode>.pt from abl_prune_criteria)
ap.add_argument("--out", default="")
a = ap.parse_args(); dev = "cuda"
out = a.out or f"{ROOT}/experiments/outputs/morse_{a.scene}" + ("" if a.imp_mode == "mass" else f"_{a.imp_mode}")
os.makedirs(out, exist_ok=True)

# ---------------- gudhi orientation self-test (two blobs + bridge) ----------------
import gudhi
def cubical_pairs(dens):
    """Superlevel persistence of a 3D density grid. Returns list of (dim, birth_val, death_val,
    birth_xyz, death_xyz) with values in density units and locations in voxel coords."""
    d = np.asarray(dens, np.float64)
    cc = gudhi.CubicalComplex(top_dimensional_cells=(-d).ravel(order="F"), dimensions=list(d.shape))
    cc.persistence()
    reg, ess = cc.cofaces_of_persistence_pairs()
    outp = []
    for dim, arr in enumerate(reg):
        for b, dth in arr:
            bx = np.unravel_index(int(b), d.shape, order="F"); dx = np.unravel_index(int(dth), d.shape, order="F")
            outp.append((dim, float(d[bx]), float(d[dx]), bx, dx))
    for dim, arr in enumerate(ess):                       # essential class (the whole subject): no death
        for b in arr:
            bx = np.unravel_index(int(b), d.shape, order="F")
            outp.append((dim, float(d[bx]), 0.0, bx, bx))
    return outp

def _self_test():
    g = np.zeros((40, 40, 40), np.float64)
    xx, yy, zz = np.meshgrid(*[np.arange(40)] * 3, indexing="ij")
    for cx in (10, 30):                                   # two blobs on the x-axis
        g += np.exp(-(((xx - cx) ** 2 + (yy - 20) ** 2 + (zz - 20) ** 2) / (2 * 3.0 ** 2)))
    g += 0.25 * np.exp(-(((xx - 20) ** 2) / (2 * 4 ** 2) + ((yy - 20) ** 2 + (zz - 20) ** 2) / (2 * 1.5 ** 2)))  # weak bridge
    prs = [p for p in cubical_pairs(g) if p[0] == 0]
    prs.sort(key=lambda p: -(p[1] - p[2]))
    okA = len(prs) >= 2                                    # 2 components: one essential + one merged pair
    sad = prs[1][4] if len(prs) > 1 else (0, 0, 0)
    okB = abs(sad[0] - 20) <= 4                            # merge saddle sits on the bridge (x ~ 20)
    print(f"[self-test] H0 pairs={len(prs)} saddle_x={sad[0]} -> {'OK' if (okA and okB) else 'MISMATCH'}")
    assert okA and okB, "gudhi orientation self-test failed"
_self_test()

# ---------------- load fitted model ----------------
ck = torch.load(a.ckpt, map_location=dev)
means, scales, quats = ck["means"].to(dev), ck["scales"].to(dev), ck["quats"].to(dev)
opac, sh0, shN = ck["opac"].to(dev), ck["sh0"].to(dev), ck["shN"].to(dev)
mu_t, log_tl, log_tr = ck["mu_t"].to(dev), ck["log_tl"].to(dev), ck["log_tr"].to(dev)
v1, v2, v3 = ck["v1"].to(dev), ck["v2"].to(dev), ck["v3"].to(dev)
N = means.shape[0]; print(f"[model] {a.ckpt}  N={N}")

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

# ---------------- eval protocol (identical to hybrid_amort: held cams x held times, fg PSNR) ----------------
SD = f"{ROOT}/data/sc_mv_all/{a.scene}"
frames = sorted([d for d in os.listdir(SD) if d.isdigit()])[:a.n_frames]; T = len(frames)
im0 = Image.open(f"{SD}/{frames[0]}/00.png"); W0, H0 = im0.size; sc = a.res / max(W0, H0)
Wr, Hr = int(round(W0 * sc)), int(round(H0 * sc))
vms, Ks = load_cams(f"{SD}/{frames[0]}/cameras.json", Wr, Hr, dev); ncam = len(vms)
held_c = list(range(1, ncam, 2)); held_t = list(range(1, T, 2))
print(f"[eval] {len(held_c)} held cams x {len(held_t)} held times @ {Wr}x{Hr}")
GT = {}
for ci in held_c:
    for ti in held_t:
        rgbp = f"{SD}/{frames[ti]}/{ci:02d}.png"; mp = f"{SD}/{frames[ti]}/masks/{ci:02d}.png"
        r = torch.tensor(np.asarray(Image.open(rgbp).convert("RGB").resize((Wr, Hr))) / 255., dtype=torch.float32)
        m = torch.tensor((np.asarray(Image.open(mp).convert("L").resize((Wr, Hr), Image.NEAREST)) > 127).astype(np.float32))
        GT[(ci, ti)] = (r, m)

def render(idx, ci, tv):
    al = (torch.sigmoid(opac[idx]) * temporal_alpha(tv, idx)).clamp(1e-4, 1 - 1e-4)
    col, _, _ = rasterization(pos_at(tv, idx), F.normalize(quats[idx], dim=-1),
                              torch.exp(scales[idx]).clamp(max=0.06), al,
                              torch.cat([sh0[idx], shN[idx]], 1), vms[ci][None], Ks[ci][None],
                              Wr, Hr, sh_degree=3, render_mode="RGB")
    return col[0].clamp(0, 1)
def psnr(m): return -10 * math.log10(m + 1e-12)
def eval_both(idx):
    es = []
    with torch.no_grad():
        for ci in held_c:
            for ti in held_t:
                r, m = GT[(ci, ti)]; m3 = m[..., None].to(dev)
                if m3.sum() < 10: continue
                es.append((((render(idx, ci, ti / (T - 1)) - r.to(dev)) ** 2 * m3).sum() / (m3.sum() * 3 + 1e-6)).item())
    return psnr(float(np.mean(es)))

# ---------------- density voxelization + persistence -> protected set ----------------
w_base = torch.sigmoid(opac)                              # opacity mass
gscale = torch.exp(scales).clamp(max=0.06).prod(-1).pow(1 / 3)   # geometric-mean scale (world)
trefs = np.linspace(0.10, 0.90, a.tref)                   # interior times: at the clip boundary the temporal
                                                          # windows fragment the density (thousands of spurious pairs)
# fg bbox over reference times (fitted fields carry few floaters after antifloat; pad generously)
with torch.no_grad():
    allp = torch.cat([pos_at(float(t)) for t in trefs])
    lo = allp.quantile(0.001, 0) - 0.10; hi = allp.quantile(0.999, 0) + 0.10
vox = (hi - lo).max().item() / a.grid
print(f"[grid] {a.grid}^3, voxel={vox * 1000:.1f}mm, bbox={[round(v,2) for v in (hi-lo).tolist()]}")

def density_at(tv, keep=None):
    """Scatter opacity mass to nearest voxel, then blur with the median Gaussian sigma."""
    with torch.no_grad():
        act = temporal_alpha(float(tv)) > 0.3
        if keep is not None: act = act & keep
        p = pos_at(float(tv), act); w = w_base[act]
        ijk = ((p - lo) / vox).round().long().clamp(0, a.grid - 1)
        lin = (ijk[:, 0] * a.grid + ijk[:, 1]) * a.grid + ijk[:, 2]
        g = torch.zeros(a.grid ** 3, device=dev).scatter_add_(0, lin, w).view(a.grid, a.grid, a.grid)
        sig = float((gscale[act].median() / vox).clamp(min=1.0))
        r = int(3 * sig); x = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
        k = torch.exp(-x ** 2 / (2 * sig ** 2)); k /= k.sum()
        g = g[None, None]
        g = F.conv3d(g, k.view(1, 1, -1, 1, 1), padding=(r, 0, 0))
        g = F.conv3d(g, k.view(1, 1, 1, -1, 1), padding=(0, r, 0))
        g = F.conv3d(g, k.view(1, 1, 1, 1, -1), padding=(0, 0, r))
        return g[0, 0], act

if a.imp_mode in ("contrib", "lightgs"):
    importance = torch.load(a.score_path or f"{ROOT}/experiments/outputs/morse_{a.scene}/score_{a.imp_mode}.pt", map_location=dev).detach()
    assert importance.shape[0] == N, "precomputed score does not match this checkpoint"
else:
    importance = (w_base if a.imp_mode == "opacity" else w_base * gscale).detach()  # visibility mass or opacity alone
order = torch.argsort(importance, descending=True)

# reference topology of the FULL field: high-persistence features per reference time
def features_of(d, thr):
    return [(dim, bx, dx, bv - dv) for dim, bv, dv, bx, dx in cubical_pairs(d)
            if dim <= 1 and (bv - dv) >= thr]
refs = {}
_t0 = time.time()
for tv in trefs:
    dgrid, _ = density_at(tv)
    d = dgrid.cpu().numpy(); thr = a.pers * d.max()
    refs[float(tv)] = (features_of(d, thr), thr)
    print(f"[topo-ref] t={tv:.2f}: {len(refs[float(tv)][0])} high-persistence features (thr={thr:.2f})")
REF_SEC = time.time() - _t0                               # one-time server-side cost of the reference diagrams

MATCH_R = 6.0                                             # feature match radius (voxels)
def topo_rate(keep_mask, detail=False):
    """Topology preservation rate: fraction of the full field's high-persistence features that
    survive in the subset (same dim, birth cell within MATCH_R voxels, >=0.5x persistence)."""
    tot = ok = 0; died = {}
    for tv in trefs:
        fr, thr = refs[float(tv)]
        dgrid, _ = density_at(tv, keep_mask)
        fs = features_of(dgrid.cpu().numpy(), 0.5 * thr)
        dd = []
        for dim, bx, dx, pers in fr:
            hit = any(fdim == dim and max(abs(bx[0] - fbx[0]), abs(bx[1] - fbx[1]), abs(bx[2] - fbx[2])) <= MATCH_R
                      and fp >= 0.5 * pers for fdim, fbx, fdx, fp in fs)
            tot += 1; ok += int(hit)
            if not hit: dd.append((dim, bx, dx, pers))
        died[float(tv)] = dd
    return (ok / max(tot, 1), died) if detail else ok / max(tot, 1)

# ---------------- prune at each budget: importance-only vs topology floor (prune-verify-restore) ----------------
res = {"scene": a.scene, "ckpt": a.ckpt, "N_full": int(N), "n_ref_feats": sum(len(v[0]) for v in refs.values()),
       "pers": a.pers, "grid": a.grid, "tref": a.tref, "protect_r": a.protect_r, "runs": []}
full_idx = torch.arange(N, device=dev)
print("[eval] full model ..."); full_psnr = eval_both(full_idx)
res["full_both"] = full_psnr; print(f"[full] both={full_psnr:.2f}  N={N}")

targets = [int(x) for x in a.targets.split(",")]
keep_sets = {}
RESTORE_Q = a.quota                                        # Gaussians restored per died feature per round
for tgt in targets:
    imp_keep = order[:tgt]                                 # baseline: pure importance
    inkeep = torch.zeros(N, dtype=torch.bool, device=dev); inkeep[imp_keep] = True
    r_imp = topo_rate(inkeep)
    # ---- topology floor: prune-then-VERIFY, restore only features that DIED, budget held fixed
    _tf0 = time.time()
    keep_mask = inkeep.clone(); restored = torch.zeros(N, dtype=torch.bool, device=dev)
    rbudget = int(a.restore_frac * tgt)                    # the floor may spend at most this many Gaussians
    for rnd in range(3):
        if int(restored.sum()) >= rbudget: break
        rate, died = topo_rate(keep_mask, detail=True)
        allfeats = [(pers, tv, bx, dx) for tv, feats in died.items() for dim, bx, dx, pers in feats]
        if not allfeats: break
        allfeats.sort(key=lambda x: -x[0])                 # most persistent (salient) structures first
        for pers, tv, bx, dx in allfeats:
            room = rbudget - int(restored.sum())
            if room <= 0: break
            act = (temporal_alpha(float(tv)) > 0.3) & (~keep_mask)
            aidx = torch.nonzero(act).squeeze(1)
            if aidx.numel() == 0: continue
            p = pos_at(float(tv), act)
            cells = torch.tensor(np.array([bx, dx], np.float32), device=dev) * vox + lo[None]
            dmin = torch.cdist(p, cells).min(-1).values    # restore mass at BOTH the peak and its saddle
            cand = torch.nonzero(dmin < a.protect_r * vox).squeeze(1)
            if cand.numel() == 0: continue
            gidx = aidx[cand]
            top = gidx[torch.argsort(importance[gidx], descending=True)[:min(RESTORE_Q, room)]]
            keep_mask[top] = True; restored[top] = True
        over = int(keep_mask.sum()) - tgt                  # hold budget: evict least-important non-restored
        if over > 0:
            kept = torch.nonzero(keep_mask & ~restored).squeeze(1)
            kept = kept[torch.argsort(importance[kept])]
            keep_mask[kept[:over]] = False
    _tfsec = time.time() - _tf0                            # verify+restore cost only (excludes PSNR evals)
    r_topo = topo_rate(keep_mask)
    # RE-VERIFY with fallback: on saturated scenes restore-evict is zero-sum (restoring feature A
    # evicts the support of feature C -> whack-a-mole); keep the floor ONLY if preservation strictly
    # improved, else fall back to plain importance -- the floor never does harm. Uses no GT.
    if r_topo < r_imp + a.accept_delta:
        keep_mask = inkeep.clone(); restored.zero_(); r_topo = r_imp
    topo_keep = torch.nonzero(keep_mask).squeeze(1)
    b1 = eval_both(imp_keep); b2 = eval_both(topo_keep)
    keep_sets[tgt] = (imp_keep, topo_keep)
    torch.save(imp_keep.cpu(), f"{out}/keep_imp_{tgt}.pt"); torch.save(topo_keep.cpu(), f"{out}/keep_topo_{tgt}.pt")
    res["runs"].append({"target": tgt, "imp_both": b1, "topo_both": b2,
                        "imp_topo_rate": r_imp, "topo_topo_rate": r_topo, "restored": int(restored.sum()),
                        "floor_sec": _tfsec, "ref_sec": REF_SEC})
    print(f"[prune {tgt:6d}] PSNR imp {b1:.2f} vs floor {b2:.2f} | topo-rate imp {r_imp:.2f} vs floor {r_topo:.2f} | restored {restored.sum().item()}")

json.dump(res, open(f"{out}/morse_prune_{a.scene}.json", "w"), indent=1)
print(f"[saved] {out}/morse_prune_{a.scene}.json")

# ---------------- qualitative: side-by-side at the smallest budget (held cam, mid clip) ----------------
tgt = targets[-1]; imp_keep, topo_keep = keep_sets[tgt]
ci = held_c[len(held_c) // 2]; ti = held_t[len(held_t) // 2]; tv = ti / (T - 1)
with torch.no_grad():
    r_full = render(full_idx, ci, tv); r_imp = render(imp_keep, ci, tv); r_topo = render(topo_keep, ci, tv)
    gt, m = GT[(ci, ti)]; m3 = m[..., None].to(dev)
    tiles = [gt.to(dev) * m3, r_full * m3, r_imp * m3, r_topo * m3]
imrow = (torch.cat(tiles, 1).cpu().numpy() * 255).astype(np.uint8)
Image.fromarray(imrow).save(f"{out}/qual_{a.scene}_cam{ci}_t{ti}_{tgt}.png")
print(f"[saved] {out}/qual_{a.scene}_cam{ci}_t{ti}_{tgt}.png  (GT | full | importance | +topo)")
