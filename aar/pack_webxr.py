"""Pack a fitted 4D field (optionally a TFS keep subset) into a compact binary for the WebXR
viewer. Per-splat record (float32/uint8, little-endian), matching viewer/main.js:
  pos[3] f32 | v1[3] f32 | v2[3] f32 | v3[3] f32 | scale[3] f32(exp,clamped) |
  tgate[3] f32 (le, re, gamma-slot unused->mu_t) | rgba u8x4 (DC color + opacity) | quat u8x4
= 18*4 + 8 = 80 bytes/splat. DC-only color (standard for mobile splat viewers).
Usage: pack_webxr.py --ckpt <model.pt> [--keep keep.pt] --out viewer/field.bin"""
import argparse, struct, os, sys
import numpy as np, torch
sys.path.insert(0, "/experiments")
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--keep", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--dcdark", type=int, default=0)           # drop dark-DC oversized splats (DC-only client shading)
ap.add_argument("--defloat", type=int, default=1)         # drop spatial outliers and oversized splats before packing
                                                          # (same criterion as the antifloat prune in hybrid_amort)
ap.add_argument("--smax", type=float, default=0.06)        # hard cap on the largest axis when packing for the client
ap.add_argument("--maxaniso", type=float, default=0)       # cap the scale ratio of a splat (0 = off); fitted splats are
                                                          # near-degenerate needles that face the training cameras and
                                                          # render as spikes from free viewpoints
ap.add_argument("--bakesh", type=int, default=1)           # bake the view-averaged radiance into the stored colour;
                                                          # the client shades with one colour per splat, and the DC term
                                                          # alone renders view-dependent splats (hair) almost black
ap.add_argument("--maskcull", type=int, default=0)
ap.add_argument("--maskfrac", type=float, default=0.0)     # keep a splat only if it lands inside the silhouette in at
                                                          # least this fraction of the views that see it (0 = hull test)
ap.add_argument("--dilate", type=int, default=25)          # silhouette slack in pixels (soft boundary splats must survive)        # drop splats that never project inside the observed foreground
ap.add_argument("--scene", default="walk")                # silhouette in ANY training view at ANY reference time:
ap.add_argument("--res", type=int, default=512)           # such splats are unobserved floaters, not part of the subject
ap.add_argument("--n_frames", type=int, default=33)
a = ap.parse_args()
d = torch.load(a.ckpt, map_location="cpu")
sel = torch.load(a.keep, map_location="cpu") if a.keep else torch.arange(d["means"].shape[0])
if a.defloat:
    m = d["means"][sel].float(); smax = torch.exp(d["scales"][sel].float()).max(-1).values
    dist = (m - m.median(0).values).norm(dim=-1)
    ok = (dist < torch.quantile(dist, 0.92) * 1.6) & (smax < 0.05)
    if a.dcdark:                                          # the client shades with the DC term only; splats whose DC
        sh0_ = d["sh0"][sel].float().reshape(-1, 3)       # colour is near-black but that rely on higher SH bands
        c_ = (0.28209479 * sh0_ + 0.5).clamp(0, 1)        # render as a dark cloud, so drop them for both packages
        lum = (c_ * torch.tensor([0.299, 0.587, 0.114])).sum(-1)
        ok = ok & ~((lum < 0.15) & (smax > 0.015))
    print(f"[defloat] {int(sel.numel() - ok.sum())} of {sel.numel()} splats dropped")
    sel = sel[ok]
if a.maskcull:
    ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, f"{ROOT}/aar")
    from PIL import Image
    from teacher_retime import load_cams
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    SD = f"{ROOT}/data/sc_mv_all/{a.scene}"
    frames = sorted([x for x in os.listdir(SD) if x.isdigit()])[:a.n_frames]; T = len(frames)
    im0 = Image.open(f"{SD}/{frames[0]}/00.png"); W0, H0 = im0.size; scl = a.res / max(W0, H0)
    Wr, Hr = int(round(W0 * scl)), int(round(H0 * scl))
    vms, Ks = load_cams(f"{SD}/{frames[0]}/cameras.json", Wr, Hr, dev); ncam = len(vms)
    mu = d["means"][sel].float().to(dev)
    v1_, v2_, v3_ = (d[k][sel].float().to(dev) for k in ("v1", "v2", "v3"))
    seen = torch.zeros(mu.shape[0], dtype=torch.bool, device=dev)   # visual hull: inside EVERY view that sees it
    best = torch.zeros(mu.shape[0], device=dev)                     # best per-time fraction of views that see it inside
    for ti in range(0, T, 4):                              # every 4th frame is enough to cover the trajectory
        tv = ti / (T - 1); sft = tv - 0.5
        p3 = mu + v1_ * sft + v2_ * sft * sft + v3_ * sft ** 3
 
        inside_all = None; nin = torch.zeros(mu.shape[0], device=dev); nsee = torch.zeros(mu.shape[0], device=dev)
        for ci in range(ncam):
            mp = f"{SD}/{frames[ti]}/masks/{ci:02d}.png"
            if not os.path.exists(mp): continue
            mk = torch.tensor((np.asarray(Image.open(mp).convert("L").resize((Wr, Hr), Image.NEAREST)) > 127),
                              device=dev)
            kk = a.dilate | 1
            mk = torch.nn.functional.max_pool2d(mk.float()[None, None], kk, 1, kk // 2)[0, 0] > 0.5   # dilate: silhouette slack
            pc = (vms[ci][:3, :3] @ p3.T + vms[ci][:3, 3:4]).T
            z = pc[:, 2].clamp(min=1e-6)
            u = (Ks[ci][0, 0] * pc[:, 0] / z + Ks[ci][0, 2]).round().long()
            v = (Ks[ci][1, 1] * pc[:, 1] / z + Ks[ci][1, 2]).round().long()
            ok_uv = (pc[:, 2] > 0) & (u >= 0) & (u < Wr) & (v >= 0) & (v < Hr)
            hit = torch.ones_like(ok_uv)                   # outside the frustum does not disprove membership
            idx = torch.nonzero(ok_uv).squeeze(1)
            if idx.numel():
                hit[idx] = mk[v[idx], u[idx]]
            inside_all = hit if inside_all is None else (inside_all & hit)
            if idx.numel():
                nsee[idx] += 1.0; nin[idx] += mk[v[idx], u[idx]].float()
        if inside_all is not None: seen |= inside_all
        best = torch.maximum(best, nin / nsee.clamp(min=1))
    if a.maskfrac > 0:
        seen = best >= a.maskfrac
        print(f"[maskcull] {int((~seen).sum())} of {seen.numel()} splats inside the silhouette in fewer than "
              f"{a.maskfrac:.0%} of the views that see them -> dropped")
    else:
        print(f"[maskcull] {int((~seen).sum())} of {seen.numel()} splats outside the visual hull at every sampled time -> dropped")
    sel = sel[seen.cpu()]

g = lambda k: d[k][sel].float().numpy()
pos = g("means"); v1, v2, v3 = g("v1"), g("v2"), g("v3")
scale = np.exp(g("scales")).clip(max=a.smax)
if a.maxaniso > 0:
    smax = scale.max(-1, keepdims=True)
    scale = np.maximum(scale, smax / a.maxaniso)
    print(f"[aniso] scale ratio capped at {a.maxaniso:g}")
op = 1/(1+np.exp(-g("opac")))
sh0 = g("sh0").reshape(-1, 3)
col = (0.28209479 * sh0 + 0.5).clip(0, 1)
if a.bakesh:
    import torch as _t
    from gsplat import spherical_harmonics
    from teacher_retime import load_cams as _lc
    _dev = "cuda" if _t.cuda.is_available() else "cpu"
    _SD = f"/data/sc_mv_all/{a.scene}"
    _fr = sorted([x for x in os.listdir(_SD) if x.isdigit()])[:a.n_frames]
    _im0 = Image.open(f"{_SD}/{_fr[0]}/00.png"); _W0, _H0 = _im0.size; _sc = a.res / max(_W0, _H0)
    _vms, _Ks = _lc(f"{_SD}/{_fr[0]}/cameras.json", int(round(_W0*_sc)), int(round(_H0*_sc)), _dev)
    _p = d["means"][sel].float().to(_dev)
    _shs = _t.cat([d["sh0"][sel].float(), d["shN"][sel].float()], 1).to(_dev)   # [N, K, 3]
    _acc = _t.zeros_like(_p); _w = _t.zeros(_p.shape[0], 1, device=_dev)
    for _v in _vms:
        _c2w = _t.linalg.inv(_v); _eye = _c2w[:3, 3]
        _dir = _t.nn.functional.normalize(_p - _eye[None], dim=-1)
        _c = (spherical_harmonics(3, _dir, _shs) + 0.5).clamp(0, 1)     # the client's shading, per view
        _acc += _c; _w += 1
    col = (_acc / _w.clamp(min=1)).cpu().numpy()
    print(f"[bakesh] colour baked from {len(_vms)} training views")
quat = g("quats")[:, [1, 2, 3, 0]]           # stored wxyz (gsplat) -> pack as xyzw
quat = quat / (np.linalg.norm(quat, axis=1, keepdims=True) + 1e-9)
mu = g("mu_t"); tl = np.exp(g("log_tl")); tr = np.exp(g("log_tr"))
le = mu - tl; re = mu + tr
# training-time edge behavior: windows touching the clip ends are fully open
le = np.where(le < 0.02, -10.0, le); re = np.where(re > 0.98, 10.0, re)
N = pos.shape[0]
buf = np.zeros((N, 80), np.uint8)
f32 = buf.view(np.float32).reshape(N, 20)
f32[:, 0:3] = pos; f32[:, 3:6] = v1; f32[:, 6:9] = v2; f32[:, 9:12] = v3
f32[:, 12:15] = scale; f32[:, 15] = le; f32[:, 16] = re; f32[:, 17] = mu
u8 = buf[:, 72:]
u8[:, 0:3] = (col * 255).astype(np.uint8); u8[:, 3] = (op * 255).astype(np.uint8)
u8[:, 4:8] = ((quat * 0.5 + 0.5) * 255).astype(np.uint8)
with open(a.out, "wb") as f:
    f.write(struct.pack("<II", 0x34444753, N))            # magic 'SGD4', count
    f.write(buf.tobytes())
print(f"packed {N} splats -> {a.out}  ({(8 + N*80)/2**20:.1f} MB)")
