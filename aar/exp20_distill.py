#!/usr/bin/env python
"""
Exp-19: CROSS-SCENE training for COMPLETION (the survey's prescribed path).

Per-scene overfit has ZERO completion ability. Completion = an amortized prior. We can't
learn it from scratch on ~9 scenes, but we LEVERAGE VGGT-Omega's foundation prior and train
a SHARED grounded decoder ACROSS scenes, so it learns the mapping (VGGT features -> 4D
gaussians that render correctly from novel views). Test on a HELD-OUT scene with MONOCULAR
input -> does the cross-scene prior generalize (= weak completion) vs per-scene overfit?

Borrows Flash3D-style STACKED DEPTH LAYERS (K gaussians per ray: visible surface + behind)
for occlusion/back completion. Backbone frozen; features cached per scene.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp3_decoder_skeleton import temporal_opacity, mlp            # noqa: E402
from gsplat.rendering import rasterization                         # noqa: E402
from teacher_retime import TeacherRetimeGS                         # noqa: E402  RetimeGS teacher for dense distillation

DIMS = {"depth": 1, "color": 3, "opacity": 1, "scale": 3, "quat": 4,
        "v1": 3, "v2": 3, "v3": 3, "tau_l": 1, "tau_r": 1}
PARAMS = list(DIMS)
PDIM = sum(DIMS.values())


class GroundedDecoderK(nn.Module):
    """per grid pixel -> K stacked layers (Flash3D-style) of grounded 4D gaussians."""
    def __init__(self, feat_dim, K=2, hidden=256):
        super().__init__()
        self.K = K
        self.head = mlp(feat_dim, hidden, K * PDIM)

    def forward(self, feat):                                        # [N, feat_dim]
        x = self.head(feat).reshape(feat.shape[0], self.K, PDIM)    # [N, K, PDIM]
        k, r = 0, {}
        for p in PARAMS:
            r[p] = x[..., k:k + DIMS[p]]; k += DIMS[p]
        return r                                                   # each [N, K, d]


DIMS_REG = {"xyz": 3, "color": 3, "opacity": 1, "scale": 3, "quat": 4,       # xyz (free) instead of depth (grounded)
            "v1": 3, "v2": 3, "v3": 3, "tau_l": 1, "tau_r": 1}
PARAMS_REG = list(DIMS_REG); PDIM_REG = sum(DIMS_REG.values())


class RegisterDecoderJ(nn.Module):
    """register token -> J FREE (non-grounded) gaussians. Covers the UNSEEN back side that
    ray-grounded gaussians (in cam0's frustum) cannot reach. xyz placed relative to scene center."""
    def __init__(self, feat_dim, J=256, hidden=256):
        super().__init__()
        self.J = J
        self.head = mlp(feat_dim, hidden, J * PDIM_REG)
    def forward(self, tok):                                         # [nreg, feat_dim]
        x = self.head(tok).reshape(tok.shape[0] * self.J, PDIM_REG)
        k, r = 0, {}
        for p in PARAMS_REG:
            r[p] = x[..., k:k + DIMS_REG[p]]; k += DIMS_REG[p]
        return r                                                   # each [nreg*J, d]


class TransformerDecoderG(nn.Module):
    """DETR-style: learned gaussian queries cross-attend to VGGT features -> FREE 4D gaussians.
    Cross-feature attention (vs per-feature MLP) lifts expressiveness; free xyz reaches the back side."""
    def __init__(self, feat_dim, M=4096, D=256, layers=3, heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(M, D) * 0.02)
        self.feat_proj = nn.Linear(feat_dim, D)
        self.time_emb = nn.Sequential(nn.Linear(1, D), nn.SiLU(), nn.Linear(D, D))
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(D, heads, D * 2, batch_first=True, dropout=0.0) for _ in range(layers)])
        self.head = mlp(D, D, PDIM_REG)
    def forward(self, feat, tval):                                 # feat [Nfeat, feat_dim], tval float
        mem = self.feat_proj(feat)[None]                          # [1, Nfeat, D]
        te = self.time_emb(torch.tensor([[tval]], device=feat.device, dtype=torch.float32))  # [1, D]
        q = self.query[None] + te[:, None]                        # [1, M, D]
        for L in self.layers: q = L(q, mem)                       # cross-attn queries -> scene features
        x = self.head(q[0])                                       # [M, PDIM_REG]
        k, r = 0, {}
        for p in PARAMS_REG: r[p] = x[..., k:k + DIMS_REG[p]]; k += DIMS_REG[p]
        return r                                                  # each [M, d]


def load_cameras(cam_json, scale):
    rigs = json.load(open(cam_json))["rigs"]
    vms, Ks, full = [], [], []
    for r in rigs:
        c = r["cameras"][0]
        R = np.array(c["R"], np.float32); cen = np.array(c["c"], np.float32)
        E = np.eye(4, dtype=np.float32); E[:3, :3] = R; E[:3, 3] = -R @ cen
        K = np.array([[c["fx"] * scale, 0, c["cx"] * scale], [0, c["fy"] * scale, c["cy"] * scale], [0, 0, 1]], np.float32)
        vms.append(E); Ks.append(K); full.append((R, cen, c["fx"], c["fy"], c["cx"], c["cy"]))
    return torch.tensor(np.stack(vms)), torch.tensor(np.stack(Ks)), full


def cam_rays(camfull, Gh, Gw, dev):
    R, cen, fx, fy, cx, cy = camfull
    gy, gx = torch.meshgrid(torch.arange(Gh), torch.arange(Gw), indexing="ij")
    u = (gx.float() + 0.5) / Gw * (2 * cx); v = (gy.float() + 0.5) / Gh * (2 * cy)
    dcam = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], -1).reshape(-1, 3)
    # R is world->cam (load_cameras builds E=[R|-R@cen]); cam->world direction = R.T @ d_cam = d_cam @ R.
    # (Previously d_cam @ R.T -> applied R, double-rotating the grounding cone; cam0 R≈I masked it as a ~20° tilt
    #  so cam0 self-render was ~9dB and ALL novel views fell outside frame -> empty completions.)
    dworld = F.normalize(dcam @ torch.tensor(R), dim=-1)
    return torch.tensor(cen, device=dev), dworld.to(dev)


def psnr(m):
    return float(-10 * np.log10(m + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)                       # dir with <scene>/<frame>/...
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--held", default="walk")                      # held-out test scene
    ap.add_argument("--render_w", type=int, default=192)
    ap.add_argument("--input_res", type=int, default=256)
    ap.add_argument("--gh", type=int, default=64)
    ap.add_argument("--gw", type=int, default=88)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fgw", type=float, default=20.0)
    ap.add_argument("--max_times", type=int, default=17)
    ap.add_argument("--teacher", type=int, default=0)              # 1 = distill from per-scene RetimeGS teachers (dense 32-view x continuous-time supervision)
    ap.add_argument("--teacher_dir", default="experiments/outputs")  # where retime_<scene>/model_<scene>.pt live
    ap.add_argument("--use_scenes", default="")                    # comma list to restrict scenes (e.g. only those with teachers)
    ap.add_argument("--eval_only", type=int, default=0)            # 1 = load saved student.pt, skip training, just re-render diagnostics
    ap.add_argument("--fg_only", type=int, default=0)              # 1 = supervise FOREGROUND only (mask out green-screen bg) -> kill the collapse-to-background shortcut
    ap.add_argument("--overfit", type=int, default=0)              # 1 = train on the held scene itself (single-scene overfit; architecture sanity check)
    ap.add_argument("--self_only", type=int, default=0)            # 1 = supervise ONLY the input cam0 (can the 2.5D student fit its own input view? isolates monocular-vs-multiview conflict)
    ap.add_argument("--input_cams", default="0")                   # views fed to student: "0"=monocular (2.5D), "0,8,16,24"=multi-view (true 3D)
    ap.add_argument("--reg_j", type=int, default=0)                # register-conditioned free back-side gaussians per token (0=off; e.g. 256)
    ap.add_argument("--arch", default="mlp", choices=["mlp", "transformer"])   # mlp=grounded per-feature; transformer=DETR-style free gaussian queries
    ap.add_argument("--M", type=int, default=4096)                 # transformer: number of gaussian queries per frame
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    try:
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(True); torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    scenes = sorted([d for d in os.listdir(a.root) if os.path.isdir(f"{a.root}/{d}")])
    if a.use_scenes: scenes = [s for s in scenes if s in a.use_scenes.split(",")]   # restrict to scenes that have teachers
    train_scenes = [s for s in scenes if s != a.held]
    if a.overfit: train_scenes = [a.held]                          # OVERFIT: train on the held scene itself (architecture sanity check)
    print(f"[info] scenes {scenes}; TRAIN {train_scenes}; HELD {a.held}")
    cam_used = [0, 8, 16, 24, 4, 12, 20, 28]                       # 4 train + 4 held cams
    train_c, held_c = cam_used[:4], cam_used[4:]
    INPUT_CAMS = [int(x) for x in str(a.input_cams).split(",")]    # student input views: "0"=monocular(2.5D), "0,8,16,24"=multi-view(true 3D)
    INPUT_CAM = INPUT_CAMS[0]                                       # primary view (diagnostics / self_only)

    # geometry shared across scenes (same rig)
    fr0 = sorted([d for d in os.listdir(f"{a.root}/{a.held}") if d.isdigit()])[0]
    W0, H0 = Image.open(f"{a.root}/{a.held}/{fr0}/{INPUT_CAM:02d}.png").size
    scale = a.render_w / W0
    Wr, Hr = a.render_w, int(round(H0 * scale))
    vms, Ks, full = load_cameras(f"{a.root}/{a.held}/{fr0}/cameras.json", scale)
    vms, Ks = vms.to(dev), Ks.to(dev)
    rays = {c: cam_rays(full[c], a.gh, a.gw, dev) for c in INPUT_CAMS}   # per-input-cam (origin, dirs)

    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    bb = VGGTOmega().to(dev).eval()
    bb.load_state_dict(torch.load(a.ckpt, map_location="cpu"))

    def scene_data(scene):
        frames = sorted([d for d in os.listdir(f"{a.root}/{scene}") if d.isdigit()])[:a.max_times]
        feats = {}; regs = {}; Ftt = Cd = None
        for c in INPUT_CAMS:                                       # VGGT features per INPUT view -> each view grounds its own gaussians
            names = [f"{a.root}/{scene}/{f}/{c:02d}.png" for f in frames]
            imgs = load_and_preprocess_images(names, image_resolution=a.input_res).to(dev)
            with torch.no_grad():
                agg, ps = bb.aggregator(imgs[None])
            fin = agg[-1]; Cd = fin.shape[-1]; Ftt = fin.shape[1]
            patch = fin[0, :, ps:, :].float()
            _, _, _, Hin, Win = imgs[None].shape
            ph, pw = Hin // 16, Win // 16
            patch = patch.reshape(Ftt, ph, pw, Cd).permute(0, 3, 1, 2)
            gy, gx = torch.meshgrid(torch.linspace(-1, 1, a.gh, device=dev), torch.linspace(-1, 1, a.gw, device=dev), indexing="ij")
            grid = torch.stack([gx, gy], -1)[None].expand(Ftt, -1, -1, -1)
            feats[c] = F.grid_sample(patch, grid, align_corners=True).permute(0, 2, 3, 1).reshape(Ftt, a.gh * a.gw, Cd).detach()
            regs[c] = fin[0, :, 1:ps, :].float().detach()             # [Ftt, nreg, Cd] register tokens (drop camera token) -> back-side gaussians
        # GT rgb+mask at render res for used cams
        rgb = torch.zeros(len(cam_used), len(frames), Hr, Wr, 3, device=dev)
        msk = torch.zeros(len(cam_used), len(frames), Hr, Wr, device=dev)
        for ci, c in enumerate(cam_used):
            for ti, f in enumerate(frames):
                im = Image.open(f"{a.root}/{scene}/{f}/{c:02d}.png").convert("RGB").resize((Wr, Hr), Image.BILINEAR)
                rgb[ci, ti] = torch.tensor(np.asarray(im, np.float32) / 255.0, device=dev)
                mp = f"{a.root}/{scene}/{f}/masks/{c:02d}.png"
                if os.path.exists(mp):
                    msk[ci, ti] = torch.tensor((np.asarray(Image.open(mp).convert("L").resize((Wr, Hr), Image.NEAREST)) > 127).astype(np.float32), device=dev)
        return {"feats": feats, "regs": regs, "rgb": rgb, "msk": msk, "F": Ftt, "Cd": Cd}

    print("[info] caching VGGT features + GT for all scenes ...")
    data = {s: scene_data(s) for s in scenes}
    Cd = data[a.held]["Cd"]
    del bb
    torch.cuda.empty_cache()
    times = [torch.tensor(t / max(a.max_times - 1, 1), device=dev) for t in range(a.max_times)]

    dec = GroundedDecoderK(feat_dim=Cd, K=a.K).to(dev)
    reg_dec = RegisterDecoderJ(feat_dim=Cd, J=a.reg_j).to(dev) if a.reg_j > 0 else None
    tdec = TransformerDecoderG(feat_dim=Cd, M=a.M).to(dev) if a.arch == "transformer" else None
    if a.eval_only:
        if tdec is not None: tdec.load_state_dict(torch.load(f"{a.out}/tdec.pt", map_location=dev))
        else: dec.load_state_dict(torch.load(f"{a.out}/student.pt", map_location=dev))
        if reg_dec is not None and os.path.exists(f"{a.out}/reg.pt"): reg_dec.load_state_dict(torch.load(f"{a.out}/reg.pt", map_location=dev))
        print("[info] eval_only: loaded")
    params_all = (list(tdec.parameters()) if tdec is not None else list(dec.parameters())) + (list(reg_dec.parameters()) if reg_dec is not None else [])
    opt = torch.optim.Adam(params_all, lr=a.lr)

    def build_g(feats, regs, F_):                                 # grounded gaussians (on cam rays) + register-conditioned free back-side gaussians
        outs = {k: [] for k in ["xyz", "color", "opacity", "scale", "quat", "v1", "v2", "v3", "mu_t", "tau_l", "tau_r"]}
        for c in INPUT_CAMS:
            ro, rd = rays[c]
            for fi in range(F_):
                r = dec(feats[c][fi])                             # [N,K,d]
                base = F.softplus(r["depth"][..., 0] + 2.0) + 0.3
                depth = base.clone()                              # stacked layers: layer k depth = cumulative
                for kk in range(1, a.K):
                    depth[:, kk] = depth[:, kk - 1] + F.softplus(r["depth"][:, kk, 0]) * 0.2 + 0.02
                N, K = depth.shape
                xyz = ro[None, None] + depth[..., None] * rd[:, None, :]   # gaussians on THIS input cam's rays
                outs["xyz"].append(xyz.reshape(N * K, 3))
                outs["color"].append(torch.sigmoid(r["color"]).reshape(N * K, 3))
                outs["opacity"].append(torch.sigmoid(r["opacity"]).reshape(N * K, 1))
                outs["scale"].append((F.softplus(r["scale"]) * 0.02 + 1e-3).reshape(N * K, 3))
                outs["quat"].append(F.normalize(r["quat"], dim=-1).reshape(N * K, 4))
                for kk in ("v1", "v2", "v3"):
                    outs[kk].append((torch.tanh(r[kk]) * 0.5).reshape(N * K, 3))
                outs["mu_t"].append(times[fi].expand(N * K, 1))
                outs["tau_l"].append((F.softplus(r["tau_l"]) * 0.08 + 1e-2).reshape(N * K, 1))
                outs["tau_r"].append((F.softplus(r["tau_r"]) * 0.08 + 1e-2).reshape(N * K, 1))
        out = {k: torch.cat(v, 0) for k, v in outs.items()}       # [Ntot, d]
        if reg_dec is not None:                                    # add FREE back-side gaussians from register tokens (Lyra-completion analog)
            gc = out["xyz"].mean(0).detach(); gr = (out["xyz"].std(0).mean() * 2 + 0.3).detach()
            for fi in range(F_):
                rr = reg_dec(regs[INPUT_CAM][fi])                 # [nreg*J, d]
                n = rr["xyz"].shape[0]
                out["xyz"] = torch.cat([out["xyz"], gc + torch.tanh(rr["xyz"]) * gr], 0)   # free placement around scene center
                out["color"] = torch.cat([out["color"], torch.sigmoid(rr["color"])], 0)
                out["opacity"] = torch.cat([out["opacity"], torch.sigmoid(rr["opacity"])], 0)
                out["scale"] = torch.cat([out["scale"], F.softplus(rr["scale"]) * 0.02 + 1e-3], 0)
                out["quat"] = torch.cat([out["quat"], F.normalize(rr["quat"], dim=-1)], 0)
                for kk in ("v1", "v2", "v3"): out[kk] = torch.cat([out[kk], torch.tanh(rr[kk]) * 0.5], 0)
                out["mu_t"] = torch.cat([out["mu_t"], times[fi].expand(n, 1)], 0)
                out["tau_l"] = torch.cat([out["tau_l"], F.softplus(rr["tau_l"]) * 0.08 + 1e-2], 0)
                out["tau_r"] = torch.cat([out["tau_r"], F.softplus(rr["tau_r"]) * 0.08 + 1e-2], 0)
        return {k: v[None] for k, v in out.items()}

    # transformer path: free gaussian queries placed around a cam0-derived scene center (deploy-friendly)
    E0 = vms[INPUT_CAM].to(dev); R0 = E0[:3, :3]; cen0 = -R0.t() @ E0[:3, 3]
    scene_center = cen0 + 2.5 * R0[2, :]; scene_radius = 1.5
    def build_g_t(feats, F_):
        outs = {k: [] for k in ["xyz", "color", "opacity", "scale", "quat", "v1", "v2", "v3", "mu_t", "tau_l", "tau_r"]}
        for fi in range(F_):
            r = tdec(feats[INPUT_CAM][fi], float(times[fi]))       # DETR-style free gaussians for this frame
            outs["xyz"].append(scene_center + torch.tanh(r["xyz"]) * scene_radius)
            outs["color"].append(torch.sigmoid(r["color"]))
            outs["opacity"].append(torch.sigmoid(r["opacity"]))
            outs["scale"].append(F.softplus(r["scale"]) * 0.02 + 1e-3)
            outs["quat"].append(F.normalize(r["quat"], dim=-1))
            for kk in ("v1", "v2", "v3"): outs[kk].append(torch.tanh(r[kk]) * 0.5)
            outs["mu_t"].append(times[fi].expand(r["xyz"].shape[0], 1))
            outs["tau_l"].append(F.softplus(r["tau_l"]) * 0.08 + 1e-2)
            outs["tau_r"].append(F.softplus(r["tau_r"]) * 0.08 + 1e-2)
        return {k: torch.cat(v, 0)[None] for k, v in outs.items()}
    def build(d):
        return build_g_t(d["feats"], d["F"]) if a.arch == "transformer" else build_g(d["feats"], d["regs"], d["F"])

    def render_ct(g, c, t):                                         # t = continuous time VALUE in [0,1] (not an index)
        s = (t - g["mu_t"])
        P = (g["xyz"] + g["v1"] * s + g["v2"] * s * s + g["v3"] * s * s * s)[0]
        al = temporal_opacity(g, t)[0].clamp(1e-4, 1 - 1e-4)
        out, _, _ = rasterization(P, g["quat"][0], g["scale"][0], al, g["color"][0], vms[c][None], Ks[c][None], Wr, Hr, render_mode="RGB")
        return out[0].clamp(0, 1)

    def fg_psnr(d, g, cams, ts):
        with torch.no_grad():
            es = []
            for ci in cams:
                for t in ts:
                    m3 = d["msk"][cam_used.index(ci), t][..., None]
                    if m3.sum() < 10:
                        continue
                    es.append(((m3 * (render_ct(g, ci, times[t]) - d["rgb"][cam_used.index(ci), t]) ** 2).sum() / (m3.sum() * 3 + 1e-6)).item())
        return psnr(float(np.mean(es))) if es else float("nan")

    Tn = a.max_times
    train_t = [t for t in range(Tn) if t % 2 == 0]; held_t = [t for t in range(Tn) if t % 2 == 1]

    # --- load RetimeGS teachers (dense distillation targets). walk has a held-out-protocol teacher under retime_walk_final/ ---
    teachers = {}
    if a.teacher:
        tpaths = {"walk": f"{a.teacher_dir}/repro_4d_clean/model_walk.pt"}
        for sc in scenes:
            for cand in [tpaths.get(sc), f"{a.teacher_dir}/base_{sc}/model_{sc}.pt", f"{a.teacher_dir}/retime_{sc}/model_{sc}.pt"]:
                if cand and os.path.exists(cand): teachers[sc] = TeacherRetimeGS(cand, dev); break
        print(f"[info] DISTILL on: {[s for s in train_scenes if s in teachers]} | dense = 32 views x continuous time | held={a.held}")
    ALL_C = list(range(len(vms)))                                   # all 32 rig cameras (dense views) vs sparse train_c

    t0 = time.time()
    for s in range(0 if a.eval_only else a.steps):
        scn = train_scenes[np.random.randint(len(train_scenes))]
        d = data[scn]
        g = build(d)
        if a.teacher and scn in teachers:                           # DENSE teacher distillation: any of 32 views x any continuous time
            if a.self_only:                                         # cam0-only: can the 2.5D student fit even its OWN input view? (isolates monocular-vs-multiview conflict)
                c = INPUT_CAM; ci_idx = cam_used.index(c)
                tv = float(np.random.rand()); ti = int(round(tv * (Tn - 1)))
                with torch.no_grad(): target = teachers[scn].render(vms[c], Ks[c], tv, Wr, Hr)
                m = d["msk"][ci_idx, ti][..., None]
                loss = (m * (render_ct(g, c, torch.tensor(tv, device=dev)) - target) ** 2).sum() / (m.sum() * 3 + 1e-6)
            elif a.fg_only:                                         # FOREGROUND-only: mask out the green-screen bg -> no shortcut, student must learn the person
                ci_idx = np.random.randint(len(cam_used)); c = cam_used[ci_idx]
                tv = float(np.random.rand()); ti = int(round(tv * (Tn - 1)))   # nearest captured frame for the FG mask
                with torch.no_grad(): target = teachers[scn].render(vms[c], Ks[c], tv, Wr, Hr)
                m = d["msk"][ci_idx, ti][..., None]
                loss = (m * (render_ct(g, c, torch.tensor(tv, device=dev)) - target) ** 2).sum() / (m.sum() * 3 + 1e-6)
            else:
                c = ALL_C[np.random.randint(len(ALL_C))]; tv = float(np.random.rand())
                with torch.no_grad(): target = teachers[scn].render(vms[c], Ks[c], tv, Wr, Hr)
                loss = ((render_ct(g, c, torch.tensor(tv, device=dev)) - target) ** 2).mean()
        else:                                                       # baseline: sparse GT (4 train cams x even times)
            c = train_c[np.random.randint(len(train_c))]; t = train_t[np.random.randint(len(train_t))]
            ci = cam_used.index(c)
            w = (1.0 + a.fgw * d["msk"][ci, t])[..., None]
            loss = (w * (render_ct(g, c, times[t]) - d["rgb"][ci, t]) ** 2).sum() / (w.sum() * 3)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params_all, 0.5); opt.step()
        if s % 500 == 0 or s == a.steps - 1:
            print(f"[step {s:4d}] {scn} loss-psnr≈{psnr(loss.item()):.2f}")

    # eval on HELD scene (monocular input -> completion of novel views)
    dh = data[a.held]
    gh = build(dh)
    res = {
        "HELD train-cam train-time": fg_psnr(dh, gh, train_c, train_t),
        "HELD novel-view (heldcam, traintime)": fg_psnr(dh, gh, held_c, train_t),
        "HELD novel-view+time (heldcam, heldtime)": fg_psnr(dh, gh, held_c, held_t),
    }
    print(f"[done] {a.steps} steps {time.time()-t0:.0f}s | held scene = {a.held} (NEVER trained on)")
    for k, v in res.items():
        print(f"[RESULT] FG PSNR  {k:42s}: {v:.2f} dB")

    # per-scene optimization CEILING on the held scene = what a feed-forward student is trying to amortize
    if a.teacher and a.held in teachers:
        tch = teachers[a.held]; es = []
        for ci in held_c:
            for t in held_t:
                m3 = dh["msk"][cam_used.index(ci), t][..., None]
                if m3.sum() < 10: continue
                with torch.no_grad(): tc = tch.render(vms[ci], Ks[ci], times[t], Wr, Hr)
                es.append(((m3 * (tc - dh["rgb"][cam_used.index(ci), t]) ** 2).sum() / (m3.sum() * 3 + 1e-6)).item())
        ceil = psnr(float(np.mean(es))) if es else float("nan")
        print(f"[RESULT] FG PSNR  {'CEILING per-scene RetimeGS teacher (held)':42s}: {ceil:.2f} dB  <- student amortizes THIS")
        res["CEILING_per_scene_teacher"] = ceil

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # INPUT view (fed to student) vs NOVEL view. 4 cols: GT | student raw | student cleaned | teacher.
    # "cleaned" = prune stray low-opacity multi-view-fusion 'fog' + composite foreground over teacher bg (display only).
    inp_t = train_t[len(train_t) // 2]; nov_c = held_c[0]; nov_t = held_t[len(held_t) // 2]
    views = [(f"INPUT cam{INPUT_CAM} t{inp_t}", INPUT_CAM, inp_t), (f"NOVEL cam{nov_c} t{nov_t}", nov_c, nov_t)]
    has_t = a.teacher and a.held in teachers
    op = gh["opacity"][0, :, 0]; keep = op > float(os.environ.get("VIZ_OPA", 0.15))          # drop the fog
    ghp = {k: (v[:, keep] if (v.dim() >= 2 and v.shape[1] == keep.shape[0]) else v) for k, v in gh.items()}
    NCOL = int(os.environ.get("VIZ_COLS", 4))                                                 # 4 = GT|raw|cleaned|teacher; 3 = GT|student|teacher
    fig, ax = plt.subplots(2, NCOL, figsize=(4 * NCOL, 6.8))
    for ri, (lbl, c, t) in enumerate(views):
        ci2 = cam_used.index(c)
        with torch.no_grad():
            rr = render_ct(gh, c, times[t]).cpu().clamp(0, 1)
            rrp = render_ct(ghp, c, times[t]).cpu().clamp(0, 1)
            tcv = (teachers[a.held].render(vms[c], Ks[c], times[t], Wr, Hr).cpu().clamp(0, 1)) if has_t else rr
        m = dh["msk"][ci2, t][..., None].cpu()
        comp = m * rrp + (1 - m) * tcv                                                        # fg over teacher bg
        if NCOL == 3:
            panels = [(dh["rgb"][ci2, t].cpu().clamp(0, 1), f"{lbl}  GT"), (comp, "student"), (tcv, "teacher")]
        else:
            panels = [(dh["rgb"][ci2, t].cpu().clamp(0, 1), f"{lbl}  GT"), (rr, "student (raw)"),
                      (comp, "student (fg, cleaned)"), (tcv, "teacher (per-scene)")]
        for j, (im, ti) in enumerate(panels):
            ax[ri, j].imshow(im); ax[ri, j].set_title(ti, fontsize=15); ax[ri, j].axis("off")
    plt.tight_layout(); fig.savefig(f"{a.out}/exp20_completion.png", dpi=120); fig.savefig(f"{a.out}/exp20_completion.pdf")
    if not a.eval_only:
        if tdec is not None: torch.save(tdec.state_dict(), f"{a.out}/tdec.pt")
        else: torch.save(dec.state_dict(), f"{a.out}/student.pt")  # save student for offline re-render
        if reg_dec is not None: torch.save(reg_dec.state_dict(), f"{a.out}/reg.pt")
    json.dump(res, open(f"{a.out}/results.json", "w"), indent=2)
    print("[done] ->", a.out)


if __name__ == "__main__":
    main()
