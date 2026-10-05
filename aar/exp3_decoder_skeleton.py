#!/usr/bin/env python
"""
Exp-3: register -> 4D-Gaussian decoder SKELETON (untrained) + minimal differentiable render.

Goal: validate the END-TO-END pipeline shapes and GRADIENT FLOW for the proposed
feed-forward 4DGS decoder — NOT quality. The VGGT-Omega backbone is frozen (no_grad);
a small trainable Decoder4D maps `registers` -> a set of continuous-time 4D Gaussians
(RetimeGS-style: regularized short-tailed temporal opacity + per-primitive trajectory);
a pure-torch differentiable splat renders at a query time; an MSE loss backprops to the
decoder. This is the "not just a head" module, in minimal form.

Known simplifications (skeleton): primitives decoded per register token (init source),
positions predicted directly (no depth anchoring yet), linear trajectory, no occlusion
ordering in the splat. These are refinements for later experiments.
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DIMS = {"mu": 3, "vel": 3, "mu_t": 1, "tau_l": 1, "tau_r": 1, "quat": 4, "scale": 3, "opacity": 1, "color": 3}
PARAMS = list(DIMS)
PDIM = sum(DIMS.values())  # 20
GAMMA = 0.02               # temporal-opacity sharpness (short tails)


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))


class Decoder4D(nn.Module):
    """Each register token -> n_per continuous-time 4D Gaussians (RetimeGS-style params)."""

    def __init__(self, token_dim=2048, n_per=8, hidden=512):
        super().__init__()
        self.n_per = n_per
        self.head = mlp(token_dim, hidden, n_per * PDIM)

    def forward(self, registers):           # [B, F, R, C]
        B, Fr, R, C = registers.shape
        x = self.head(registers).reshape(B, Fr * R * self.n_per, PDIM)
        k, r = 0, {}
        for p in PARAMS:
            r[p] = x[..., k:k + DIMS[p]]; k += DIMS[p]
        return {
            "mu": r["mu"],
            "vel": r["vel"],
            "mu_t": torch.sigmoid(r["mu_t"]),                 # normalized time center [0,1]
            "tau_l": F.softplus(r["tau_l"]) + 1e-2,           # short-tailed half-widths
            "tau_r": F.softplus(r["tau_r"]) + 1e-2,
            "quat": F.normalize(r["quat"], dim=-1),
            "scale": F.softplus(r["scale"]) + 1e-3,
            "opacity": torch.sigmoid(r["opacity"]),
            "color": torch.sigmoid(r["color"]),
        }


def temporal_opacity(g, t):
    # RetimeGS-style boundary-aware short-tailed sigmoid product
    mu, tl, tr = g["mu_t"][..., 0], g["tau_l"][..., 0], g["tau_r"][..., 0]
    left = torch.sigmoid((t - (mu - tl)) / GAMMA)
    right = torch.sigmoid(((mu + tr) - t) / GAMMA)
    return g["opacity"][..., 0] * left * right                # [B, N]


def positions_at(g, t):
    return g["mu"] + g["vel"] * (t - g["mu_t"])               # [B, N, 3]


def splat_render(P, color, alpha, extr, intr, H, W, scale, splat_sigma=1.2):
    """P[N,3] world; extr[3,4] world->cam (OpenCV); intr[3,3] at full res; render at `scale`*res."""
    dev = P.device
    R, T = extr[:3, :3], extr[:3, 3]
    cam = P @ R.t() + T
    z = cam[:, 2].clamp(min=1e-3)
    fx, fy, cx, cy = intr[0, 0] * scale, intr[1, 1] * scale, intr[0, 2] * scale, intr[1, 2] * scale
    u = fx * cam[:, 0] / z + cx
    v = fy * cam[:, 1] / z + cy
    ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    px, py = xs.reshape(-1).float(), ys.reshape(-1).float()
    du, dv = u[:, None] - px[None, :], v[:, None] - py[None, :]            # [N, HW]
    w = alpha[:, None] * torch.exp(-0.5 * (du * du + dv * dv) / splat_sigma**2) * (1.0 / z)[:, None]
    img = (w.t() @ color) / (w.sum(0)[:, None] + 1e-6)                     # [HW, 3]
    return img.reshape(H, W, 3)


def scene_stats(depth, extr, intr):
    """centroid + per-axis std of depth-unprojected world points (frame 0), to anchor primitives."""
    d, e, k = depth[0], extr[0], intr[0]
    H, W = d.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    z = d
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    cam = np.stack([x, y, z], -1).reshape(-1, 3)
    R, T = e[:3, :3], e[:3, 3]
    world = (cam - T) @ R
    world = world[np.isfinite(world).all(1)]
    return world.mean(0).astype(np.float32), (world.std(0) + 1e-6).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--n_per", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

    names = sorted(sum([glob.glob(a.image_dir + e) for e in ("/*.png", "/*.jpg")], []))[:a.frames]
    print(f"[info] {len(names)} frames, res={a.res}, dev={dev}")

    # ---- frozen backbone ----
    bb = VGGTOmega().to(dev).eval()
    bb.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    imgs = load_and_preprocess_images(names, image_resolution=a.res).to(dev)
    with torch.no_grad():
        pred = bb(imgs)
    registers = pred["camera_and_register_tokens"][:, :, 1:].float().clone()   # [1,F,16,2048]
    extr, intr = encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])
    extr, intr = extr.float().clone(), intr.float().clone()
    images = pred["images"].float().clone()                                    # [1,F,3,H,W]
    depth = pred["depth"].float().clone()                                      # [1,F,H,W,1]
    B, Fr, _, Himg, Wimg = images.shape
    print(f"[info] registers {list(registers.shape)}, images {list(images.shape)}")

    # scene bbox from depth-unprojection -> anchor primitives into recovered geometry
    s_center, s_scale = scene_stats(depth[0, ..., 0].cpu().numpy(),
                                    extr[0].cpu().numpy(), intr[0].cpu().numpy())
    center = torch.tensor(s_center, device=dev)
    scale = torch.tensor(s_scale, device=dev)
    print(f"[info] scene center {np.round(s_center, 2)}, scale {np.round(s_scale, 2)}")

    # ---- trainable decoder ----
    dec = Decoder4D(token_dim=registers.shape[-1], n_per=a.n_per).to(dev)
    g = dec(registers)
    g["mu"] = center + torch.tanh(g["mu"]) * (2.5 * scale)     # positions within the scene
    g["vel"] = torch.tanh(g["vel"]) * (0.1 * scale)            # small initial velocities
    N = g["mu"].shape[1]
    print(f"[info] decoded {N} 4D primitives (param dim {PDIM}); decoder params "
          f"{sum(p.numel() for p in dec.parameters())/1e6:.2f}M")

    # ---- render at an observed frame's time+camera, MSE vs that frame ----
    f_t = Fr // 2
    t = torch.tensor(float(f_t) / max(Fr - 1, 1), device=dev)
    P = positions_at(g, t)[0]
    alpha = temporal_opacity(g, t)[0]
    color = g["color"][0]
    sc = 0.5
    Hr, Wr = int(Himg * sc), int(Wimg * sc)
    render = splat_render(P, color, alpha, extr[0, f_t], intr[0, f_t], Hr, Wr, scale=sc)
    target = F.interpolate(images[:, f_t], size=(Hr, Wr), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)

    loss = F.mse_loss(render, target)
    loss.backward()
    gnorm = sum(p.grad.norm().item() for p in dec.parameters() if p.grad is not None)
    n_with_grad = sum(1 for p in dec.parameters() if p.grad is not None)
    print(f"[result] render {list(render.shape)} | loss {loss.item():.4f} | "
          f"decoder_grad_norm {gnorm:.4f} ({n_with_grad} tensors)")
    print(f"[result] temporal_opacity@t={t.item():.2f}: range "
          f"[{alpha.min().item():.3f},{alpha.max().item():.3f}], active(>0.01) {(alpha>0.01).sum().item()}/{N}")
    assert gnorm > 0, "GRAD FLOW BROKEN: decoder received zero gradient"
    print("[ok] end-to-end gradient flow VALIDATED (registers -> 4D decoder -> render -> loss -> backward)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
    ax[0].imshow(target.detach().cpu().clamp(0, 1)); ax[0].set_title(f"target frame {f_t}", fontsize=9); ax[0].axis("off")
    ax[1].imshow(render.detach().cpu().clamp(0, 1)); ax[1].set_title("untrained 4D-decoder render", fontsize=9); ax[1].axis("off")
    plt.tight_layout()
    out = f"{a.out}/exp3_skeleton_render.png"
    plt.savefig(out, dpi=120)
    print("[done] saved", out)


if __name__ == "__main__":
    main()
