"""Render the transmitted packages at an evaluation viewpoint next to ground truth.

The deployment figure is rendered from free viewpoints, where a blurrier package can look tidier
than a sharper one. This script instead renders at a held camera and a held timestamp, the
viewpoint the reported metric is defined on, and puts ground truth beside it, so the comparison is
about agreement with the capture rather than about smoothness.
Usage: fig_pkg_vs_gt.py --scene walk --budget 5000 --cam -1 --t -1
"""
import os, sys, glob, math, argparse
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from gsplat import rasterization
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/aar")
from teacher_retime import load_cams

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="walk"); ap.add_argument("--budget", type=int, default=5000)
ap.add_argument("--ckpt", default=f"{ROOT}/experiments/outputs/clean_walk/model_cleaned_96000.pt")
ap.add_argument("--cols", default="")                     # name:keep_path;... ("" = full field only)
ap.add_argument("--cam", type=int, default=-1); ap.add_argument("--t", type=int, default=-1)
ap.add_argument("--res", type=int, default=512); ap.add_argument("--n_frames", type=int, default=33)
ap.add_argument("--zoom", type=float, default=3.0)
ap.add_argument("--out", default=f"{ROOT}/paper_vr/picture/webxr_decode.png")
a = ap.parse_args(); dev = "cuda"

ck = torch.load(a.ckpt, map_location=dev)
means, scales, quats = ck["means"], ck["scales"], ck["quats"]
opac, sh0, shN = ck["opac"], ck["sh0"], ck["shN"]
mu_t, ltl, ltr = ck["mu_t"], ck["log_tl"], ck["log_tr"]
v1, v2, v3 = ck["v1"], ck["v2"], ck["v3"]
N = means.shape[0]

SD = f"{ROOT}/data/sc_mv_all/{a.scene}"
frames = sorted([d for d in os.listdir(SD) if d.isdigit()])[:a.n_frames]; T = len(frames)
im0 = Image.open(f"{SD}/{frames[0]}/00.png"); W0, H0 = im0.size; sc = a.res / max(W0, H0)
Wr, Hr = int(round(W0 * sc)), int(round(H0 * sc))
vms, Ks = load_cams(f"{SD}/{frames[0]}/cameras.json", Wr, Hr, dev); ncam = len(vms)
held_c = list(range(1, ncam, 2)); held_t = list(range(1, T, 2))
ci = held_c[len(held_c) // 2] if a.cam < 0 else a.cam
ti = held_t[len(held_t) // 2] if a.t < 0 else a.t
tv = ti / (T - 1); GAMMA = 0.005

def render(idx):
    tl = torch.exp(ltl[idx]); tr = torch.exp(ltr[idx]); mt = mu_t[idx]
    le = mt - tl; re = mt + tr
    ta = torch.where(le < 0.02, torch.ones_like(le), torch.sigmoid((tv - le) / GAMMA)) * \
         torch.where(re > 0.98, torch.ones_like(re), torch.sigmoid((re - tv) / GAMMA))
    s = tv - 0.5
    pos = means[idx] + v1[idx] * s + v2[idx] * s * s + v3[idx] * s ** 3
    al = (torch.sigmoid(opac[idx]) * ta).clamp(1e-4, 1 - 1e-4)
    col, _, _ = rasterization(pos, F.normalize(quats[idx], dim=-1), torch.exp(scales[idx]).clamp(max=0.06),
                              al, torch.cat([sh0[idx], shN[idx]], 1), vms[ci][None], Ks[ci][None],
                              Wr, Hr, sh_degree=3, render_mode="RGB")
    return col[0].clamp(0, 1)

gt = torch.tensor(np.asarray(Image.open(f"{SD}/{frames[ti]}/{ci:02d}.png").convert("RGB").resize((Wr, Hr))) / 255., dtype=torch.float32, device=dev)
m = torch.tensor((np.asarray(Image.open(f"{SD}/{frames[ti]}/masks/{ci:02d}.png").convert("L").resize((Wr, Hr), Image.NEAREST)) > 127).astype(np.float32), device=dev)[..., None]
m = -F.max_pool2d(-m.permute(2, 0, 1)[None], 3, 1, 1)[0].permute(1, 2, 0)
bg = torch.ones(Hr, Wr, 3, device=dev)

panels = [("GT", gt * m + bg * (1 - m), None)]
with torch.no_grad():
    for spec in [c for c in a.cols.split(";") if c]:
        name, path = spec.split(":")
        idx = torch.load(path, map_location=dev) if path else torch.arange(N, device=dev)
        r = render(idx)
        e = (((r - gt) ** 2 * m).sum() / (m.sum() * 3 + 1e-6)).item()
        panels.append((name, r * m + bg * (1 - m), -10 * math.log10(e + 1e-12)))

# zoom box: where the two pruned packages differ most inside the subject
if len(panels) >= 4:
    d = ((panels[-1][1] - panels[-2][1]).abs().mean(-1, keepdim=True) * m).permute(2, 0, 1)[None]
    CW, CH = int(Wr * 0.34), int(Hr * 0.34)
    scr = F.avg_pool2d(d, (CH, CW), stride=8)
    iy, ix = np.unravel_index(scr[0, 0].argmax().cpu().numpy(), scr[0, 0].shape)
    y0, x0 = min(max(0, int(iy * 8)), Hr - CH), min(max(0, int(ix * 8)), Wr - CW)
else:
    CW, CH, x0, y0 = Wr // 3, Hr // 3, Wr // 3, Hr // 3

def to_img(t): return Image.fromarray((t.cpu().numpy() * 255).astype(np.uint8))
YELLOW = (255, 210, 60)
top, bot = [], []
for name, t, p in panels:
    im = to_img(t); ImageDraw.Draw(im).rectangle([x0, y0, x0 + CW, y0 + CH], outline=YELLOW, width=3)
    top.append(im)
    z = to_img(t).crop((x0, y0, x0 + CW, y0 + CH)).resize((Wr, int(CH * Wr / CW)), Image.LANCZOS)
    ImageDraw.Draw(z).rectangle([1, 1, z.width - 2, z.height - 2], outline=YELLOW, width=4)
    bot.append(z)

W, H1, H2 = Wr, Hr, bot[0].height
try:
    FB = ImageFont.truetype("/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf", int(W * 0.085))
    FR = ImageFont.truetype("/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf", int(W * 0.075))
except Exception:
    FB = FR = ImageFont.load_default()
HN, HP, GAP = int(W * 0.10), int(W * 0.095), 6
canvas = Image.new("RGB", (len(panels) * W + (len(panels) - 1) * GAP, HN + HP + H1 + H2 + GAP), (255, 255, 255))
d = ImageDraw.Draw(canvas)
for k, (name, _, p) in enumerate(panels):
    x = k * (W + GAP)
    bb = d.textbbox((0, 0), name, font=FB)
    d.text((x + (W - (bb[2] - bb[0])) / 2, (HN - (bb[3] - bb[1])) / 2 - bb[1]), name, font=FB, fill=(20, 20, 20))
    if p is not None:
        lab = f"{p:.2f} dB"
        b2 = d.textbbox((0, 0), lab, font=FR)
        d.text((x + (W - (b2[2] - b2[0])) / 2, HN + (HP - (b2[3] - b2[1])) / 2 - b2[1]), lab, font=FR, fill=(80, 80, 80))
    canvas.paste(top[k], (x, HN + HP)); canvas.paste(bot[k], (x, HN + HP + H1 + GAP))
canvas.save(a.out)
print(f"saved {a.out} {canvas.size}  cam{ci} t{ti}  crop=({x0},{y0},{CW}x{CH})")
for name, _, p in panels[1:]:
    print(f"  {name:<14} {p:.2f} dB at this view")
