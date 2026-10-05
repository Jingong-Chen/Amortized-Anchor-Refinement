"""Streaming package comparison figure: the fitted field before and after our pruning + coding,
rendered from free orbit viewpoints with the paper's rasterizer (identical treatment for both rows,
full SH, foreground composited on white). Two rows x three viewpoints.
Usage: fig_package_cmp.py --scene walk --budget 40000"""
import os, argparse, sys, glob
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from gsplat import rasterization
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/aar")
from teacher_retime import load_cams

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="walk"); ap.add_argument("--budget", type=int, default=40000)
ap.add_argument("--ckpt", default="")
ap.add_argument("--keep", default="")
ap.add_argument("--pkg_mb", default="2.8")
ap.add_argument("--shifty", type=float, default=0.0)   # move the principal point so the subject sits in frame
ap.add_argument("--zoomout", type=float, default=1.0)   # widen the field of view so the whole subject stays in frame
ap.add_argument("--res", type=int, default=512); ap.add_argument("--n_frames", type=int, default=33)
ap.add_argument("--t", type=float, default=0.45)
ap.add_argument("--yaws", default="-35,0,35")            # degrees around the rig centre, relative to a held camera
ap.add_argument("--out", default=f"{ROOT}/paper_vr/picture/webxr_decode.png")
a = ap.parse_args(); dev = "cuda"

ck = torch.load(a.ckpt or sorted(glob.glob(f"{ROOT}/experiments/outputs/nine_models/{a.scene}/model_*_96000.pt"))[0], map_location=dev)
means, scales, quats = ck["means"].to(dev), ck["scales"].to(dev), ck["quats"].to(dev)
opac, sh0, shN = ck["opac"].to(dev), ck["sh0"].to(dev), ck["shN"].to(dev)
mu_t, log_tl, log_tr = ck["mu_t"].to(dev), ck["log_tl"].to(dev), ck["log_tr"].to(dev)
v1, v2, v3 = ck["v1"].to(dev), ck["v2"].to(dev), ck["v3"].to(dev)
N = means.shape[0]
keep = torch.load(a.keep or f"{ROOT}/experiments/outputs/morse_{a.scene}_opacity/keep_topo_{a.budget}.pt", map_location=dev)

SD = f"{ROOT}/data/sc_mv_all/{a.scene}"
frames = sorted([d for d in os.listdir(SD) if d.isdigit()])[:a.n_frames]; T = len(frames)
im0 = Image.open(f"{SD}/{frames[0]}/00.png"); W0, H0 = im0.size; sc = a.res / max(W0, H0)
Wr, Hr = int(round(W0 * sc)), int(round(H0 * sc))
vms, Ks = load_cams(f"{SD}/{frames[0]}/cameras.json", Wr, Hr, dev); ncam = len(vms)
if a.zoomout != 1.0 or a.shifty:
    Ks = Ks.clone(); Ks[:, 0, 0] /= a.zoomout; Ks[:, 1, 1] /= a.zoomout
    Ks[:, 1, 2] += a.shifty * Hr
ci = ncam // 2
GAMMA = 0.005

def orbit(view, deg):
    """Rotate a world-to-camera matrix about the world up axis through the scene centre."""
    c2w = torch.linalg.inv(view)
    ctr = means.median(0).values
    th = float(np.deg2rad(deg))
    R = torch.tensor([[np.cos(th), 0.0, np.sin(th)], [0.0, 1.0, 0.0], [-np.sin(th), 0.0, np.cos(th)]],
                     dtype=torch.float32, device=dev)
    c2w = c2w.clone()
    c2w[:3, 3] = R @ (c2w[:3, 3] - ctr) + ctr
    c2w[:3, :3] = R @ c2w[:3, :3]
    return torch.linalg.inv(c2w)

def render(idx, view, tv):
    tl = torch.exp(log_tl[idx]); tr = torch.exp(log_tr[idx]); mt = mu_t[idx]
    le = mt - tl; re = mt + tr
    ta = torch.where(le < 0.02, torch.ones_like(le), torch.sigmoid((tv - le) / GAMMA)) * \
         torch.where(re > 0.98, torch.ones_like(re), torch.sigmoid((re - tv) / GAMMA))
    s = tv - 0.5
    pos = means[idx] + v1[idx] * s + v2[idx] * s * s + v3[idx] * s ** 3
    al = (torch.sigmoid(opac[idx]) * ta).clamp(1e-4, 1 - 1e-4)
    col, alpha, _ = rasterization(pos, F.normalize(quats[idx], dim=-1), torch.exp(scales[idx]).clamp(max=0.06),
                                  al, torch.cat([sh0[idx], shN[idx]], 1), view[None], Ks[ci][None],
                                  Wr, Hr, sh_degree=3, render_mode="RGB+ED")
    rgb = col[0, ..., :3].clamp(0, 1)
    acc = alpha[0].clamp(0, 1)
    return (rgb * acc + (1 - acc)).clamp(0, 1)                   # composite on white

yaws = [float(x) for x in a.yaws.split(",")]
full_idx = torch.arange(N, device=dev)
rows = []
with torch.no_grad():
    for idx in (full_idx, keep):
        rows.append([render(idx, orbit(vms[ci], y), a.t) for y in yaws])

def to_img(t): return Image.fromarray((t.cpu().numpy() * 255).astype(np.uint8))
imgs = [[to_img(t) for t in row] for row in rows]
W, H = imgs[0][0].size
try:
    FB = ImageFont.truetype("/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf", int(W * 0.058))
    FR = ImageFont.truetype("/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf", int(W * 0.050))
except Exception:
    FB = FR = ImageFont.load_default()
LAB = int(W * 0.24); GAP = 6
labels = [("Uncompressed field", f"{N/1000:.0f}k Gaussians, {(N*71*4)/2**20:.1f} MB"),
          ("Ours, streamed package", f"{keep.numel()/1000:.0f}k Gaussians, {a.pkg_mb} MB")]
canvas = Image.new("RGB", (LAB + 3 * W + 2 * GAP, 2 * H + GAP), (255, 255, 255))
for r, (name, sub) in enumerate(labels):
    y = r * (H + GAP)
    for k in range(3):
        canvas.paste(imgs[r][k], (LAB + k * (W + GAP), y))
    strip = Image.new("RGB", (max(H, LAB * 4), LAB), (255, 255, 255))
    sd = ImageDraw.Draw(strip)
    bb = sd.textbbox((0, 0), name, font=FB)
    sd.text(((strip.width - (bb[2] - bb[0])) / 2, LAB * 0.10), name, font=FB, fill=(20, 20, 20))
    bb2 = sd.textbbox((0, 0), sub, font=FR)
    sd.text(((strip.width - (bb2[2] - bb2[0])) / 2, LAB * 0.10 + (bb[3] - bb[1]) + LAB * 0.24), sub, font=FR, fill=(90, 90, 90))
    rot = strip.rotate(90, expand=True)
    canvas.paste(rot, (0, y + (H - rot.height) // 2))
canvas.save(a.out)
print(f"saved {a.out} {canvas.size}  full N={N}  ours N={keep.numel()}")
