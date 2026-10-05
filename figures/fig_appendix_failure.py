"""Appendix figure: the three residual error modes, as zoom crops against ground truth at a held
timestamp. Each column is given as scene:label:fx:fy:fw, where fx,fy locate the crop centre inside
the subject bounding box and fw is the crop width as a fraction of that box.
Usage: fig_appendix_failure.py --cols "newundress:Loose fabric:0.5:0.6:0.55;..."
"""
import os, sys, glob, argparse
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from gsplat import rasterization
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/aar")
from teacher_retime import load_cams

ap = argparse.ArgumentParser()
ap.add_argument("--cols", default="newundress:Loose fabric:0.50:0.62:0.60;"
                                  "passdoll:Two-subject contact:0.62:0.42:0.60;"
                                  "stretch:Fast limb motion:0.50:0.45:0.70")
ap.add_argument("--cam", type=int, default=3); ap.add_argument("--ti", type=int, default=15)
ap.add_argument("--res", type=int, default=1024); ap.add_argument("--n_frames", type=int, default=33)
ap.add_argument("--tile", type=int, default=340); ap.add_argument("--crop_ar", type=float, default=1.0)
ap.add_argument("--out", default=f"{ROOT}/paper_vr/picture/appendix_failure.png")
a = ap.parse_args(); dev = "cuda"; GAMMA = 0.005

def do(scene, fx, fy, fw):
    ck = torch.load(sorted(glob.glob(f"{ROOT}/experiments/outputs/nine_models/{scene}/model_*_96000.pt"))[0], map_location=dev)
    P = {k: ck[k].to(dev) for k in ("means", "scales", "quats", "opac", "sh0", "shN", "mu_t", "log_tl", "log_tr", "v1", "v2", "v3")}
    SD = f"{ROOT}/data/sc_mv_all/{scene}"
    frames = sorted(d for d in os.listdir(SD) if d.isdigit())[:a.n_frames]; T = len(frames)
    W0, H0 = Image.open(f"{SD}/{frames[0]}/00.png").size; s0 = a.res / max(W0, H0)
    Wr, Hr = int(round(W0 * s0)), int(round(H0 * s0))
    vms, Ks = load_cams(f"{SD}/{frames[0]}/cameras.json", Wr, Hr, dev)
    tv = a.ti / (T - 1)
    tl, tr = torch.exp(P["log_tl"]), torch.exp(P["log_tr"]); le, re = P["mu_t"] - tl, P["mu_t"] + tr
    ta = torch.where(le < 0.02, torch.ones_like(le), torch.sigmoid((tv - le) / GAMMA)) * \
         torch.where(re > 0.98, torch.ones_like(re), torch.sigmoid((re - tv) / GAMMA))
    s = tv - 0.5
    pos = P["means"] + P["v1"] * s + P["v2"] * s * s + P["v3"] * s ** 3
    al = (torch.sigmoid(P["opac"]) * ta).clamp(1e-4, 1 - 1e-4)
    with torch.no_grad():
        col, _, _ = rasterization(pos, F.normalize(P["quats"], dim=-1), torch.exp(P["scales"]).clamp(max=0.06), al,
                                  torch.cat([P["sh0"], P["shN"]], 1), vms[a.cam][None], Ks[a.cam][None],
                                  Wr, Hr, sh_degree=3, render_mode="RGB")
    bs = torch.zeros(Hr, Wr, 3, device=dev); bc = torch.zeros(Hr, Wr, 1, device=dev)
    for f in frames:
        g = torch.tensor(np.asarray(Image.open(f"{SD}/{f}/{a.cam:02d}.png").convert("RGB").resize((Wr, Hr))) / 255., dtype=torch.float32, device=dev)
        mm = torch.tensor((np.asarray(Image.open(f"{SD}/{f}/masks/{a.cam:02d}.png").convert("L").resize((Wr, Hr), Image.NEAREST)) > 127).astype(np.float32), device=dev)[..., None]
        bs += g * (1 - mm); bc += (1 - mm)
    plate = bs / bc.clamp(min=1)
    if (bc < 1).any(): plate[bc[..., 0] < 1] = plate[bc[..., 0] >= 1].median(0).values
    m = torch.tensor((np.asarray(Image.open(f"{SD}/{frames[a.ti]}/masks/{a.cam:02d}.png").convert("L").resize((Wr, Hr), Image.NEAREST)) > 127).astype(np.float32), device=dev)[..., None]
    m = -F.max_pool2d(-m.permute(2, 0, 1)[None], 3, 1, 1)[0].permute(1, 2, 0)
    ours = (col[0].clamp(0, 1) * m + plate * (1 - m)).clamp(0, 1).cpu().numpy()
    gt = np.asarray(Image.open(f"{SD}/{frames[a.ti]}/{a.cam:02d}.png").convert("RGB").resize((Wr, Hr))) / 255.
    ys, xs = np.where(m[..., 0].cpu().numpy() > 0.5)
    by0, by1, bx0, bx1 = ys.min(), ys.max(), xs.min(), xs.max()
    cx = bx0 + fx * (bx1 - bx0); cy = by0 + fy * (by1 - by0)
    ww = fw * (bx1 - bx0); hh = ww / a.crop_ar
    x0 = int(max(0, min(cx - ww / 2, Wr - ww))); y0 = int(max(0, min(cy - hh / 2, Hr - hh)))
    x1, y1 = int(min(Wr, x0 + ww)), int(min(Hr, y0 + hh))
    del ck, P; torch.cuda.empty_cache()
    return (Image.fromarray((gt[y0:y1, x0:x1] * 255).astype(np.uint8)),
            Image.fromarray((ours[y0:y1, x0:x1] * 255).astype(np.uint8)))

cols = []
for spec in a.cols.split(";"):
    scene, label, fx, fy, fw = spec.split(":")
    cols.append((label, do(scene, float(fx), float(fy), float(fw)))); print("done", scene, flush=True)

TW = a.tile; TH = int(TW * cols[0][1][0].height / cols[0][1][0].width)
GAP, HDR, LAB = 6, int(TW * 0.15), int(TW * 0.16)
n = len(cols)
canvas = Image.new("RGB", (LAB + n * TW + (n - 1) * GAP, HDR + 2 * TH + GAP), (255, 255, 255))
d = ImageDraw.Draw(canvas)
def fnt(sz):
    try: return ImageFont.truetype("/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf", sz)
    except Exception: return ImageFont.load_default()
FB = fnt(int(TW * 0.085))
for k, (label, (gt, ours)) in enumerate(cols):
    x = LAB + k * (TW + GAP); bb = d.textbbox((0, 0), label, font=FB)
    d.text((x + (TW - (bb[2] - bb[0])) / 2, (HDR - (bb[3] - bb[1])) / 2 - bb[1]), label, font=FB, fill=(20, 20, 20))
    canvas.paste(gt.resize((TW, TH), Image.LANCZOS), (x, HDR))
    canvas.paste(ours.resize((TW, TH), Image.LANCZOS), (x, HDR + TH + GAP))
for r, name in enumerate(["GT", "Ours"]):
    strip = Image.new("RGB", (TH, LAB), (255, 255, 255)); sd = ImageDraw.Draw(strip)
    bb = sd.textbbox((0, 0), name, font=FB)
    sd.text(((TH - (bb[2] - bb[0])) / 2, (LAB - (bb[3] - bb[1])) / 2 - bb[1]), name, font=FB, fill=(20, 20, 20))
    canvas.paste(strip.rotate(90, expand=True), (0, HDR + r * (TH + GAP)))
canvas.save(a.out)
print("saved", a.out, canvas.size)
