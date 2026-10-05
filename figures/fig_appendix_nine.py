"""Appendix figure: ground truth and our reconstruction on all nine scenes, at a held camera and a
held timestamp, composited over the person-free studio plate the same way as the main figures.
Usage: fig_appendix_nine.py --out picture/appendix_nine.png"""
import os, sys, glob, argparse
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from gsplat import rasterization
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/aar")
from teacher_retime import load_cams

ap = argparse.ArgumentParser()
ap.add_argument("--scenes", default="walk,bear,doll,newundress,opencase,passdoll,pickupdoll,putcomputer,stretch")
ap.add_argument("--cam", type=int, default=3); ap.add_argument("--ti", type=int, default=16)
ap.add_argument("--res", type=int, default=512); ap.add_argument("--n_frames", type=int, default=33)
ap.add_argument("--tile", type=int, default=300)
ap.add_argument("--crop", type=float, default=0.0)   # >0: crop to the subject box padded by this fraction
ap.add_argument("--crop_ar", type=float, default=0.80)
ap.add_argument("--out", default=f"{ROOT}/paper_vr/picture/appendix_nine.png")
a = ap.parse_args(); dev = "cuda"
GAMMA = 0.005

def render_scene(scene):
    ck = torch.load(sorted(glob.glob(f"{ROOT}/experiments/outputs/nine_models/{scene}/model_*_96000.pt"))[0], map_location=dev)
    P = {k: ck[k].to(dev) for k in ("means", "scales", "quats", "opac", "sh0", "shN", "mu_t", "log_tl", "log_tr", "v1", "v2", "v3")}
    SD = f"{ROOT}/data/sc_mv_all/{scene}"
    frames = sorted(d for d in os.listdir(SD) if d.isdigit())[:a.n_frames]; T = len(frames)
    im0 = Image.open(f"{SD}/{frames[0]}/00.png"); W0, H0 = im0.size; sc = a.res / max(W0, H0)
    Wr, Hr = int(round(W0 * sc)), int(round(H0 * sc))
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
    pred = col[0].clamp(0, 1)
    bs = torch.zeros(Hr, Wr, 3, device=dev); bc = torch.zeros(Hr, Wr, 1, device=dev)
    for f in frames:
        g = torch.tensor(np.asarray(Image.open(f"{SD}/{f}/{a.cam:02d}.png").convert("RGB").resize((Wr, Hr))) / 255., dtype=torch.float32, device=dev)
        mm = torch.tensor((np.asarray(Image.open(f"{SD}/{f}/masks/{a.cam:02d}.png").convert("L").resize((Wr, Hr), Image.NEAREST)) > 127).astype(np.float32), device=dev)[..., None]
        bs += g * (1 - mm); bc += (1 - mm)
    plate = bs / bc.clamp(min=1)
    if (bc < 1).any(): plate[bc[..., 0] < 1] = plate[bc[..., 0] >= 1].median(0).values
    gt = torch.tensor(np.asarray(Image.open(f"{SD}/{frames[a.ti]}/{a.cam:02d}.png").convert("RGB").resize((Wr, Hr))) / 255., dtype=torch.float32, device=dev)
    m = torch.tensor((np.asarray(Image.open(f"{SD}/{frames[a.ti]}/masks/{a.cam:02d}.png").convert("L").resize((Wr, Hr), Image.NEAREST)) > 127).astype(np.float32), device=dev)[..., None]
    m = -F.max_pool2d(-m.permute(2, 0, 1)[None], 3, 1, 1)[0].permute(1, 2, 0)
    ours = (pred * m + plate * (1 - m)).clamp(0, 1)
    del ck, P; torch.cuda.empty_cache()
    g, o = gt.cpu().numpy(), ours.cpu().numpy()
    if a.crop > 0:
        ys, xs = np.where(m[..., 0].cpu().numpy() > 0.5)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        h = (y1 - y0) * (1 + 2 * a.crop); w = max((x1 - x0) * (1 + 2 * a.crop), h * a.crop_ar)
        h = max(h, w / a.crop_ar)
        y0 = int(max(0, min(cy - h / 2, Hr - h))); x0 = int(max(0, min(cx - w / 2, Wr - w)))
        y1 = int(min(Hr, y0 + h)); x1 = int(min(Wr, x0 + w))
        g, o = g[y0:y1, x0:x1], o[y0:y1, x0:x1]
    return g, o

scenes = a.scenes.split(",")
cols = []
for sc in scenes:
    gt, ours = render_scene(sc)
    cols.append((sc, Image.fromarray((gt * 255).astype(np.uint8)), Image.fromarray((ours * 255).astype(np.uint8))))
    print("rendered", sc, flush=True)

TW = a.tile; TH = int(TW * cols[0][1].height / cols[0][1].width)
GAP = 6; HDR = int(TW * 0.16); LAB = int(TW * 0.20)
n = len(cols)
canvas = Image.new("RGB", (LAB + n * TW + (n - 1) * GAP, HDR + 2 * TH + GAP), (255, 255, 255))
d = ImageDraw.Draw(canvas)
try:
    FB = ImageFont.truetype("/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf", int(TW * 0.10))
except Exception:
    FB = ImageFont.load_default()
for k, (sc, gt, ours) in enumerate(cols):
    x = LAB + k * (TW + GAP)
    bb = d.textbbox((0, 0), sc, font=FB)
    d.text((x + (TW - (bb[2] - bb[0])) / 2, (HDR - (bb[3] - bb[1])) / 2 - bb[1]), sc, font=FB, fill=(20, 20, 20))
    canvas.paste(gt.resize((TW, TH), Image.LANCZOS), (x, HDR))
    canvas.paste(ours.resize((TW, TH), Image.LANCZOS), (x, HDR + TH + GAP))
for r, name in enumerate(["GT", "Ours"]):
    strip = Image.new("RGB", (TH, LAB), (255, 255, 255))
    sd = ImageDraw.Draw(strip)
    bb = sd.textbbox((0, 0), name, font=FB)
    sd.text(((TH - (bb[2] - bb[0])) / 2, (LAB - (bb[3] - bb[1])) / 2 - bb[1]), name, font=FB, fill=(20, 20, 20))
    canvas.paste(strip.rotate(90, expand=True), (0, HDR + r * (TH + GAP)))
canvas.save(a.out)
print("saved", a.out, canvas.size)
