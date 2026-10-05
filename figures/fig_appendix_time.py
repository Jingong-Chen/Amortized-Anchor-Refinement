"""Appendix figure: continuous-time rendering between two supervised frames. The trajectory is a
function of time, so the field can be evaluated at timestamps that no frame of the capture
observes. Top row shows ground truth where it exists, bottom row our render at every step.
Usage: fig_appendix_time.py --scene doll --t0 14 --steps 8
"""
import os, sys, glob, argparse
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from gsplat import rasterization
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/aar")
from teacher_retime import load_cams

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="doll")
ap.add_argument("--ckpt", default="")
ap.add_argument("--cam", type=int, default=3)
ap.add_argument("--t0", type=int, default=14)             # supervised (even) frame the strip starts on
ap.add_argument("--span", type=int, default=2)            # frames covered, ending on another supervised frame
ap.add_argument("--steps", type=int, default=8)           # sub-steps across the span
ap.add_argument("--res", type=int, default=900); ap.add_argument("--n_frames", type=int, default=33)
ap.add_argument("--tile", type=int, default=250); ap.add_argument("--crop", type=float, default=0.06)
ap.add_argument("--crop_ar", type=float, default=0.80)
ap.add_argument("--out", default=f"{ROOT}/paper_vr/picture/appendix_time.png")
a = ap.parse_args(); dev = "cuda"; GAMMA = 0.005

ck = torch.load(a.ckpt or sorted(glob.glob(f"{ROOT}/experiments/outputs/nine_models/{a.scene}/model_*_96000.pt"))[0], map_location=dev)
P = {k: ck[k].to(dev) for k in ("means", "scales", "quats", "opac", "sh0", "shN", "mu_t", "log_tl", "log_tr", "v1", "v2", "v3")}
SD = f"{ROOT}/data/sc_mv_all/{a.scene}"
frames = sorted(d for d in os.listdir(SD) if d.isdigit())[:a.n_frames]; T = len(frames)
W0, H0 = Image.open(f"{SD}/{frames[0]}/00.png").size; s0 = a.res / max(W0, H0)
Wr, Hr = int(round(W0 * s0)), int(round(H0 * s0))
vms, Ks = load_cams(f"{SD}/{frames[0]}/cameras.json", Wr, Hr, dev)

def render(tv):
    tl, tr = torch.exp(P["log_tl"]), torch.exp(P["log_tr"]); le, re = P["mu_t"] - tl, P["mu_t"] + tr
    ta = torch.where(le < 0.02, torch.ones_like(le), torch.sigmoid((tv - le) / GAMMA)) * \
         torch.where(re > 0.98, torch.ones_like(re), torch.sigmoid((re - tv) / GAMMA))
    s = tv - 0.5
    pos = P["means"] + P["v1"] * s + P["v2"] * s * s + P["v3"] * s ** 3
    al = (torch.sigmoid(P["opac"]) * ta).clamp(1e-4, 1 - 1e-4)
    with torch.no_grad():
        col, alp, _ = rasterization(pos, F.normalize(P["quats"], dim=-1), torch.exp(P["scales"]).clamp(max=0.06), al,
                                    torch.cat([P["sh0"], P["shN"]], 1), vms[a.cam][None], Ks[a.cam][None],
                                    Wr, Hr, sh_degree=3, render_mode="RGB")
    return col[0].clamp(0, 1), alp[0].clamp(0, 1)

bs = torch.zeros(Hr, Wr, 3, device=dev); bc = torch.zeros(Hr, Wr, 1, device=dev)
for f in frames:
    g = torch.tensor(np.asarray(Image.open(f"{SD}/{f}/{a.cam:02d}.png").convert("RGB").resize((Wr, Hr))) / 255., dtype=torch.float32, device=dev)
    mm = torch.tensor((np.asarray(Image.open(f"{SD}/{f}/masks/{a.cam:02d}.png").convert("L").resize((Wr, Hr), Image.NEAREST)) > 127).astype(np.float32), device=dev)[..., None]
    bs += g * (1 - mm); bc += (1 - mm)
plate = bs / bc.clamp(min=1)
if (bc < 1).any(): plate[bc[..., 0] < 1] = plate[bc[..., 0] >= 1].median(0).values

# crop window: union of the subject silhouette over the covered frames, so nothing leaves the box
acc = np.zeros((Hr, Wr), bool)
for k in range(a.t0, a.t0 + a.span + 1):
    acc |= np.asarray(Image.open(f"{SD}/{frames[k]}/masks/{a.cam:02d}.png").convert("L").resize((Wr, Hr), Image.NEAREST)) > 127
ys, xs = np.where(acc)
y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
hh = (y1 - y0) * (1 + 2 * a.crop); ww = max((x1 - x0) * (1 + 2 * a.crop), hh * a.crop_ar); hh = max(hh, ww / a.crop_ar)
cy0 = int(max(0, min(cy - hh / 2, Hr - hh))); cx0 = int(max(0, min(cx - ww / 2, Wr - ww)))
cy1, cx1 = int(min(Hr, cy0 + hh)), int(min(Wr, cx0 + ww))

times = [a.t0 + a.span * i / a.steps for i in range(a.steps + 1)]
ours, gts, kinds = [], [], []
for tf in times:
    rgb, alp = render(tf / (T - 1))
    comp = (rgb * alp + plate * (1 - alp)).clamp(0, 1).cpu().numpy()
    ours.append(Image.fromarray((comp[cy0:cy1, cx0:cx1] * 255).astype(np.uint8)))
    if abs(tf - round(tf)) < 1e-6:
        k = int(round(tf))
        g = np.asarray(Image.open(f"{SD}/{frames[k]}/{a.cam:02d}.png").convert("RGB").resize((Wr, Hr))) / 255.
        gts.append(Image.fromarray((g[cy0:cy1, cx0:cx1] * 255).astype(np.uint8)))
        kinds.append("supervised" if k % 2 == 0 else "held out")
    else:
        gts.append(None); kinds.append("no frame")
print("rendered", len(times), "timestamps")

TW = a.tile; TH = int(TW * ours[0].height / ours[0].width)
GAP, HDR, LAB, FOOT = 6, int(TW * 0.20), int(TW * 0.34), int(TW * 0.17)
n = len(times)
canvas = Image.new("RGB", (LAB + n * TW + (n - 1) * GAP, HDR + 2 * TH + GAP + FOOT), (255, 255, 255))
d = ImageDraw.Draw(canvas)
def fnt(sz, bold=True):
    p = "/usr/share/fonts/opentype/urw-base35/NimbusRoman-%s.otf" % ("Bold" if bold else "Regular")
    try: return ImageFont.truetype(p, sz)
    except Exception: return ImageFont.load_default()
FB, FR = fnt(int(TW * 0.115)), fnt(int(TW * 0.095), False)
def centre(txt, x, y, h, font, fill=(20, 20, 20)):
    bb = d.textbbox((0, 0), txt, font=font)
    d.text((x + (TW - (bb[2] - bb[0])) / 2, y + (h - (bb[3] - bb[1])) / 2 - bb[1]), txt, font=font, fill=fill)
for k, tf in enumerate(times):
    x = LAB + k * (TW + GAP)
    centre(("%d" % round(tf)) if abs(tf - round(tf)) < 1e-6 else "%.2f" % tf, x, 0, HDR, FB)
    if gts[k] is not None:
        canvas.paste(gts[k].resize((TW, TH), Image.LANCZOS), (x, HDR))
    else:
        d.rectangle([x, HDR, x + TW, HDR + TH], fill=(244, 244, 244), outline=(215, 215, 215), width=2)
    canvas.paste(ours[k].resize((TW, TH), Image.LANCZOS), (x, HDR + TH + GAP))
    centre(kinds[k], x, HDR + 2 * TH + GAP, FOOT, FR, (110, 110, 110))
for r, name in enumerate(["Capture", "Ours"]):
    strip = Image.new("RGB", (TH, LAB), (255, 255, 255)); sd = ImageDraw.Draw(strip)
    bb = sd.textbbox((0, 0), name, font=FB)
    sd.text(((TH - (bb[2] - bb[0])) / 2, (LAB - (bb[3] - bb[1])) / 2 - bb[1]), name, font=FB, fill=(20, 20, 20))
    canvas.paste(strip.rotate(90, expand=True), (0, HDR + r * (TH + GAP)))
canvas.save(a.out)
print("saved", a.out, canvas.size)
