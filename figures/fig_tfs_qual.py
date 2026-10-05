"""TFS qualitative figure with zoom insets. Top row: GT | full | importance-only | +topology floor
(person-on-black, held camera). Bottom row: 3x zoom of the region where the two prunings differ
most (auto-located), marked by a rectangle in the top row. Usage:
  fig_tfs_qual.py --scene walk --budget 20000 [--cam -1 --t -1]  (-1 = middle held)"""
import os, argparse, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from gsplat import rasterization
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/aar")
from teacher_retime import load_cams

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="walk"); ap.add_argument("--budget", type=int, default=20000)
ap.add_argument("--cam", type=int, default=-1); ap.add_argument("--t", type=int, default=-1)
ap.add_argument("--res", type=int, default=512); ap.add_argument("--n_frames", type=int, default=33)
ap.add_argument("--out", default="")
a = ap.parse_args(); dev = "cuda"
ck_path = sorted(__import__("glob").glob(f"{ROOT}/experiments/outputs/nine_models/{a.scene}/model_*_96000.pt"))[0]
ck = torch.load(ck_path, map_location=dev)
means, scales, quats = ck["means"].to(dev), ck["scales"].to(dev), ck["quats"].to(dev)
opac, sh0, shN = ck["opac"].to(dev), ck["sh0"].to(dev), ck["shN"].to(dev)
mu_t, log_tl, log_tr = ck["mu_t"].to(dev), ck["log_tl"].to(dev), ck["log_tr"].to(dev)
v1, v2, v3 = ck["v1"].to(dev), ck["v2"].to(dev), ck["v3"].to(dev)
N = means.shape[0]
keep_imp = torch.load(f"{ROOT}/experiments/outputs/morse_{a.scene}/keep_imp_{a.budget}.pt", map_location=dev)
keep_topo = torch.load(f"{ROOT}/experiments/outputs/morse_{a.scene}/keep_topo_{a.budget}.pt", map_location=dev)

SD = f"{ROOT}/data/sc_mv_all/{a.scene}"
frames = sorted([d for d in os.listdir(SD) if d.isdigit()])[:a.n_frames]; T = len(frames)
im0 = Image.open(f"{SD}/{frames[0]}/00.png"); W0, H0 = im0.size; sc = a.res / max(W0, H0)
Wr, Hr = int(round(W0 * sc)), int(round(H0 * sc))
vms, Ks = load_cams(f"{SD}/{frames[0]}/cameras.json", Wr, Hr, dev); ncam = len(vms)
held_c = list(range(1, ncam, 2)); held_t = list(range(1, T, 2))
ci = held_c[len(held_c) // 2] if a.cam < 0 else a.cam
ti = held_t[len(held_t) // 2] if a.t < 0 else a.t
tv = ti / (T - 1)

GAMMA = 0.005
def render(idx):
    tl = torch.exp(log_tl[idx]); tr = torch.exp(log_tr[idx]); mt = mu_t[idx]
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

with torch.no_grad():
    r_full = render(torch.arange(N, device=dev)); r_imp = render(keep_imp); r_topo = render(keep_topo)
gt = torch.tensor(np.asarray(Image.open(f"{SD}/{frames[ti]}/{ci:02d}.png").convert("RGB").resize((Wr, Hr))) / 255., dtype=torch.float32, device=dev)
m = torch.tensor((np.asarray(Image.open(f"{SD}/{frames[ti]}/masks/{ci:02d}.png").convert("L").resize((Wr, Hr), Image.NEAREST)) > 127).astype(np.float32), device=dev)[..., None]
m = -F.max_pool2d(-m.permute(2, 0, 1)[None], 3, 1, 1)[0].permute(1, 2, 0)   # erode 1px -> trim loose green edge
# background: white, or the static studio plate (person-free per-pixel bg mean of this camera's GT)
BG = os.environ.get("TFS_BG", "white")
if BG == "studio":
    bs = torch.zeros(Hr, Wr, 3, device=dev); bc = torch.zeros(Hr, Wr, 1, device=dev)
    for f in frames:
        g = torch.tensor(np.asarray(Image.open(f"{SD}/{f}/{ci:02d}.png").convert("RGB").resize((Wr, Hr))) / 255., dtype=torch.float32, device=dev)
        mm = torch.tensor((np.asarray(Image.open(f"{SD}/{f}/masks/{ci:02d}.png").convert("L").resize((Wr, Hr), Image.NEAREST)) > 127).astype(np.float32), device=dev)[..., None]
        bs += g * (1 - mm); bc += (1 - mm)
    bg = bs / bc.clamp(min=1)
    if (bc < 1).any(): bg[bc[..., 0] < 1] = bg[bc[..., 0] >= 1].median(0).values
else:
    bg = torch.ones(Hr, Wr, 3, device=dev)
tiles = [gt * m + bg * (1 - m), r_full * m + bg * (1 - m), r_imp * m + bg * (1 - m), r_topo * m + bg * (1 - m)]

# auto-locate the crop: window with max |imp - floor| inside the mask
diff = ((r_imp - r_topo).abs().mean(-1, keepdim=True) * m).permute(2, 0, 1)[None]
CW, CH = 170, 125                                          # crop size (4:3-ish; x3 zoom -> 510x375)
score = F.avg_pool2d(diff, (CH, CW), stride=8)
iy, ix = np.unravel_index(score[0, 0].argmax().cpu().numpy(), score[0, 0].shape)
y0, x0 = int(iy * 8), int(ix * 8)
y0 = min(max(0, y0), Hr - CH); x0 = min(max(0, x0), Wr - CW)

def to_img(t): return Image.fromarray((t.cpu().numpy() * 255).astype(np.uint8))
row1, row2 = [], []
YELLOW = (255, 210, 60)
for t in tiles:
    im = to_img(t)
    d = ImageDraw.Draw(im); d.rectangle([x0, y0, x0 + CW, y0 + CH], outline=YELLOW, width=3)
    row1.append(im)
    zoom = to_img(t).crop((x0, y0, x0 + CW, y0 + CH)).resize((Wr, int(CH * Wr / CW)), Image.LANCZOS)
    zd = ImageDraw.Draw(zoom); zd.rectangle([1, 1, zoom.width - 2, zoom.height - 2], outline=YELLOW, width=4)
    row2.append(zoom)
W_, H1, H2 = Wr, Hr, row2[0].height
# column headers, matching Table 5's wording
TITLES = ["GT", "Full field", "Importance", "+Topology"]
try:
    FONT = ImageFont.truetype("/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf", int(Wr * 0.115))
except Exception:
    FONT = ImageFont.load_default()
HDR = int(Wr * 0.105)
canvas = Image.new("RGB", (4 * W_ + 3 * 6, HDR + H1 + H2 + 6), (255, 255, 255))
hd = ImageDraw.Draw(canvas)
for k in range(4):
    x = k * (W_ + 6)
    bb = hd.textbbox((0, 0), TITLES[k], font=FONT)
    hd.text((x + (W_ - (bb[2] - bb[0])) / 2, (HDR - (bb[3] - bb[1])) / 2 - bb[1]),
            TITLES[k], font=FONT, fill=(20, 20, 20))
    canvas.paste(row1[k], (x, HDR)); canvas.paste(row2[k], (x, HDR + H1 + 6))
out = a.out or f"{ROOT}/paper_vr/picture/tfs_qual.png"
canvas.save(out)
print(f"saved {out}  cam{ci} t{ti} budget{a.budget}  crop=({x0},{y0},{CW}x{CH})")
