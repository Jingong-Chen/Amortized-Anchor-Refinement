"""Appendix figure: how a package degrades along the budget ladder, with and without the verified
topology floor. Leading column carries the reference (ground truth over the uncompressed field);
the remaining columns are packages at decreasing Gaussian budgets.
Usage: fig_appendix_ladder.py --scene opencase --dir morse_opencase_opacity
"""
import os, sys, glob, argparse
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from gsplat import rasterization
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/aar")
from teacher_retime import load_cams

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="opencase")
ap.add_argument("--dir", default="morse_opencase_opacity")
ap.add_argument("--budgets", default="80000,40000,20000,10000,5000")
ap.add_argument("--cam", type=int, default=3); ap.add_argument("--ti", type=int, default=16)
ap.add_argument("--res", type=int, default=900); ap.add_argument("--n_frames", type=int, default=33)
ap.add_argument("--tile", type=int, default=300); ap.add_argument("--crop", type=float, default=0.10)
ap.add_argument("--crop_ar", type=float, default=0.80)
ap.add_argument("--out", default=f"{ROOT}/paper_vr/picture/appendix_ladder.png")
a = ap.parse_args(); dev = "cuda"; GAMMA = 0.005

ck = torch.load(sorted(glob.glob(f"{ROOT}/experiments/outputs/nine_models/{a.scene}/model_*_96000.pt"))[0], map_location=dev)
P = {k: ck[k].to(dev) for k in ("means", "scales", "quats", "opac", "sh0", "shN", "mu_t", "log_tl", "log_tr", "v1", "v2", "v3")}
SD = f"{ROOT}/data/sc_mv_all/{a.scene}"
frames = sorted(d for d in os.listdir(SD) if d.isdigit())[:a.n_frames]; T = len(frames)
W0, H0 = Image.open(f"{SD}/{frames[0]}/00.png").size; s0 = a.res / max(W0, H0)
Wr, Hr = int(round(W0 * s0)), int(round(H0 * s0))
vms, Ks = load_cams(f"{SD}/{frames[0]}/cameras.json", Wr, Hr, dev)
tv = a.ti / (T - 1)

def render(idx=None):
    tl, tr = torch.exp(P["log_tl"]), torch.exp(P["log_tr"]); le, re = P["mu_t"] - tl, P["mu_t"] + tr
    ta = torch.where(le < 0.02, torch.ones_like(le), torch.sigmoid((tv - le) / GAMMA)) * \
         torch.where(re > 0.98, torch.ones_like(re), torch.sigmoid((re - tv) / GAMMA))
    s = tv - 0.5
    pos = P["means"] + P["v1"] * s + P["v2"] * s * s + P["v3"] * s ** 3
    al = (torch.sigmoid(P["opac"]) * ta).clamp(1e-4, 1 - 1e-4)
    sh = torch.cat([P["sh0"], P["shN"]], 1)
    q, sc = F.normalize(P["quats"], dim=-1), torch.exp(P["scales"]).clamp(max=0.06)
    if idx is not None:
        pos, q, sc, al, sh = pos[idx], q[idx], sc[idx], al[idx], sh[idx]
    with torch.no_grad():
        col, _, _ = rasterization(pos, q, sc, al, sh, vms[a.cam][None], Ks[a.cam][None],
                                  Wr, Hr, sh_degree=3, render_mode="RGB")
    return col[0].clamp(0, 1)

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

ys, xs = np.where(m[..., 0].cpu().numpy() > 0.5)
y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
hh = (y1 - y0) * (1 + 2 * a.crop); ww = max((x1 - x0) * (1 + 2 * a.crop), hh * a.crop_ar); hh = max(hh, ww / a.crop_ar)
cy0 = int(max(0, min(cy - hh / 2, Hr - hh))); cx0 = int(max(0, min(cx - ww / 2, Wr - ww)))
cy1, cx1 = int(min(Hr, cy0 + hh)), int(min(Wr, cx0 + ww))

def comp(t): return Image.fromarray(((t * m + plate * (1 - m)).clamp(0, 1).cpu().numpy()[cy0:cy1, cx0:cx1] * 255).astype(np.uint8))
def crop(t): return Image.fromarray((t.cpu().numpy()[cy0:cy1, cx0:cx1] * 255).astype(np.uint8))

D = f"{ROOT}/experiments/outputs/{a.dir}"
budgets = [int(b) for b in a.budgets.split(",")]
col_head = ["Reference"] + [f"{b // 1000}k" if b >= 1000 else str(b) for b in budgets]
grid = [[crop(gt)], [comp(render())]]
for b in budgets:
    grid[0].append(comp(render(torch.load(f"{D}/keep_imp_{b}.pt", map_location=dev))))
    grid[1].append(comp(render(torch.load(f"{D}/keep_topo_{b}.pt", map_location=dev))))
    print("rendered", b, flush=True)
row_head = ["Ranking only", "Ranking $+$ floor"]

TW = a.tile; TH = int(TW * grid[0][0].height / grid[0][0].width)
GAP, HDR, LAB = 6, int(TW * 0.17), int(TW * 0.30)
n = len(col_head)
canvas = Image.new("RGB", (LAB + n * TW + (n - 1) * GAP, HDR + 2 * TH + GAP), (255, 255, 255))
d = ImageDraw.Draw(canvas)
def fnt(sz, bold=True):
    p = "/usr/share/fonts/opentype/urw-base35/NimbusRoman-%s.otf" % ("Bold" if bold else "Regular")
    try: return ImageFont.truetype(p, sz)
    except Exception: return ImageFont.load_default()
FB, FR = fnt(int(TW * 0.11)), fnt(int(TW * 0.085), False)
for k, h in enumerate(col_head):
    x = LAB + k * (TW + GAP)
    bb = d.textbbox((0, 0), h, font=FB)
    d.text((x + (TW - (bb[2] - bb[0])) / 2, (HDR - (bb[3] - bb[1])) / 2 - bb[1]), h, font=FB, fill=(20, 20, 20))
for r in range(2):
    for k in range(n):
        canvas.paste(grid[r][k].resize((TW, TH), Image.LANCZOS), (LAB + k * (TW + GAP), HDR + r * (TH + GAP)))
labels = [["Ground truth", "Uncompressed"][r] for r in range(2)]
for r, name in enumerate(["Ranking only", "Ours"]):
    strip = Image.new("RGB", (TH, LAB), (255, 255, 255)); sd = ImageDraw.Draw(strip)
    bb = sd.textbbox((0, 0), name, font=FB)
    sd.text(((TH - (bb[2] - bb[0])) / 2, (LAB - (bb[3] - bb[1])) / 2 - bb[1]), name, font=FB, fill=(20, 20, 20))
    canvas.paste(strip.rotate(90, expand=True), (0, HDR + r * (TH + GAP)))
for r, name in enumerate(labels):                      # caption the two reference tiles in-place
    bb = d.textbbox((0, 0), name, font=FR)
    bx = LAB + (TW - (bb[2] - bb[0])) / 2; by = HDR + r * (TH + GAP) + TH - (bb[3] - bb[1]) - 12
    d.rectangle([bx - 6, by - 4, bx + (bb[2] - bb[0]) + 6, by + (bb[3] - bb[1]) + 8], fill=(255, 255, 255))
    d.text((bx, by - bb[1]), name, font=FR, fill=(20, 20, 20))
canvas.save(a.out)
print("saved", a.out, canvas.size)
