"""Pruning-criterion ablation figure (paper Fig. 5). At a fixed budget, compare
Random | Opacity | Importance | +Topology floor. Top row: person-on-white render on a
held camera/frame with scene-level held PSNR under each method name. Bottom row: zoom
of the region where importance and topology pruning differ most.
Usage: fig_tfs_ablation.py --scene walk --budget 20000 [--cam -1 --t -1]"""
import os, argparse, math, sys, json
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
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="")
a = ap.parse_args(); dev = "cuda"
ck_path = sorted(__import__("glob").glob(f"{ROOT}/experiments/outputs/nine_models/{a.scene}/model_*_96000.pt"))[0]
ck = torch.load(ck_path, map_location=dev)
means, scales, quats = ck["means"].to(dev), ck["scales"].to(dev), ck["quats"].to(dev)
opac, sh0, shN = ck["opac"].to(dev), ck["sh0"].to(dev), ck["shN"].to(dev)
mu_t, log_tl, log_tr = ck["mu_t"].to(dev), ck["log_tl"].to(dev), ck["log_tr"].to(dev)
v1, v2, v3 = ck["v1"].to(dev), ck["v2"].to(dev), ck["v3"].to(dev)
N = means.shape[0]

# ---- keep sets at the same budget: random, two SOTA criteria, and ours (strongest base + floor) ----
MD = f"{ROOT}/experiments/outputs/morse_{a.scene}"
g = torch.Generator(device="cpu").manual_seed(a.seed)
keep_rand = torch.randperm(N, generator=g)[:a.budget].to(dev)
keep_lg = torch.load(f"{MD}/keep_lightgs_{a.budget}.pt", map_location=dev)
keep_ct = torch.load(f"{MD}/keep_contrib_{a.budget}.pt", map_location=dev)
keep_ours = torch.load(f"{MD}_opacity/keep_topo_{a.budget}.pt", map_location=dev)   # opacity base + verified floor

SD = f"{ROOT}/data/sc_mv_all/{a.scene}"
frames = sorted([d for d in os.listdir(SD) if d.isdigit()])[:a.n_frames]; T = len(frames)
im0 = Image.open(f"{SD}/{frames[0]}/00.png"); W0, H0 = im0.size; sc = a.res / max(W0, H0)
Wr, Hr = int(round(W0 * sc)), int(round(H0 * sc))
vms, Ks = load_cams(f"{SD}/{frames[0]}/cameras.json", Wr, Hr, dev); ncam = len(vms)
held_c = list(range(1, ncam, 2)); held_t = list(range(1, T, 2))
ci = held_c[len(held_c) // 2] if a.cam < 0 else a.cam
ti = held_t[len(held_t) // 2] if a.t < 0 else a.t

GAMMA = 0.005
def render(idx, cam, tv):
    tl = torch.exp(log_tl[idx]); tr = torch.exp(log_tr[idx]); mt = mu_t[idx]
    le = mt - tl; re = mt + tr
    ta = torch.where(le < 0.02, torch.ones_like(le), torch.sigmoid((tv - le) / GAMMA)) * \
         torch.where(re > 0.98, torch.ones_like(re), torch.sigmoid((re - tv) / GAMMA))
    s = tv - 0.5
    pos = means[idx] + v1[idx] * s + v2[idx] * s * s + v3[idx] * s ** 3
    al = (torch.sigmoid(opac[idx]) * ta).clamp(1e-4, 1 - 1e-4)
    col, _, _ = rasterization(pos, F.normalize(quats[idx], dim=-1), torch.exp(scales[idx]).clamp(max=0.06),
                              al, torch.cat([sh0[idx], shN[idx]], 1), vms[cam][None], Ks[cam][None],
                              Wr, Hr, sh_degree=3, render_mode="RGB")
    return col[0].clamp(0, 1)

# ---- scene-level held PSNR per keep set (identical protocol to morse_prune) ----
GT = {}
for c in held_c:
    for t in held_t:
        r = torch.tensor(np.asarray(Image.open(f"{SD}/{frames[t]}/{c:02d}.png").convert("RGB").resize((Wr, Hr))) / 255., dtype=torch.float32)
        m = torch.tensor((np.asarray(Image.open(f"{SD}/{frames[t]}/masks/{c:02d}.png").convert("L").resize((Wr, Hr), Image.NEAREST)) > 127).astype(np.float32))
        GT[(c, t)] = (r, m)
def eval_both(idx):
    es = []
    with torch.no_grad():
        for c in held_c:
            for t in held_t:
                r, m = GT[(c, t)]; m3 = m[..., None].to(dev)
                if m3.sum() < 10: continue
                es.append((((render(idx, c, t / (T - 1)) - r.to(dev)) ** 2 * m3).sum() / (m3.sum() * 3 + 1e-6)).item())
    return -10 * math.log10(float(np.mean(es)) + 1e-12)

METHODS = [("Random", keep_rand), ("LightGaussian", keep_lg), ("Contribution", keep_ct), ("Ours", keep_ours)]
cache_p = f"{ROOT}/experiments/outputs/morse_{a.scene}/abl_fig_{a.budget}.json"
cached = json.load(open(cache_p))["psnr"] if os.path.exists(cache_p) else {}
psnrs = []
for name, kp in METHODS:
    p = cached[name] if name in cached else eval_both(kp)
    psnrs.append(p); print(f"[{a.scene} {a.budget}] {name}: both={p:.2f}  N={kp.numel()}")

# ---- render figure tiles at the shown camera/frame (GT first, no dB label) ----
tv = ti / (T - 1)
with torch.no_grad():
    tiles_r = [render(kp, ci, tv) for _, kp in METHODS]
gtim, m = GT[(ci, ti)]
gtim = gtim.to(dev); m = m.to(dev)[..., None]
m = -F.max_pool2d(-m.permute(2, 0, 1)[None], 3, 1, 1)[0].permute(1, 2, 0)
bg = torch.ones(Hr, Wr, 3, device=dev)
tiles = [gtim * m + bg * (1 - m)] + [r * m + bg * (1 - m) for r in tiles_r]
names = ["GT"] + [n for n, _ in METHODS]
labels = [""] + [f"{p:.2f} dB" for p in psnrs]

# crop: window where the strongest baseline and ours differ most (thin-structure region)
diff = ((tiles_r[1] - tiles_r[3]).abs().mean(-1, keepdim=True) * m).permute(2, 0, 1)[None]
CW, CH = 170, 125
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
NC = len(tiles)
try:
    FONT = ImageFont.truetype("/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf", int(Wr * 0.125))
    FONT2 = ImageFont.truetype("/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf", int(Wr * 0.110))
except Exception:
    FONT = FONT2 = ImageFont.load_default()
H_NAME = int(Wr * 0.115); H_PSNR = int(Wr * 0.108); HDR = H_NAME + H_PSNR
canvas = Image.new("RGB", (NC * W_ + (NC - 1) * 6, HDR + H1 + H2 + 6), (255, 255, 255))
hd = ImageDraw.Draw(canvas)
for k in range(NC):
    x = k * (W_ + 6)
    bb = hd.textbbox((0, 0), names[k], font=FONT)
    hd.text((x + (W_ - (bb[2] - bb[0])) / 2, (H_NAME - (bb[3] - bb[1])) / 2 - bb[1]), names[k], font=FONT, fill=(20, 20, 20))
    if labels[k]:
        bb = hd.textbbox((0, 0), labels[k], font=FONT2)
        hd.text((x + (W_ - (bb[2] - bb[0])) / 2, H_NAME + (H_PSNR - (bb[3] - bb[1])) / 2 - bb[1]), labels[k], font=FONT2, fill=(80, 80, 80))
    canvas.paste(row1[k], (x, HDR)); canvas.paste(row2[k], (x, HDR + H1 + 6))
out = a.out or f"{ROOT}/paper_vr/picture/tfs_ablation.png"
canvas.save(out)
json.dump({"scene": a.scene, "budget": a.budget, "cam": ci, "t": ti,
           "psnr": {name: p for (name, _), p in zip(METHODS, psnrs)}},
          open(cache_p, "w"), indent=1)
print(f"saved {out}  cam{ci} t{ti} crop=({x0},{y0},{CW}x{CH})")
