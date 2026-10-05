"""Appendix figure: held-timestamp interpolation against the published baselines on several scenes.
Both baselines dump their test renders in (held frame)-major order over 32 cameras, so a render is
addressed by (test index, camera). Ours is composited over the person-free plate as elsewhere.
Usage: fig_appendix_baselines.py --scenes bear,doll,passdoll,stretch
"""
import os, sys, glob, argparse
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from gsplat import rasterization
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/aar")
from teacher_retime import load_cams

ap = argparse.ArgumentParser()
ap.add_argument("--scenes", default="bear,doll,passdoll,stretch")
ap.add_argument("--cam", type=int, default=3); ap.add_argument("--ti", type=int, default=15)
ap.add_argument("--res", type=int, default=900); ap.add_argument("--n_frames", type=int, default=33)
ap.add_argument("--tile", type=int, default=300); ap.add_argument("--crop", type=float, default=0.10)
ap.add_argument("--crop_ar", type=float, default=0.80)
ap.add_argument("--out", default=f"{ROOT}/paper_vr/picture/appendix_baselines.png")
a = ap.parse_args(); dev = "cuda"; GAMMA = 0.005
assert a.ti % 2 == 1, "pick a held (odd) frame so the baselines have a render for it"

def our_render(P, tv, vms, Ks, Wr, Hr):
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
    return col[0].clamp(0, 1)

def baseline_dir(scene, kind):
    pat = {"dgs": f"{ROOT}/experiments/outputs/dgs33_{scene}/test/ours_*/renders",
           "4dgs": f"{ROOT}/experiments/outputs/fourdgs_{scene}/test/ours_*/renders"}[kind]
    d = sorted(glob.glob(pat))
    return d[-1] if d else ""

def do_scene(scene):
    ck = torch.load(sorted(glob.glob(f"{ROOT}/experiments/outputs/nine_models/{scene}/model_*_96000.pt"))[0], map_location=dev)
    P = {k: ck[k].to(dev) for k in ("means", "scales", "quats", "opac", "sh0", "shN", "mu_t", "log_tl", "log_tr", "v1", "v2", "v3")}
    SD = f"{ROOT}/data/sc_mv_all/{scene}"
    frames = sorted(d for d in os.listdir(SD) if d.isdigit())[:a.n_frames]; T = len(frames)
    W0, H0 = Image.open(f"{SD}/{frames[0]}/00.png").size; s0 = a.res / max(W0, H0)
    Wr, Hr = int(round(W0 * s0)), int(round(H0 * s0))
    vms, Ks = load_cams(f"{SD}/{frames[0]}/cameras.json", Wr, Hr, dev)
    bs = torch.zeros(Hr, Wr, 3, device=dev); bc = torch.zeros(Hr, Wr, 1, device=dev)
    for f in frames:
        g = torch.tensor(np.asarray(Image.open(f"{SD}/{f}/{a.cam:02d}.png").convert("RGB").resize((Wr, Hr))) / 255., dtype=torch.float32, device=dev)
        mm = torch.tensor((np.asarray(Image.open(f"{SD}/{f}/masks/{a.cam:02d}.png").convert("L").resize((Wr, Hr), Image.NEAREST)) > 127).astype(np.float32), device=dev)[..., None]
        bs += g * (1 - mm); bc += (1 - mm)
    plate = bs / bc.clamp(min=1)
    if (bc < 1).any(): plate[bc[..., 0] < 1] = plate[bc[..., 0] >= 1].median(0).values
    m = torch.tensor((np.asarray(Image.open(f"{SD}/{frames[a.ti]}/masks/{a.cam:02d}.png").convert("L").resize((Wr, Hr), Image.NEAREST)) > 127).astype(np.float32), device=dev)[..., None]
    m = -F.max_pool2d(-m.permute(2, 0, 1)[None], 3, 1, 1)[0].permute(1, 2, 0)
    gt = np.asarray(Image.open(f"{SD}/{frames[a.ti]}/{a.cam:02d}.png").convert("RGB").resize((Wr, Hr))) / 255.
    ours = ((our_render(P, a.ti / (T - 1), vms, Ks, Wr, Hr) * m + plate * (1 - m)).clamp(0, 1)).cpu().numpy()
    idx = [i for i in range(T) if i % 2 == 1].index(a.ti) * 32 + a.cam
    tiles = {}
    for kind in ("dgs", "4dgs"):
        d = baseline_dir(scene, kind)
        f = sorted(glob.glob(f"{d}/*.png"))[idx] if d else ""
        tiles[kind] = np.asarray(Image.open(f).convert("RGB").resize((Wr, Hr))) / 255. if f else np.ones((Hr, Wr, 3))
    tiles["ours"] = ours; tiles["gt"] = gt
    ys, xs = np.where(m[..., 0].cpu().numpy() > 0.5)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    hh = (y1 - y0) * (1 + 2 * a.crop); ww = max((x1 - x0) * (1 + 2 * a.crop), hh * a.crop_ar); hh = max(hh, ww / a.crop_ar)
    cy0 = int(max(0, min(cy - hh / 2, Hr - hh))); cx0 = int(max(0, min(cx - ww / 2, Wr - ww)))
    cy1, cx1 = int(min(Hr, cy0 + hh)), int(min(Wr, cx0 + ww))
    del ck, P; torch.cuda.empty_cache()
    return {k: Image.fromarray((v[cy0:cy1, cx0:cx1] * 255).astype(np.uint8)) for k, v in tiles.items()}

scenes = a.scenes.split(",")
rows = []
for s in scenes:
    rows.append((s, do_scene(s))); print("done", s, flush=True)

order = [("dgs", "Deform-3DGS"), ("4dgs", "4D-GS"), ("ours", "Ours"), ("gt", "Ground truth")]
TW = a.tile; TH = int(TW * rows[0][1]["gt"].height / rows[0][1]["gt"].width)
GAP, HDR, LAB = 6, int(TW * 0.16), int(TW * 0.30)
n = len(order)
canvas = Image.new("RGB", (LAB + n * TW + (n - 1) * GAP, HDR + len(rows) * (TH + GAP) - GAP), (255, 255, 255))
d = ImageDraw.Draw(canvas)
def fnt(sz):
    try: return ImageFont.truetype("/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf", sz)
    except Exception: return ImageFont.load_default()
FB = fnt(int(TW * 0.10))
for k, (_, h) in enumerate(order):
    x = LAB + k * (TW + GAP); bb = d.textbbox((0, 0), h, font=FB)
    d.text((x + (TW - (bb[2] - bb[0])) / 2, (HDR - (bb[3] - bb[1])) / 2 - bb[1]), h, font=FB, fill=(20, 20, 20))
for r, (sname, tiles) in enumerate(rows):
    for k, (key, _) in enumerate(order):
        canvas.paste(tiles[key].resize((TW, TH), Image.LANCZOS), (LAB + k * (TW + GAP), HDR + r * (TH + GAP)))
    strip = Image.new("RGB", (TH, LAB), (255, 255, 255)); sd = ImageDraw.Draw(strip)
    bb = sd.textbbox((0, 0), sname, font=FB)
    sd.text(((TH - (bb[2] - bb[0])) / 2, (LAB - (bb[3] - bb[1])) / 2 - bb[1]), sname, font=FB, fill=(20, 20, 20))
    canvas.paste(strip.rotate(90, expand=True), (0, HDR + r * (TH + GAP)))
canvas.save(a.out)
print("saved", a.out, canvas.size)
