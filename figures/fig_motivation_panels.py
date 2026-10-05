"""Export the motivation comparison as six separate panels: wide view and zoom for the
from-scratch field, ours, and ground truth. The wide views carry the box that marks the zoomed
region, so the pair can be laid out freely in a slide.
Usage: fig_motivation_panels.py --out docs/fig1_panels
"""
import os, glob, sys, argparse
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw
from gsplat import rasterization
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/aar")
from teacher_retime import load_cams

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="walk")
ap.add_argument("--scratch_ckpt", default="")             # from-scratch model for this scene (skipped when absent)
ap.add_argument("--ours_ckpt", default="")
ap.add_argument("--cam", type=int, default=3); ap.add_argument("--ti", type=int, default=16)
ap.add_argument("--res", type=int, default=1024); ap.add_argument("--n_frames", type=int, default=33)
ap.add_argument("--box_color", default="255,170,40"); ap.add_argument("--box_w", type=int, default=5)
ap.add_argument("--zoom_h", type=float, default=0.34)     # zoom box height as a fraction of the image height
ap.add_argument("--out", default=f"{ROOT}/docs/fig1_panels")
a = ap.parse_args(); dev = "cuda"

def load_model(path):
    d = torch.load(path, map_location=dev)
    return {k: d[k].to(dev) for k in ("means", "scales", "quats", "opac", "sh0", "shN",
                                      "mu_t", "log_tl", "log_tr", "v1", "v2", "v3")}
_scr = a.scratch_ckpt or (glob.glob(f"{ROOT}/experiments/outputs/naive_{a.scene}/model_from-scratch*.pt") or
                          glob.glob(f"{ROOT}/experiments/outputs/naive_viz/model_from-scratch*.pt") if a.scene == "walk" else
                          glob.glob(f"{ROOT}/experiments/outputs/naive_{a.scene}/model_from-scratch*.pt"))
_scr = _scr[0] if isinstance(_scr, list) and _scr else (_scr if isinstance(_scr, str) else "")
M_scr = load_model(_scr) if _scr else None
M_our = load_model(a.ours_ckpt or sorted(glob.glob(f"{ROOT}/experiments/outputs/nine_models/{a.scene}/model_*_96000.pt"))[0])

SD = f"{ROOT}/data/sc_mv_all/{a.scene}"
frames = sorted(d for d in os.listdir(SD) if d.isdigit())[:a.n_frames]; T = len(frames)
im0 = Image.open(f"{SD}/{frames[0]}/00.png"); W0, H0 = im0.size; sc = a.res / max(W0, H0)
Wr, Hr = int(round(W0 * sc)), int(round(H0 * sc))
vms, Ks = load_cams(f"{SD}/{frames[0]}/cameras.json", Wr, Hr, dev)
GAMMA = 0.005

def render(P, tv):
    tl, tr = torch.exp(P["log_tl"]), torch.exp(P["log_tr"]); le, re = P["mu_t"] - tl, P["mu_t"] + tr
    ta = torch.where(le < 0.02, torch.ones_like(le), torch.sigmoid((tv - le) / GAMMA)) * \
         torch.where(re > 0.98, torch.ones_like(re), torch.sigmoid((re - tv) / GAMMA))
    s = tv - 0.5
    pos = P["means"] + P["v1"] * s + P["v2"] * s * s + P["v3"] * s ** 3
    al = (torch.sigmoid(P["opac"]) * ta).clamp(1e-4, 1 - 1e-4)
    col, _, _ = rasterization(pos, F.normalize(P["quats"], dim=-1), torch.exp(P["scales"]).clamp(max=0.06), al,
                              torch.cat([P["sh0"], P["shN"]], 1), vms[a.cam][None], Ks[a.cam][None],
                              Wr, Hr, sh_degree=3, render_mode="RGB")
    return col[0].clamp(0, 1)

bs = torch.zeros(Hr, Wr, 3, device=dev); bc = torch.zeros(Hr, Wr, 1, device=dev)
for f in frames:                                           # person-free studio plate for compositing
    g = torch.tensor(np.asarray(Image.open(f"{SD}/{f}/{a.cam:02d}.png").convert("RGB").resize((Wr, Hr))) / 255., dtype=torch.float32, device=dev)
    mm = torch.tensor((np.asarray(Image.open(f"{SD}/{f}/masks/{a.cam:02d}.png").convert("L").resize((Wr, Hr), Image.NEAREST)) > 127).astype(np.float32), device=dev)[..., None]
    bs += g * (1 - mm); bc += (1 - mm)
plate = bs / bc.clamp(min=1)
if (bc < 1).any(): plate[bc[..., 0] < 1] = plate[bc[..., 0] >= 1].median(0).values

tv = a.ti / (T - 1)
gt = torch.tensor(np.asarray(Image.open(f"{SD}/{frames[a.ti]}/{a.cam:02d}.png").convert("RGB").resize((Wr, Hr))) / 255., dtype=torch.float32, device=dev)
m = torch.tensor((np.asarray(Image.open(f"{SD}/{frames[a.ti]}/masks/{a.cam:02d}.png").convert("L").resize((Wr, Hr), Image.NEAREST)) > 127).astype(np.float32), device=dev)[..., None]
m = -F.max_pool2d(-m.permute(2, 0, 1)[None], 3, 1, 1)[0].permute(1, 2, 0)

def comp(P): return (render(P, tv) * m + plate * (1 - m)).clamp(0, 1).cpu().numpy()
tiles = {"2_ours": comp(M_our), "3_gt": gt.cpu().numpy()}
if M_scr is not None: tiles["1_wo_anchor"] = comp(M_scr)

ys, xs = np.where(m[..., 0].cpu().numpy() > 0.5)           # zoom on the head and shoulders
cy = int(ys.min() + 0.18 * (ys.max() - ys.min())); cx = int((xs.min() + xs.max()) / 2)
zh = int(a.zoom_h * Hr); zw = int(zh * 0.85)
y0 = max(0, min(cy - zh // 2, Hr - zh)); x0 = max(0, min(cx - zw // 2, Wr - zw))
BOX = tuple(int(v) for v in a.box_color.split(","))

os.makedirs(a.out, exist_ok=True)
for name, arr in tiles.items():
    img = Image.fromarray((arr * 255).astype(np.uint8))
    wide = img.copy()
    ImageDraw.Draw(wide).rectangle([x0, y0, x0 + zw, y0 + zh], outline=BOX, width=a.box_w)
    wide.save(f"{a.out}/{name}_wide.png")
    zoom = img.crop((x0, y0, x0 + zw, y0 + zh))
    zoom = zoom.resize((zoom.width * 2, zoom.height * 2), Image.LANCZOS)
    ImageDraw.Draw(zoom).rectangle([1, 1, zoom.width - 2, zoom.height - 2], outline=BOX, width=a.box_w * 2)
    zoom.save(f"{a.out}/{name}_zoom.png")
    print("saved", name, wide.size, zoom.size)
print(f"[zoom box] x={x0} y={y0} w={zw} h={zh} of {Wr}x{Hr}")
