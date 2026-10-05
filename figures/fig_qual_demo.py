"""Rebuild qualitative_demo: held-timestamp interpolation on a held camera, ours against the
official Deformable-3DGS baseline and the ground truth. Our DYNAMIC foreground is composited
over a STATIC studio plate.
- plate: per-pixel mean of GT frames over the timesteps where that pixel is background (mask=0)
  -> the moving person is removed cleanly, no ghost.
- composite: our render carries its own alpha; final = rgb + plate*(1-alpha). No GT mask needed,
  works at any continuous time, green-screen fringe blends into the studio.
"""
import os, glob, sys
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
for _f in ("NimbusRoman-Regular.otf", "NimbusRoman-Italic.otf", "NimbusRoman-Bold.otf"):
    _fm.fontManager.addfont(f"/usr/share/fonts/opentype/urw-base35/{_f}")
matplotlib.rcParams.update({"font.family": "Nimbus Roman",
    "mathtext.fontset": "custom", "mathtext.rm": "Nimbus Roman",
    "mathtext.it": "Nimbus Roman:italic", "mathtext.bf": "Nimbus Roman:bold"})
from PIL import Image
from gsplat import rasterization
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/aar")
from teacher_retime import load_cams
dev = "cuda"; scene = "walk"; res = 1024; nf = 33; CAM = 3
FIDX = [3, 9, 15, 19, 25, 29]                             # held (odd) frames; masks exist -> clip floaters

def load_model(path):
    d = torch.load(path, map_location=dev)
    return {k: d[k].to(dev) for k in ("means","scales","quats","opac","sh0","shN","mu_t","log_tl","log_tr","v1","v2","v3")}
M_interp = load_model(sorted(glob.glob(f"{ROOT}/experiments/outputs/nine_models/{scene}/model_*_96000.pt"))[0])
DGS = f"{ROOT}/experiments/outputs/dgs33_{scene}/test/ours_40000/renders"   # official Deform-3DGS

SD = f"{ROOT}/data/sc_mv_all/{scene}"; frames = sorted(d for d in os.listdir(SD) if d.isdigit())[:nf]; T = len(frames)
im0 = Image.open(f"{SD}/{frames[0]}/00.png"); W0, H0 = im0.size; sc = res / max(W0, H0)
Wr, Hr = int(round(W0*sc)), int(round(H0*sc))
vms, Ks = load_cams(f"{SD}/{frames[0]}/cameras.json", Wr, Hr, dev)
GAMMA = 0.005
def render(P, tv):
    tl, tr = torch.exp(P["log_tl"]), torch.exp(P["log_tr"]); le, re = P["mu_t"]-tl, P["mu_t"]+tr
    ta = torch.where(le<0.02, torch.ones_like(le), torch.sigmoid((tv-le)/GAMMA)) * \
         torch.where(re>0.98, torch.ones_like(re), torch.sigmoid((re-tv)/GAMMA))
    s = tv-0.5; pos = P["means"] + P["v1"]*s + P["v2"]*s*s + P["v3"]*s**3
    al = (torch.sigmoid(P["opac"])*ta).clamp(1e-4, 1-1e-4)
    col, alpha, _ = rasterization(pos, F.normalize(P["quats"],dim=-1), torch.exp(P["scales"]).clamp(max=0.06), al,
                                  torch.cat([P["sh0"],P["shN"]],1), vms[CAM][None], Ks[CAM][None], Wr, Hr,
                                  sh_degree=3, render_mode="RGB")
    return col[0].clamp(0,1), alpha[0].clamp(0,1)         # (H,W,3), (H,W,1)

# clean static plate: per-pixel mean of GT over frames where pixel is background (mask==0)
bg_sum = torch.zeros(Hr, Wr, 3, device=dev); bg_cnt = torch.zeros(Hr, Wr, 1, device=dev)
for f in frames:
    g = torch.tensor(np.asarray(Image.open(f"{SD}/{f}/{CAM:02d}.png").convert("RGB").resize((Wr,Hr)))/255., dtype=torch.float32, device=dev)
    m = torch.tensor((np.asarray(Image.open(f"{SD}/{f}/masks/{CAM:02d}.png").convert("L").resize((Wr,Hr),Image.NEAREST))>127).astype(np.float32), device=dev)[...,None]
    bg = 1 - m
    bg_sum += g * bg; bg_cnt += bg
plate = bg_sum / bg_cnt.clamp(min=1)                       # person-free studio
# fill any never-background pixel (cnt==0) with the global median
if (bg_cnt < 1).any():
    med = plate[bg_cnt[...,0] >= 1].median(0).values
    plate[bg_cnt[...,0] < 1] = med

MASKS = {ti: torch.tensor((np.asarray(Image.open(f"{SD}/{frames[ti]}/masks/{CAM:02d}.png").convert("L").resize((Wr,Hr),Image.NEAREST))>127).astype(np.float32), device=dev)[...,None] for ti in FIDX}
def erode(m, r=1):                                         # trim the loose mask edge (green-screen bleed) by r px
    return -F.max_pool2d(-m.permute(2,0,1)[None], 2*r+1, 1, r)[0].permute(1,2,0)
def composite(P, ti):
    rgb, _ = render(P, ti/(T-1)); m = erode(MASKS[ti], 1)
    return (rgb * m + plate * (1 - m)).clamp(0,1).cpu().numpy()   # clip to silhouette -> no floaters

def tidy(img):                                             # trim the empty foreground floor
    return img[int(img.shape[0]*0.05) : int(img.shape[0] * 0.90)]
def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB").resize((Wr, Hr))) / 255.

# official Deform-3DGS renders: test set is (odd frames) x 32 cams, uid order ti-major
_test_frames = [i for i in range(T) if i % 2 == 1]
_dgs = sorted(glob.glob(f"{DGS}/*.png"))
def dgs_at(ti):
    return load_rgb(_dgs[_test_frames.index(ti) * 32 + CAM])
def gt_at(ti):
    return load_rgb(f"{SD}/{frames[ti]}/{CAM:02d}.png")

ncol = len(FIDX)
rows = [("Deform-3DGS", dgs_at), ("Ours", lambda ti: composite(M_interp, ti)), ("Ground truth", gt_at)]
comp = {(r, c): tidy(fn(ti)) for r, (_, fn) in enumerate(rows) for c, ti in enumerate(FIDX)}
fig, ax = plt.subplots(len(rows), ncol, figsize=(ncol * 2.25, 5.2))
for r, (rlab, _) in enumerate(rows):
    for c, ti in enumerate(FIDX):
        a = ax[r, c]; a.imshow(comp[(r, c)]); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_color("0.80"); sp.set_linewidth(0.6)
        if r == 0: a.set_title(f"$t={ti/(T-1):.2f}$", fontsize=12, pad=5)
    ax[r, 0].set_ylabel(rlab, fontsize=13, labelpad=6)
plt.subplots_adjust(left=0.038, right=0.998, top=0.925, bottom=0.006, wspace=0.028, hspace=0.014)
plt.savefig(f"{ROOT}/paper_vr/picture/qualitative_demo.pdf")
plt.savefig(f"{ROOT}/paper_vr/picture/qualitative_demo.png", dpi=200)
print(f"saved qualitative_demo  (3 rows, cam{CAM}, {ncol} held timestamps)")
