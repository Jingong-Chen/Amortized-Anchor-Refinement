"""Rebuild Fig.1 motivation over the static studio background (no hard mask edge; consistent
with the demo). Three tiles: from-scratch collapse / our anchor-refine / GT, each with a zoom.
Foreground clipped to the GT silhouette (eroded) and composited over the person-free studio plate."""
import os, glob, sys
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from gsplat import rasterization
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/aar")
from teacher_retime import load_cams
dev = "cuda"; scene = "walk"; res = 512; nf = 33; CAM = 3; TI = 16

def load_model(path):
    d = torch.load(path, map_location=dev)
    return {k: d[k].to(dev) for k in ("means","scales","quats","opac","sh0","shN","mu_t","log_tl","log_tr","v1","v2","v3")}
M_scr = load_model(glob.glob(f"{ROOT}/experiments/outputs/naive_viz/model_from-scratch*.pt")[0])
M_our = load_model(sorted(glob.glob(f"{ROOT}/experiments/outputs/nine_models/{scene}/model_*_96000.pt"))[0])
print(f"from-scratch N={M_scr['means'].shape[0]}  ours N={M_our['means'].shape[0]}")

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
    col,_,_ = rasterization(pos, F.normalize(P["quats"],dim=-1), torch.exp(P["scales"]).clamp(max=0.06), al,
                            torch.cat([P["sh0"],P["shN"]],1), vms[CAM][None], Ks[CAM][None], Wr, Hr, sh_degree=3, render_mode="RGB")
    return col[0].clamp(0,1)

# static plate (person-free bg mean) + GT mask (eroded)
bs = torch.zeros(Hr,Wr,3,device=dev); bc = torch.zeros(Hr,Wr,1,device=dev)
for f in frames:
    g = torch.tensor(np.asarray(Image.open(f"{SD}/{f}/{CAM:02d}.png").convert("RGB").resize((Wr,Hr)))/255., dtype=torch.float32, device=dev)
    mm = torch.tensor((np.asarray(Image.open(f"{SD}/{f}/masks/{CAM:02d}.png").convert("L").resize((Wr,Hr),Image.NEAREST))>127).astype(np.float32), device=dev)[...,None]
    bs += g*(1-mm); bc += (1-mm)
plate = bs/bc.clamp(min=1)
if (bc<1).any(): plate[bc[...,0]<1] = plate[bc[...,0]>=1].median(0).values
tv = TI/(T-1)
gt = torch.tensor(np.asarray(Image.open(f"{SD}/{frames[TI]}/{CAM:02d}.png").convert("RGB").resize((Wr,Hr)))/255., dtype=torch.float32, device=dev)
m = torch.tensor((np.asarray(Image.open(f"{SD}/{frames[TI]}/masks/{CAM:02d}.png").convert("L").resize((Wr,Hr),Image.NEAREST))>127).astype(np.float32), device=dev)[...,None]
m = -F.max_pool2d(-m.permute(2,0,1)[None],3,1,1)[0].permute(1,2,0)
def comp(P): return (render(P,tv)*m + plate*(1-m)).clamp(0,1).cpu().numpy()
tiles = [comp(M_scr), comp(M_our), gt.cpu().numpy()]
labs = ["From scratch (16.4 dB)", "Ours: Anchor–Refine (24.2 dB)", "Ground truth"]

# zoom on the head/torso
ys, xs = np.where(m[...,0].cpu().numpy() > 0.5)
cy, cx = int(ys.min()+0.18*(ys.max()-ys.min())), int((xs.min()+xs.max())/2)
zh, zw = int(0.34*Hr), int(0.34*Hr*0.85)
y0 = max(0,cy-zh//2); x0 = max(0,cx-zw//2)

fig, ax = plt.subplots(2, 3, figsize=(9.6, 5.2))
for c in range(3):
    ax[0,c].imshow(tiles[c]); ax[0,c].set_title(labs[c], fontsize=10)
    ax[1,c].imshow(tiles[c][y0:y0+zh, x0:x0+zw])
    for r in range(2):
        ax[r,c].set_xticks([]); ax[r,c].set_yticks([])
        for sp in ax[r,c].spines.values(): sp.set_visible(False)
ax[1,0].set_ylabel("zoom", fontsize=9)
plt.tight_layout()
plt.savefig(f"{ROOT}/paper_vr/picture/motivation_real.png", dpi=150, bbox_inches="tight")
print("saved motivation_real.png (studio bg)")
