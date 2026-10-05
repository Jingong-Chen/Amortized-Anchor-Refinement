"""Measure render fps of the 33-frame field at its actual gaussian count.
The paper's 765 fps is a 9-frame (5.6万GS) number; 33 frames use ~22.6万GS
(4x), so fps must be re-measured. Independent (copies Hybrid4D, no viz_deploy
import that would trigger its main). Usage: measure_fps.py [model.pt]
"""
import os, sys, time
import torch, torch.nn.functional as F
from PIL import Image
from gsplat import rasterization
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/aar")
from teacher_retime import load_cams
dev = "cuda"; RES = 512; GAMMA = 0.005

class Hybrid4D:
    def __init__(self, path):
        P = torch.load(path, map_location=dev); self.P = {k: v.to(dev) for k, v in P.items()}
        self.N = self.P["means"].shape[0]
    def render(self, vm, K, tv, Wr, Hr):
        P = self.P; s = tv - 0.5
        pos = P["means"] + P["v1"]*s + P["v2"]*s*s + P["v3"]*s*s*s
        tl = torch.exp(P["log_tl"]); tr = torch.exp(P["log_tr"]); le = P["mu_t"]-tl; re = P["mu_t"]+tr
        left = torch.where(le < 0.02, torch.ones_like(le), torch.sigmoid((tv-le)/GAMMA))
        right = torch.where(re > 0.98, torch.ones_like(re), torch.sigmoid((re-tv)/GAMMA))
        al = (torch.sigmoid(P["opac"])*left*right).clamp(1e-4, 1-1e-4)
        col, _, _ = rasterization(pos, F.normalize(P["quats"], dim=-1),
                                  torch.exp(P["scales"]).clamp(max=0.06), al,
                                  torch.cat([P["sh0"], P["shN"]], 1), vm[None], K[None],
                                  Wr, Hr, sh_degree=3, render_mode="RGB")
        return col[0]

SD = f"{ROOT}/data/sc_mv_all/walk"
frames = sorted([d for d in os.listdir(SD) if d.isdigit()])[:33]
im0 = Image.open(f"{SD}/{frames[0]}/00.png"); W0, H0 = im0.size
sc = RES / max(W0, H0); Wr, Hr = int(round(W0*sc)), int(round(H0*sc))
vms, Ks = load_cams(f"{SD}/{frames[0]}/cameras.json", Wr, Hr, dev)
ncam = vms.shape[0]

default = f"{ROOT}/experiments/outputs/unified/main_walk/model_feed-forward-init_densify-minn200k_mirror-back_gt-sup-allcam_96000.pt"
m = Hybrid4D(sys.argv[1] if len(sys.argv) > 1 else default)
print(f"N gaussians: {m.N}  @ {Wr}x{Hr}")

for _ in range(20): m.render(vms[0], Ks[0], 0.5, Wr, Hr)   # warmup
torch.cuda.synchronize()
# full 4D render (incl. advection): vary view AND time
N = 500; t0 = time.time()
for i in range(N):
    m.render(vms[i % ncam], Ks[i % ncam], (i % 33) / 32.0, Wr, Hr)
torch.cuda.synchronize()
dt4d = time.time() - t0
# raster-only (fixed t=0.5, advection precomputed once)
torch.cuda.synchronize(); t0 = time.time()
for i in range(N):
    m.render(vms[i % ncam], Ks[i % ncam], 0.5, Wr, Hr)
torch.cuda.synchronize()
dtr = time.time() - t0
print(f"33-frame field, {m.N} gaussians @ {Wr}x{Hr}:")
print(f"  {N/dt4d:.0f} fps (full 4D, view+time varying)")
print(f"  {N/dtr:.0f} fps (fixed-time render)")
print(f"  [9-frame reference was 765/1058 at 5.6万GS]")
