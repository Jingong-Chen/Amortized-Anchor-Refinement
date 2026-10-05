"""Evaluate official Deformable-3DGS renders under OUR protocol metrics:
foreground-masked PSNR / SSIM / LPIPS on the held-out (odd) frames, matching
hybrid_amort.py's fg_psnr / fg_perceptual formulas exactly (person-on-black).
Usage: eval_dgs33.py <render_dir e.g. .../dgs33_walk/test/ours_40000> <scene> [nframes] [data_root]"""
import os, sys, json, math, glob, re
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); dev = "cuda"
RD = sys.argv[1]; SCENE = sys.argv[2]; NF = int(sys.argv[3]) if len(sys.argv) > 3 else 33
DATA_ROOT = sys.argv[4] if len(sys.argv) > 4 else "data/sc_mv_all"   # second dataset lives under its own root
SD = f"{ROOT}/{DATA_ROOT}/{SCENE}"
frames = sorted([d for d in os.listdir(SD) if d.isdigit()])[:NF]

# --- SSIM (box window, same as hybrid_amort) ---
def ssim(x, y):
    C1, C2 = 0.01**2, 0.03**2
    mu_x = F.avg_pool2d(x, 11, 1, 5); mu_y = F.avg_pool2d(y, 11, 1, 5)
    sx = F.avg_pool2d(x*x, 11, 1, 5) - mu_x**2; sy = F.avg_pool2d(y*y, 11, 1, 5) - mu_y**2
    sxy = F.avg_pool2d(x*y, 11, 1, 5) - mu_x*mu_y
    return (((2*mu_x*mu_y + C1)*(2*sxy + C2)) / ((mu_x**2 + mu_y**2 + C1)*(sx + sy + C2))).mean()

import lpips as _lp
lp = _lp.LPIPS(net="vgg", verbose=False).to(dev)
for p in lp.parameters(): p.requires_grad_(False)

rdir, gdir = f"{RD}/renders", f"{RD}/gt"
rs = sorted(glob.glob(f"{rdir}/*.png"))
assert rs, f"no renders in {rdir}"
# render name index maps to test-camera order: frames odd x 32 cams (uid order = ti-major)
test_frames = [i for i in range(len(frames)) if i % 2 == 1]
pairs = []  # (render, gt, mask)
for k, rp in enumerate(rs):
    ti = test_frames[k // 32]; ci = k % 32
    mp = f"{SD}/{frames[ti]}/masks/{ci:02d}.png"
    pairs.append((rp, f"{gdir}/{os.path.basename(rp)}", mp))
mses, sss, lpsv = [], [], []
with torch.no_grad():
    for rp, gp, mp in pairs:
        r = torch.tensor(np.asarray(Image.open(rp).convert("RGB"), np.float32)/255., device=dev)
        g = torch.tensor(np.asarray(Image.open(gp).convert("RGB"), np.float32)/255., device=dev)
        H, W = r.shape[:2]
        m = torch.tensor((np.asarray(Image.open(mp).convert("L").resize((W, H), Image.NEAREST)) > 127).astype(np.float32), device=dev)[..., None]
        if m.sum() < 10: continue
        mses.append(((m*(r-g)**2).sum()/(m.sum()*3+1e-6)).item())                      # fg PSNR (same as ours)
        pr = (r*m).permute(2,0,1)[None].clamp(0,1); gt = (g*m).permute(2,0,1)[None].clamp(0,1)
        sss.append(float(ssim(pr, gt)))                                                # fg SSIM person-on-black
        rd_ = F.interpolate(pr, scale_factor=0.5, mode="bilinear", align_corners=False)*2-1
        gl_ = F.interpolate(gt, scale_factor=0.5, mode="bilinear", align_corners=False)*2-1
        lpsv.append(float(lp(rd_, gl_).mean()))                                        # fg LPIPS (0.5x, same as ours)
res = {"scene": SCENE, "n": len(mses),
       "fg_psnr": -10*math.log10(float(np.mean(mses))+1e-12),
       "fg_ssim": float(np.mean(sss)), "fg_lpips": float(np.mean(lpsv))}
json.dump(res, open(f"{os.path.dirname(RD)}/../eval33.json", "w"), indent=1)
print(f"[DGS33 {SCENE}] n={res['n']} fg_psnr={res['fg_psnr']:.2f} fg_ssim={res['fg_ssim']:.3f} fg_lpips={res['fg_lpips']:.3f}")
