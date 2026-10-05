"""Perceptual metrics (SSIM/LPIPS) for a saved 33f checkpoint, matching hybrid_amort's
fg_perceptual exactly: person-on-black, box-window SSIM, LPIPS(vgg) at half res.
Computes the two regimes Table III reports: Interp (held cams x held times) and
Recon (train cams x train times). PSNRs already live in the fit's result json.
Usage: eval_full_ckpt.py --scene walk   (auto-finds nine_models/<scene> ckpt)"""
import os, json, argparse, glob, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from gsplat import rasterization
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/aar")
from teacher_retime import load_cams

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="walk")
ap.add_argument("--res", type=int, default=512); ap.add_argument("--n_frames", type=int, default=33)
a = ap.parse_args(); dev = "cuda"
ck = torch.load(sorted(glob.glob(f"{ROOT}/experiments/outputs/nine_models/{a.scene}/model_*_96000.pt"))[0], map_location=dev)
means, scales, quats = ck["means"].to(dev), ck["scales"].to(dev), ck["quats"].to(dev)
opac, sh0, shN = ck["opac"].to(dev), ck["sh0"].to(dev), ck["shN"].to(dev)
mu_t, log_tl, log_tr = ck["mu_t"].to(dev), ck["log_tl"].to(dev), ck["log_tr"].to(dev)
v1, v2, v3 = ck["v1"].to(dev), ck["v2"].to(dev), ck["v3"].to(dev)

SD = f"{ROOT}/data/sc_mv_all/{a.scene}"
frames = sorted([d for d in os.listdir(SD) if d.isdigit()])[:a.n_frames]; T = len(frames)
im0 = Image.open(f"{SD}/{frames[0]}/00.png"); W0, H0 = im0.size; sc = a.res / max(W0, H0)
Wr, Hr = int(round(W0 * sc)), int(round(H0 * sc))
vms, Ks = load_cams(f"{SD}/{frames[0]}/cameras.json", Wr, Hr, dev); ncam = len(vms)
train_c = list(range(0, ncam, 2)); held_c = list(range(1, ncam, 2))
train_t = list(range(0, T, 2)); held_t = list(range(1, T, 2))

GAMMA = 0.005
def render(ci, tv):
    tl = torch.exp(log_tl); tr = torch.exp(log_tr); le = mu_t - tl; re = mu_t + tr
    ta = torch.where(le < 0.02, torch.ones_like(le), torch.sigmoid((tv - le) / GAMMA)) * \
         torch.where(re > 0.98, torch.ones_like(re), torch.sigmoid((re - tv) / GAMMA))
    s = tv - 0.5
    pos = means + v1 * s + v2 * s * s + v3 * s ** 3
    al = (torch.sigmoid(opac) * ta).clamp(1e-4, 1 - 1e-4)
    col, _, _ = rasterization(pos, F.normalize(quats, dim=-1), torch.exp(scales).clamp(max=0.06), al,
                              torch.cat([sh0, shN], 1), vms[ci][None], Ks[ci][None], Wr, Hr,
                              sh_degree=3, render_mode="RGB")
    return col[0].clamp(0, 1)
def ssim(x, y):
    win = torch.ones(3, 1, 11, 11, device=x.device) / 121.0
    mx = F.conv2d(x, win, padding=5, groups=3); my = F.conv2d(y, win, padding=5, groups=3)
    mx2, my2, mxy = mx * mx, my * my, mx * my
    sx = F.conv2d(x * x, win, padding=5, groups=3) - mx2; sy = F.conv2d(y * y, win, padding=5, groups=3) - my2
    sxy = F.conv2d(x * y, win, padding=5, groups=3) - mxy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mxy + c1) * (2 * sxy + c2)) / ((mx2 + my2 + c1) * (sx + sy + c2))).mean()
import lpips as lpips_mod
lp = lpips_mod.LPIPS(net="vgg").to(dev).eval()
def sweep(cams, ts):
    ss, ls = [], []
    with torch.no_grad():
        for ci in cams:
            for ti in ts:
                gt = torch.tensor(np.asarray(Image.open(f"{SD}/{frames[ti]}/{ci:02d}.png").convert("RGB").resize((Wr, Hr))) / 255., dtype=torch.float32, device=dev)
                m = torch.tensor((np.asarray(Image.open(f"{SD}/{frames[ti]}/masks/{ci:02d}.png").convert("L").resize((Wr, Hr), Image.NEAREST)) > 127).astype(np.float32), device=dev)[..., None]
                if m.sum() < 10: continue
                prc = (render(ci, ti / (T - 1)) * m).permute(2, 0, 1)[None].clamp(0, 1)
                gtc = (gt * m).permute(2, 0, 1)[None].clamp(0, 1)
                ss.append(float(ssim(prc, gtc)))
                rd = F.interpolate(prc, scale_factor=0.5, mode="bilinear", align_corners=False) * 2 - 1
                gl = F.interpolate(gtc, scale_factor=0.5, mode="bilinear", align_corners=False) * 2 - 1
                ls.append(float(lp(rd, gl).mean()))
    return float(np.mean(ss)), float(np.mean(ls))
si, li = sweep(held_c, held_t)      # Interp regime
sr, lr = sweep(train_c, train_t)    # Recon regime
out = {"scene": a.scene, "interp_ssim": si, "interp_lpips": li, "recon_ssim": sr, "recon_lpips": lr}
json.dump(out, open(f"{ROOT}/experiments/outputs/nine_models/{a.scene}/perceptual.json", "w"), indent=1)
print(f"{a.scene}: Interp SSIM {si:.3f} LPIPS {li:.3f} | Recon SSIM {sr:.3f} LPIPS {lr:.3f}")
