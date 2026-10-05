"""Short cleanup fine-tune of a fitted field: keep the geometry, remove what the training objective
never penalised. The photometric target is black outside the foreground silhouette, so dark Gaussians
placed there cost almost nothing during fitting and appear as haze once a client renders on a
non-black background. Here we (i) penalise rendered coverage outside the silhouette and (ii) bound
splat anisotropy, with densification switched off so the field size cannot change.
Usage: cleanup_finetune.py --scene doll --ckpt <model.pt> --iters 6000 --sil_w 2.0 --aniso_r 20
"""
import os, sys, math, json, glob, time, argparse
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from gsplat import rasterization
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/aar")
from teacher_retime import load_cams

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="doll"); ap.add_argument("--ckpt", default="")
ap.add_argument("--iters", type=int, default=6000)
ap.add_argument("--sil_w", type=float, default=2.0)       # weight on rendered coverage outside the silhouette
ap.add_argument("--aniso_r", type=float, default=20.0)    # allowed scale ratio before the anisotropy penalty starts
ap.add_argument("--aniso_w", type=float, default=0.02)
ap.add_argument("--lr", type=float, default=0.002)
ap.add_argument("--opacity_only", type=int, default=0)    # only fade what covers the background; never touch geometry,
                                                          # so the foreground-masked metric is unaffected by construction
ap.add_argument("--res", type=int, default=512); ap.add_argument("--n_frames", type=int, default=33)
ap.add_argument("--out", default="")
a = ap.parse_args(); dev = "cuda"
ck_path = a.ckpt or sorted(glob.glob(f"{ROOT}/experiments/outputs/nine_models/{a.scene}/model_*_96000.pt"))[0]
ck = torch.load(ck_path, map_location=dev)
TRAIN = ("opac",) if "--opacity_only 1" in " ".join(sys.argv) else ("opac", "scales", "means")
P = {k: ck[k].clone().to(dev).requires_grad_(k in TRAIN) for k in
     ("means", "scales", "quats", "opac", "sh0", "shN", "mu_t", "log_tl", "log_tr", "v1", "v2", "v3")}
N = P["means"].shape[0]
print(f"[cleanup] {os.path.basename(ck_path)}  N={N}")

SD = f"{ROOT}/data/sc_mv_all/{a.scene}"
frames = sorted([d for d in os.listdir(SD) if d.isdigit()])[:a.n_frames]; T = len(frames)
im0 = Image.open(f"{SD}/{frames[0]}/00.png"); W0, H0 = im0.size; sc = a.res / max(W0, H0)
Wr, Hr = int(round(W0 * sc)), int(round(H0 * sc))
vms, Ks = load_cams(f"{SD}/{frames[0]}/cameras.json", Wr, Hr, dev); ncam = len(vms)
train_c = list(range(0, ncam, 2)); held_c = list(range(1, ncam, 2))
train_t = list(range(0, T, 2)); held_t = list(range(1, T, 2))
RGB, MSK = {}, {}
for c in range(ncam):
    for t in range(T):
        RGB[(c, t)] = torch.tensor(np.asarray(Image.open(f"{SD}/{frames[t]}/{c:02d}.png").convert("RGB").resize((Wr, Hr))) / 255., dtype=torch.float32)
        MSK[(c, t)] = torch.tensor((np.asarray(Image.open(f"{SD}/{frames[t]}/masks/{c:02d}.png").convert("L").resize((Wr, Hr), Image.NEAREST)) > 127).astype(np.float32))

GAMMA = 0.005
def render(c, tv, ret_alpha=False):
    tl = torch.exp(P["log_tl"]); tr = torch.exp(P["log_tr"]); mt = P["mu_t"]
    le = mt - tl; re = mt + tr
    ta = torch.where(le < 0.02, torch.ones_like(le), torch.sigmoid((tv - le) / GAMMA)) * \
         torch.where(re > 0.98, torch.ones_like(re), torch.sigmoid((re - tv) / GAMMA))
    s = tv - 0.5
    pos = P["means"] + P["v1"] * s + P["v2"] * s * s + P["v3"] * s ** 3
    al = (torch.sigmoid(P["opac"]) * ta).clamp(1e-4, 1 - 1e-4)
    col, acc, _ = rasterization(pos, F.normalize(P["quats"], dim=-1), torch.exp(P["scales"]).clamp(max=0.06),
                                al, torch.cat([P["sh0"], P["shN"]], 1), vms[c][None], Ks[c][None],
                                Wr, Hr, sh_degree=3, render_mode="RGB")
    return (col[0].clamp(0, 1), acc[0].clamp(0, 1)) if ret_alpha else col[0].clamp(0, 1)

def held_psnr():
    es = []
    with torch.no_grad():
        for c in held_c:
            for t in held_t:
                m = MSK[(c, t)][..., None].to(dev)
                if m.sum() < 10: continue
                es.append((((render(c, t / (T - 1)) - RGB[(c, t)].to(dev)) ** 2 * m).sum() / (m.sum() * 3 + 1e-6)).item())
    return -10 * math.log10(float(np.mean(es)) + 1e-12)

print(f"[cleanup] held PSNR before: {held_psnr():.2f} dB")
groups = [{"params": [P["opac"]], "lr": a.lr * 5}]
if not a.opacity_only:
    groups += [{"params": [P["scales"]], "lr": a.lr}, {"params": [P["means"]], "lr": a.lr * 0.05}]
opt = torch.optim.Adam(groups)
t0 = time.time()
for it in range(a.iters):
    c = train_c[np.random.randint(len(train_c))]; t = train_t[np.random.randint(len(train_t))]
    fg = MSK[(c, t)][..., None].to(dev); tgt = RGB[(c, t)].to(dev) * fg
    pred, acc = render(c, t / (T - 1), ret_alpha=True)
    w = 1 + 10.0 * fg
    l1 = (w * (pred - tgt).abs()).sum() / (w.sum() * 3)
    a_ = acc if acc.dim() == 3 else acc[..., None]
    sil = (a_ * (1 - fg)).sum() / ((1 - fg).sum() + 1e-6)
    if a.opacity_only:
        ani = torch.zeros((), device=dev)
    else:
        lr_ = P["scales"].max(-1).values - P["scales"].min(-1).values
        ani = F.relu(lr_ - math.log(a.aniso_r)).mean()
    loss = l1 + a.sil_w * sil + a.aniso_w * ani
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if (it + 1) % 1500 == 0:
        print(f"[it {it+1:5d}] {time.time()-t0:.0f}s  L1 {l1.item():.4f}  sil {sil.item():.4f}  ani {ani.item():.3f}  held {held_psnr():.2f}", flush=True)
out = a.out or f"{ROOT}/experiments/outputs/clean_{a.scene}"
os.makedirs(out, exist_ok=True)
save = {k: (P[k].detach() if k in P else ck[k]) for k in ck.keys()}
torch.save(save, f"{out}/model_cleaned_96000.pt")
print(f"[cleanup] held PSNR after: {held_psnr():.2f} dB")
print(f"[saved] {out}/model_cleaned_96000.pt")
