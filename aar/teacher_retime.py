"""TeacherRetimeGS: load a trained RetimeGS model_<scene>.pt and render CLEAN (view x time) images on demand,
to use as dense distillation targets for a cross-scene feed-forward student. Faithfully reproduces the
cr + short-tail render path of retimegs_4d.py (GAMMA=0.005, smax=0.1, Catmull-Rom trajectory)."""
import json, numpy as np, torch, torch.nn.functional as F
from gsplat import rasterization
GAMMA, EPS, SMAX = 0.005, 0.02, 0.1


def load_cams(cam_json, Wr, Hr, dev="cuda"):
    rigs = json.load(open(cam_json))["rigs"]; vms, Ks = [], []
    for r in rigs:
        c = r["cameras"][0]; sx, sy = Wr / c["w"], Hr / c["h"]
        R = np.array(c["R"], np.float32); cen = np.array(c["c"], np.float32)
        E = np.eye(4, dtype=np.float32); E[:3, :3] = R; E[:3, 3] = -R @ cen
        K = np.array([[c["fx"]*sx, 0, c["cx"]*sx], [0, c["fy"]*sy, c["cy"]*sy], [0, 0, 1]], np.float32)
        vms.append(E); Ks.append(K)
    return torch.tensor(np.stack(vms), device=dev), torch.tensor(np.stack(Ks), device=dev)


class TeacherRetimeGS:
    def __init__(self, model_path, dev="cuda"):
        P = torch.load(model_path, map_location=dev); self.P = {k: v.to(dev) for k, v in P.items()}; self.dev = dev

    def temporal_alpha(self, t):
        P = self.P; tl = torch.exp(P["log_tl"]); tr = torch.exp(P["log_tr"]); mu = P["mu_t"]
        le = mu - tl; re = mu + tr
        left = torch.where(le < EPS, torch.ones_like(le), torch.sigmoid((t - le) / GAMMA))
        right = torch.where(re > 1 - EPS, torch.ones_like(re), torch.sigmoid((re - t) / GAMMA))
        return torch.sigmoid(P["opacities"]) * left * right

    def pos_at(self, t):                                              # Catmull-Rom trajectory (cr=1)
        P = self.P; mu = P["means"]; mt = P["mu_t"]
        dti = (torch.exp(P["log_tl"]) + torch.exp(P["log_tr"])).clamp(min=1e-4)[:, None]
        v2, v1, v3 = P["vel"], P["v1"], P["v3"]
        p1 = mu - 0.5 * dti * v2; p2 = mu + 0.5 * dti * v2; p0 = p1 - dti * v1; p3 = p2 + dti * v3
        u = ((t - mt)[:, None] / dti + 0.5).clamp(0, 1)
        return 0.5 * (2*p1 + (-p0+p2)*u + (2*p0-5*p1+4*p2-p3)*u**2 + (-p0+3*p1-3*p2+p3)*u**3)

    @torch.no_grad()
    def render(self, vm, K, t, Wr, Hr):                              # vm (4,4) world2cam, K (3,3), t float -> [H,W,3]
        P = self.P; t = torch.tensor(float(t), device=self.dev)
        pos = self.pos_at(t); quat = F.normalize(P["quats"], dim=-1); scale = torch.exp(P["scales"]).clamp(max=SMAX)
        al = self.temporal_alpha(t).clamp(1e-4, 1 - 1e-4); sh = torch.cat([P["sh0"], P["shN"]], 1)
        col, _, _ = rasterization(pos, quat, scale, al, sh, vm[None].to(self.dev), K[None].to(self.dev),
                                  Wr, Hr, sh_degree=3, render_mode="RGB", packed=False)
        return col[0].clamp(0, 1)


if __name__ == "__main__":                                           # verify: teacher render == GT (should be ~28dB FG-PSNR, the training value)
    import os, sys
    from PIL import Image
    ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); dev = "cuda"
    scene = sys.argv[1] if len(sys.argv) > 1 else "bear"
    SD = f"{ROOT}/data/sc_mv_all/{scene}"; frames = sorted([d for d in os.listdir(SD) if d.isdigit()])[:9]; T = len(frames)
    W0, H0 = Image.open(f"{SD}/{frames[0]}/00.png").size; Wr = 512; Hr = int(round(H0 * Wr / W0))
    vms, Ks = load_cams(f"{SD}/{frames[0]}/cameras.json", Wr, Hr, dev)
    tch = TeacherRetimeGS(f"{ROOT}/experiments/outputs/retime_{scene}/model_{scene}.pt", dev)
    es = []
    for fi in [0, 4, 8]:
        t = fi / (T - 1)
        for c in [0, 8, 16, 24]:
            gt = torch.tensor(np.asarray(Image.open(f"{SD}/{frames[fi]}/{c:02d}.png").convert("RGB").resize((Wr, Hr), Image.BILINEAR), np.float32) / 255, device=dev)
            mp = f"{SD}/{frames[fi]}/masks/{c:02d}.png"
            m = torch.tensor(np.asarray(Image.open(mp).convert("L").resize((Wr, Hr), Image.NEAREST)) > 127, device=dev).float()[..., None] if os.path.exists(mp) else torch.ones_like(gt[..., :1])
            col = tch.render(vms[c], Ks[c], t, Wr, Hr)
            mse = ((m * (col - gt) ** 2).sum() / (m.sum() * 3 + 1e-6)).item()
            es.append(mse)
    p = -10 * np.log10(np.mean(es) + 1e-12)
    print(f"[verify] {scene}: teacher-render vs GT  FG-PSNR = {p:.2f} dB  (should match the ~28 training value -> teacher render is faithful)")
