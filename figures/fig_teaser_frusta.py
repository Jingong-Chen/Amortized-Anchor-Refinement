"""Teaser panel: the reconstructed field with the capture rig drawn around it.

The dataset cameras are drawn as small wireframe frusta at their calibrated poses, so the ring the
rig forms is visible. Frusta behind the subject are composited under it and frusta in front are
drawn over it at lower opacity, which keeps the ring continuous without hiding the subject. All
line work is drawn at a higher resolution and downsampled, otherwise thin strokes alias badly.
Usage: fig_teaser_frusta.py --azim 200 --elev 34 --dist 9
"""
import os, sys, math, argparse
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw
from gsplat import rasterization
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/aar")
from teacher_retime import load_cams

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="walk")
ap.add_argument("--ckpt", default=f"{ROOT}/experiments/outputs/clean_walk/model_cleaned_96000.pt")
ap.add_argument("--keep", default="")
ap.add_argument("--azim", type=float, default=200.0)
ap.add_argument("--elev", type=float, default=34.0)
ap.add_argument("--dist", type=float, default=9.0)
ap.add_argument("--fov", type=float, default=50.0)
ap.add_argument("--glyph", type=float, default=0.030)      # drawn camera size as a fraction of its distance to the
                                                          # view camera, so every glyph has the same size on screen
ap.add_argument("--half_angle", type=float, default=30.0) # apex angle of the little pyramid behind the square
ap.add_argument("--ring", type=int, default=1)            # curve through the camera centres
ap.add_argument("--dome", type=int, default=0)            # draw a dome of virtual viewpoints instead of the capture rig
ap.add_argument("--dome_rings", type=int, default=4)
ap.add_argument("--dome_per", type=int, default=14)
ap.add_argument("--dome_radius", type=float, default=2.9)
ap.add_argument("--dome_top", type=float, default=68.0)   # highest elevation on the dome, degrees
ap.add_argument("--color", default="31,111,180")          # frustum line colour (the paper accent blue)
ap.add_argument("--crop_bottom", type=float, default=1.0)  # keep this top fraction of the content box
ap.add_argument("--dome_dy", type=float, default=0.0)     # shift the dome centre downward (world y is down)
ap.add_argument("--maskcull", type=int, default=0)        # keep only gaussians inside the visual hull of the masks
ap.add_argument("--dilate", type=int, default=25)
ap.add_argument("--lw", type=float, default=1.0)          # line width multiplier for the frustum wireframe
ap.add_argument("--ss", type=int, default=3)              # supersampling factor for the line work
ap.add_argument("--t", type=float, default=0.45)
ap.add_argument("--res", type=int, default=1200); ap.add_argument("--n_frames", type=int, default=33)
ap.add_argument("--out", default="/tmp/teaser_frusta.png")
a = ap.parse_args(); dev = "cuda"

ck = torch.load(a.ckpt, map_location=dev)
means, scales, quats = ck["means"], ck["scales"], ck["quats"]
opac, sh0, shN = ck["opac"], ck["sh0"], ck["shN"]
mu_t, ltl, ltr = ck["mu_t"], ck["log_tl"], ck["log_tr"]
v1, v2, v3 = ck["v1"], ck["v2"], ck["v3"]
idx = torch.load(a.keep, map_location=dev) if a.keep else torch.arange(means.shape[0], device=dev)
_s = torch.exp(scales[idx]).max(-1).values
idx = idx[_s < 0.05]                                       # oversized gaussians only

SD = f"{ROOT}/data/sc_mv_all/{a.scene}"
frames = sorted([d for d in os.listdir(SD) if d.isdigit()])[:a.n_frames]
im0 = Image.open(f"{SD}/{frames[0]}/00.png"); W0, H0 = im0.size
sc = 512 / max(W0, H0)
vms, Ks = load_cams(f"{SD}/{frames[0]}/cameras.json", int(round(W0 * sc)), int(round(H0 * sc)), dev)
ncam = len(vms)

if a.maskcull:                                             # the masks say what the subject is; specks above the head
    _fr = frames[::4]                                      # fall outside the silhouette in the views that see them
    _keep = torch.zeros(idx.numel(), dtype=torch.bool, device=dev)
    _kk = a.dilate | 1
    for _f in _fr:
        _tv = frames.index(_f) / (len(frames) - 1); _sft = _tv - 0.5
        _p = means[idx] + v1[idx] * _sft + v2[idx] * _sft ** 2 + v3[idx] * _sft ** 3
        _inside_all = None
        for _c in range(ncam):
            _mp = f"{SD}/{_f}/masks/{_c:02d}.png"
            if not os.path.exists(_mp): continue
            _mk = torch.tensor((np.asarray(Image.open(_mp).convert("L").resize(
                (int(round(W0 * sc)), int(round(H0 * sc))), Image.NEAREST)) > 127).astype(np.float32), device=dev)
            _mk = F.max_pool2d(_mk[None, None], _kk, 1, _kk // 2)[0, 0] > 0.5
            _pc = (vms[_c][:3, :3] @ _p.T).T + vms[_c][:3, 3]
            _z = _pc[:, 2].clamp(min=1e-6)
            _u = (Ks[_c][0, 0] * _pc[:, 0] / _z + Ks[_c][0, 2]).round().long()
            _v = (Ks[_c][1, 1] * _pc[:, 1] / _z + Ks[_c][1, 2]).round().long()
            _ok = (_pc[:, 2] > 0) & (_u >= 0) & (_u < _mk.shape[1]) & (_v >= 0) & (_v < _mk.shape[0])
            _hit = torch.ones_like(_ok)
            _ii = torch.nonzero(_ok).squeeze(1)
            if _ii.numel(): _hit[_ii] = _mk[_v[_ii], _u[_ii]]
            _inside_all = _hit if _inside_all is None else (_inside_all & _hit)
        if _inside_all is not None: _keep |= _inside_all
    print(f"[hull-cull] {int((~_keep).sum())} of {_keep.numel()} outside the visual hull -> dropped")
    idx = idx[_keep]

ctr = means.median(0).values
th, ph = math.radians(a.azim), math.radians(a.elev)
eye = ctr + torch.tensor([a.dist * math.cos(ph) * math.sin(th),
                          -a.dist * math.sin(ph),
                          a.dist * math.cos(ph) * math.cos(th)], device=dev, dtype=torch.float32)
fwd = F.normalize(ctr - eye, dim=0)
world_down = torch.tensor([0.0, 1.0, 0.0], device=dev)
right = F.normalize(torch.cross(world_down, fwd, dim=0), dim=0)
up = torch.cross(fwd, right, dim=0)
if float(up[1]) < 0: right, up = -right, -up
R = torch.stack([right, up, fwd], 0)
view = torch.eye(4, device=dev); view[:3, :3] = R; view[:3, 3] = -R @ eye
Wv, Hv = a.res, int(a.res * 0.80)
f = 0.5 * Wv / math.tan(math.radians(a.fov) / 2)
K = torch.tensor([[f, 0, Wv / 2], [0, f, Hv / 2], [0, 0, 1]], device=dev, dtype=torch.float32)

GAMMA = 0.005
tl = torch.exp(ltl[idx]); tr = torch.exp(ltr[idx]); mt = mu_t[idx]
le = mt - tl; re = mt + tr
ta = torch.where(le < 0.02, torch.ones_like(le), torch.sigmoid((a.t - le) / GAMMA)) * \
     torch.where(re > 0.98, torch.ones_like(re), torch.sigmoid((re - a.t) / GAMMA))
s = a.t - 0.5
pos = means[idx] + v1[idx] * s + v2[idx] * s * s + v3[idx] * s ** 3
al = (torch.sigmoid(opac[idx]) * ta).clamp(1e-4, 1 - 1e-4)
with torch.no_grad():
    col, acc, _ = rasterization(pos, F.normalize(quats[idx], dim=-1), torch.exp(scales[idx]).clamp(max=0.06),
                                al, torch.cat([sh0[idx], shN[idx]], 1), view[None], K[None],
                                Wv, Hv, sh_degree=3, render_mode="RGB")
rgb = col[0].clamp(0, 1); alpha = acc[0].clamp(0, 1)
if alpha.dim() == 2: alpha = alpha[..., None]
subject = Image.fromarray(np.concatenate([(rgb.cpu().numpy() * 255).astype(np.uint8),
                                          (alpha.cpu().numpy() * 255).astype(np.uint8)], -1), "RGBA")
_sa = np.asarray(subject)[..., 3] > 40                     # subject silhouette, to keep the face clear
_sy, _sx = np.where(_sa)
SBOX = (_sx.min(), _sy.min(), _sx.max(), _sy.max()) if len(_sx) else None

SS = max(1, a.ss)
def project(p_w):
    pc = (R @ (p_w - eye).T).T
    z = pc[:, 2].clamp(min=1e-6)
    u = (f * pc[:, 0] / z + Wv / 2) * SS
    v = (f * pc[:, 1] / z + Hv / 2) * SS
    return torch.stack([u, v], -1), pc[:, 2]

z_ctr = float((R @ (ctr - eye))[2])
back = Image.new("RGBA", (Wv * SS, Hv * SS), (0, 0, 0, 0))
front = Image.new("RGBA", (Wv * SS, Hv * SS), (0, 0, 0, 0))
db, df = ImageDraw.Draw(back), ImageDraw.Draw(front)
INK = tuple(int(v) for v in a.color.split(","))
tt = math.tan(math.radians(a.half_angle))

def dome_poses():
    """Virtual viewpoints on a hemisphere around the subject, each looking at its centre."""
    out = []
    for r in range(a.dome_rings):
        el = math.radians(6 + (a.dome_top - 6) * r / max(1, a.dome_rings - 1))
        n = max(4, int(a.dome_per * math.cos(el)))
        for k in range(n):
            az = 2 * math.pi * (k + 0.5 * (r % 2)) / n
            cen = ctr + torch.tensor([a.dome_radius * math.cos(el) * math.sin(az),
                                      -a.dome_radius * math.sin(el) + a.dome_dy,
                                      a.dome_radius * math.cos(el) * math.cos(az)],
                                     device=dev, dtype=torch.float32)
            fz = F.normalize(ctr - cen, dim=0)
            fx = F.normalize(torch.cross(torch.tensor([0.0, 1.0, 0.0], device=dev), fz, dim=0), dim=0)
            fy = torch.cross(fz, fx, dim=0)
            m = torch.eye(4, device=dev); m[:3, 0], m[:3, 1], m[:3, 2] = fx, fy, fz; m[:3, 3] = cen
            out.append(m)
    return out

poses = dome_poses() if a.dome else [torch.linalg.inv(vms[c]) for c in range(ncam)]
centres = []
for c2w in poses:
    cen = c2w[:3, 3]
    dview = float((cen - eye).norm())
    size = a.glyph * dview                                 # constant apparent size
    dep = size * 2.4                                       # a long pyramid reads as a solid, not as a flat card
    sq = torch.tensor([[-size, -size, 0.0], [size, -size, 0.0],
                       [size, size, 0.0], [-size, size, 0.0]], device=dev)
    apex_cam = torch.tensor([[0.0, 0.0, -dep]], device=dev)
    pts_cam = torch.cat([sq, apex_cam], 0)                 # square on the image plane, apex behind the camera
    pts_w = (c2w[:3, :3] @ pts_cam.T).T + cen
    pts, zs = project(torch.cat([pts_w, cen[None]], 0))
    if (zs <= 0).any(): continue
    q = [tuple(x) for x in pts.cpu().numpy()]
    sqp, apex = q[:4], q[4]
    behind = float(zs[-1]) > z_ctr
    if not behind and SBOX is not None:                    # a camera in front must not sit on the subject
        xs_ = [q[i][0] / SS for i in range(4)]; ys_ = [q[i][1] / SS for i in range(4)]
        if not (max(xs_) < SBOX[0] or min(xs_) > SBOX[2] or max(ys_) < SBOX[1] or min(ys_) > SBOX[3]):
            continue
    d = db if behind else df
    A = 210 if behind else 105
    lw = max(1, int(round(SS * a.lw)))
    def wash(f):                                           # f=0 dark face, f=1 light face
        return tuple(min(255, int(c + (255 - c) * (0.55 + 0.42 * f))) for c in INK)
    # side faces, shaded by how much each one turns away from the viewer
    order = []
    for k in range(4):
        p0, p1 = sqp[k], sqp[(k + 1) % 4]
        cx3 = (p0[0] + p1[0] + apex[0]) / 3.0; cy3 = (p0[1] + p1[1] + apex[1]) / 3.0
        area = abs((p1[0] - p0[0]) * (apex[1] - p0[1]) - (apex[0] - p0[0]) * (p1[1] - p0[1])) / 2.0
        order.append((area, k, (p0, p1)))
    order.sort()                                            # draw the most edge-on faces first
    amax = max(o[0] for o in order) or 1.0
    for area, k, (p0, p1) in order:
        d.polygon([p0, p1, apex], fill=wash(area / amax) + (int(A * 0.85),))
    d.polygon(sqp, fill=wash(1.0) + (235 if behind else 120,))     # the image plane closes the solid
    for k in range(4): d.line([sqp[k], sqp[(k + 1) % 4]], fill=INK + (A,), width=lw)
    d.line([sqp[0], sqp[2]], fill=INK + (int(A * 0.8),), width=lw)
    d.line([sqp[1], sqp[3]], fill=INK + (int(A * 0.8),), width=lw)
    for k in range(4): d.line([sqp[k], apex], fill=INK + (A,), width=lw)
    centres.append((q[-1], behind, float(zs[-1])))


if a.ring:                                                 # the rig circle itself, sampled in 3D and projected
    cens = torch.stack([torch.linalg.inv(vms[c])[:3, 3] for c in range(ncam)])
    rad = float((cens - ctr).norm(dim=-1).mean())
    hgt = float((cens - ctr)[:, 1].mean())
    ang = torch.linspace(0, 2 * math.pi, 240, device=dev)
    circle = ctr[None] + torch.stack([rad * torch.sin(ang),
                                      torch.full_like(ang, hgt),
                                      rad * torch.cos(ang)], -1)
    cp, cz = project(circle)
    cp = cp.cpu().numpy(); cz = cz.cpu().numpy()
    seg_b, seg_f = [], []
    for i in range(len(cp) - 1):
        if cz[i] <= 0 or cz[i + 1] <= 0: continue
        (seg_b if cz[i] > z_ctr else seg_f).append((tuple(cp[i]), tuple(cp[i + 1])))
    for d, segs, alpha_ring in ((db, seg_b, 52), (df, seg_f, 30)):
        for s0, s1 in segs:
            d.line([s0, s1], fill=INK + (alpha_ring,), width=max(1, SS // 2))

out = Image.new("RGBA", (Wv * SS, Hv * SS), (255, 255, 255, 255))
out.alpha_composite(back)
out.alpha_composite(subject.resize((Wv * SS, Hv * SS), Image.LANCZOS))
out.alpha_composite(front)
out = out.convert("RGB").resize((Wv, Hv), Image.LANCZOS)

bb = np.asarray(out.convert("L")) < 248
ys, xs = np.where(bb)
if len(xs):
    pad = 16
    y0c, y1c = max(0, ys.min() - pad), min(out.height, ys.max() + pad)
    if a.crop_bottom < 1.0:
        y1c = int(y0c + (y1c - y0c) * a.crop_bottom)
    out = out.crop((max(0, xs.min() - pad), y0c, min(out.width, xs.max() + pad), y1c))
out.save(a.out)
print("saved", a.out, out.size, f"{len(centres)} cameras drawn")
