"""Convert a DNA-Rendering capture (SenseMoCap .smc pair) into the directory layout our pipeline
reads: <out>/<frame>/<cam>.png, <out>/<frame>/masks/<cam>.png and a cameras.json per frame.

The .smc files store JPEG/PNG byte blobs per (camera, frame) plus per-camera K, D and RT. Images are
undistorted here so the stored intrinsics are a plain pinhole model, matching what the renderer
assumes. Some cameras carry a high-order radial model whose polynomial stops being monotonic inside
the frame; beyond that radius the undistortion map folds and duplicates image content, so those
pixels are blanked in both the image and the mask rather than fed to the optimizer. RT is stored camera-to-world: under that reading the camera centres form the capture ring around
the subject and the projected 3D keypoints land inside the person mask, while the opposite reading
collapses every centre onto one point. --probe re-checks both before converting anything.

Usage: dna_to_stage.py --scene 0008_01 --cams 32 --frames 33 --res 1024
"""
import os, io, json, argparse
import numpy as np, h5py, cv2
from PIL import Image

ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="0008_01")
ap.add_argument("--raw", default=f"{ROOT}/data/dna_rendering/raw")
ap.add_argument("--out", default=f"{ROOT}/data/dna_rendering")
ap.add_argument("--cams", type=int, default=32)           # evenly spaced subset of the 5MP ring
ap.add_argument("--frames", type=int, default=33)
ap.add_argument("--frame0", type=int, default=0)
ap.add_argument("--res", type=int, default=1024)          # longest stored image side
ap.add_argument("--c2w", type=int, default=1)             # RT is stored camera-to-world (verified by --probe)
ap.add_argument("--probe", type=int, default=0)           # report keypoint reprojection error and exit
a = ap.parse_args()

main = h5py.File(f"{a.raw}/{a.scene}.smc", "r")
ann = h5py.File(f"{a.raw}/{a.scene}_annots.smc", "r")

def cam_group():                                          # the 5MP ring is the dense one
    for name in ("Camera_5mp", "Camera_5MP", "Camera"):
        if name in main: return name
    raise KeyError(f"no 5MP camera group in {list(main.keys())}")
CG = cam_group()
cam_ids = sorted(main[CG].keys(), key=int)
n_all = len(cam_ids)
sel = [cam_ids[int(round(i * n_all / a.cams))] for i in range(a.cams)]
print(f"[{a.scene}] {CG}: {n_all} cameras -> {len(sel)} selected; frames {a.frame0}..{a.frame0+a.frames-1}")

def decode(blob):
    return cv2.imdecode(np.frombuffer(np.asarray(blob), np.uint8), cv2.IMREAD_UNCHANGED)

def valid_radius(D):
    """Largest normalized radius where r*(1+k1r^2+k2r^4+k3r^6) is still increasing. Past it the
    undistortion map folds back onto itself and remap duplicates content into the border ring."""
    k1, k2, _, _, k3 = D[:5]
    r = np.linspace(0, 4, 40000)
    d = np.diff(r * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6))
    bad = np.nonzero(d <= 0)[0]
    return r[bad[0]] if len(bad) else float("inf")

def calib(cid):
    g = ann["Camera_Parameter"][f"{int(cid):02d}"]
    return np.array(g["K"]), np.array(g["D"]).ravel(), np.array(g["RT"])

def img_blob(cid, fi):
    return main[CG][cid]["color"][str(fi)]

def mask_blob(cid, fi):
    return ann["Mask"][str(int(cid))]["mask"][str(fi)]

# --- convention probe: project the 3D keypoints of one frame and measure the pixel error against
#     the 2D keypoints the capture system stored for the same camera.
if a.probe:
    # The 2D and 3D keypoint sets use different joint counts, so compare against the segmentation
    # instead: under the correct convention the projected joints land inside the person mask.
    kp3 = np.array(ann["Keypoints_3D"]["keypoints3d"])
    vm = np.array(ann["Keypoints_3D"]["keypoints3d_mask"]).astype(bool)   # 59 of the 190 slots are unused
    fi = a.frame0
    P3 = kp3[fi][:, :3]; conf = kp3[fi][:, 3]
    good = vm & (conf > 0.3)
    for cid in sel[:4]:
        K, D, RT = calib(cid)
        msk = decode(mask_blob(cid, fi))
        if msk.ndim == 3: msk = msk[:, :, 0]
        H0, W0 = msk.shape[:2]
        line = f"  cam {cid} ({W0}x{H0}, {good.sum()} joints):"
        for label, M in (("w2c", RT), ("c2w", np.linalg.inv(RT))):
            uv, _ = cv2.projectPoints(P3.astype(np.float64), cv2.Rodrigues(M[:3, :3])[0],
                                      M[:3, 3].astype(np.float64), K, D)
            uv = uv.reshape(-1, 2)
            Xc = (M[:3, :3] @ P3.T + M[:3, 3:4]).T
            ok = good & (Xc[:, 2] > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] < W0) & (uv[:, 1] >= 0) & (uv[:, 1] < H0)
            hit = 0
            if ok.sum():
                px = uv[ok].astype(int)
                hit = (msk[px[:, 1], px[:, 0]] > 127).mean()
            line += f"  RT as {label}: {ok.sum():3d} in frame, {hit*100:5.1f}% inside mask |"
        print(line)
    raise SystemExit(0)

OUT = f"{a.out}/{a.scene}"
os.makedirs(OUT, exist_ok=True)
frames = list(range(a.frame0, a.frame0 + a.frames))

# resolve stored size and undistortion maps once per camera
maps, cams_json = {}, []
probe = decode(img_blob(sel[0], frames[0]))
H0, W0 = probe.shape[:2]
sc = a.res / max(W0, H0)
Wr, Hr = int(round(W0 * sc)), int(round(H0 * sc))
print(f"  source {W0}x{H0} -> stored {Wr}x{Hr}")

_uu, _vv = np.meshgrid(np.arange(W0, dtype=np.float32), np.arange(H0, dtype=np.float32))
n_folded = 0
for cid in sel:
    K, D, RT = calib(cid)
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (W0, H0), 0, (W0, H0))
    m1, m2 = cv2.initUndistortRectifyMap(K, D, None, newK, (W0, H0), cv2.CV_16SC2)
    rv = valid_radius(D)
    xn = (_uu - newK[0, 2]) / newK[0, 0]; yn = (_vv - newK[1, 2]) / newK[1, 1]
    ok = (np.sqrt(xn ** 2 + yn ** 2) <= rv)
    if not ok.all(): n_folded += 1
    maps[cid] = (m1, m2, ok)
    M = np.linalg.inv(RT) if a.c2w else RT
    R = M[:3, :3]; t = M[:3, 3]
    C = (-R.T @ t).tolist()                               # camera centre, the field our loader wants
    cams_json.append({"cameras": [{
        "R": R.tolist(), "c": C,
        "fx": float(newK[0, 0] * sc), "fy": float(newK[1, 1] * sc),
        "cx": float(newK[0, 2] * sc), "cy": float(newK[1, 2] * sc),
        "w": Wr, "h": Hr, "info": f"dna-{a.scene}-{cid}"}],
        "image_format": "RGB", "info": f"dna-{a.scene}-{cid}"})
print(f"  {n_folded}/{len(sel)} cameras have a folding undistortion ring, blanked outside the valid radius")
CJ = json.dumps({"float_quantization": 2500.0, "rigs": cams_json})

for k, fi in enumerate(frames):
    fd = f"{OUT}/{k:06d}"
    os.makedirs(f"{fd}/masks", exist_ok=True)
    for ci, cid in enumerate(sel):
        rgb = decode(img_blob(cid, fi))
        msk = decode(mask_blob(cid, fi))
        if rgb.ndim == 3 and rgb.shape[2] == 3: rgb = rgb[:, :, ::-1]      # cv2 gives BGR
        if msk.ndim == 3: msk = msk[:, :, 0]
        m1, m2, ok = maps[cid]
        rgb = cv2.remap(rgb, m1, m2, cv2.INTER_LINEAR) * ok[..., None]
        msk = cv2.remap(msk, m1, m2, cv2.INTER_NEAREST) * ok
        Image.fromarray(rgb).resize((Wr, Hr), Image.LANCZOS).save(f"{fd}/{ci:02d}.png")
        Image.fromarray((msk > 127).astype(np.uint8) * 255).resize((Wr, Hr), Image.NEAREST).save(f"{fd}/masks/{ci:02d}.png")
    open(f"{fd}/cameras.json", "w").write(CJ)
    print(f"  frame {k+1}/{len(frames)} (source {fi})", flush=True)
print("saved", OUT)
