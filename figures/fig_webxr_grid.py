"""Compose the on-device comparison as a grid with methods in COLUMNS (headers on the figure)
and viewpoints in rows, keeping every panel a full-body view.
Frames come from shoot_viewer.py / the headset capture hook as cap_<tag>_<k>.png.
Usage: fig_webxr_grid.py --cols "d_full:Uncompressed:;d_lg:LightGaussian:16.22 dB;d_ours:Ours:19.40 dB" --views 0,1
"""
import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VD = f"{ROOT}/viewer_webxr"

ap = argparse.ArgumentParser()
ap.add_argument("--cols", required=True)                 # tag:name:sublabel;...
ap.add_argument("--views", default="0,1")                # capture indices used as rows
ap.add_argument("--width", type=int, default=560)        # per-panel width after scaling
ap.add_argument("--pad", type=int, default=14)           # margin around the content box
ap.add_argument("--transpose", type=int, default=0)       # columns become viewpoints, rows become methods
ap.add_argument("--aspect", type=float, default=0.0)     # pad the crop horizontally to this width/height ratio
ap.add_argument("--out", default=f"{ROOT}/paper_vr/picture/webxr_decode.png")
a = ap.parse_args()

cols = [c.split(":") for c in a.cols.split(";")]
views = [int(v) for v in a.views.split(",")]
if a.transpose:
    imgs = [[Image.open(f"{VD}/cap_{tag}_{k}.png").convert("RGB") for k in views] for tag, *_ in cols]
else:
    imgs = [[Image.open(f"{VD}/cap_{tag}_{k}.png").convert("RGB") for tag, *_ in cols] for k in views]

def bbox(im):
    g = np.asarray(im.convert("L"), np.uint8)
    ys, xs = np.where(g < 245)
    return (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else (0, 0, im.width - 1, im.height - 1)

# one crop box per row (same for every method so the comparison is aligned), full body kept
rows = []
for r, row in enumerate(imgs):
    bs = [bbox(im) for im in row]
    x0 = max(0, min(b[0] for b in bs) - a.pad); y0 = max(0, min(b[1] for b in bs) - a.pad)
    x1 = min(row[0].width, max(b[2] for b in bs) + a.pad); y1 = min(row[0].height, max(b[3] for b in bs) + a.pad)
    if a.aspect > 0:                                    # widen the box so full-height subjects still give wide panels
        need = int((y1 - y0) * a.aspect)
        if need > (x1 - x0):
            cx = (x0 + x1) // 2
            x0 = max(0, cx - need // 2); x1 = min(row[0].width, x0 + need); x0 = max(0, x1 - need)
    sc = a.width / (x1 - x0)
    rows.append([im.crop((x0, y0, x1, y1)).resize((a.width, int((y1 - y0) * sc)), Image.LANCZOS) for im in row])

W = a.width; Hs = [r[0].height for r in rows]
try:
    FB = ImageFont.truetype("/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf", int(W * 0.085))
    FR = ImageFont.truetype("/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf", int(W * 0.075))
except Exception:
    FB = FR = ImageFont.load_default()
HDR = int(W * 0.115) + (int(W * 0.105) if any(len(c) > 2 and c[2] for c in cols) else 0)
GAP = 8
NC = len(rows[0]); LABW = int(W * 0.30) if a.transpose else 0
canvas = Image.new("RGB", (LABW + NC * W + (NC - 1) * GAP, HDR + sum(Hs) + GAP * (len(Hs) - 1)), (255, 255, 255))
d = ImageDraw.Draw(canvas)
hdr = [("", "")] * len(rows[0]) if a.transpose else [(c[1], c[2] if len(c) > 2 else "") for c in cols]
for c, (name, sub) in enumerate(hdr):
    x = LABW + c * (W + GAP)
    if not name: continue
    bb = d.textbbox((0, 0), name, font=FB)
    d.text((x + (W - (bb[2] - bb[0])) / 2, int(W * 0.012)), name, font=FB, fill=(20, 20, 20))
    if sub:
        b2 = d.textbbox((0, 0), sub, font=FR)
        d.text((x + (W - (b2[2] - b2[0])) / 2, int(W * 0.012) + (bb[3] - bb[1]) + int(W * 0.045)),
               sub, font=FR, fill=(80, 80, 80))
y = HDR
for r, row in enumerate(rows):
    for c, im in enumerate(row):
        canvas.paste(im, (LABW + c * (W + GAP), y))
    if a.transpose:
        name = cols[r][1]; sub = cols[r][2] if len(cols[r]) > 2 else ""
        strip = Image.new("RGB", (max(row[0].height, LABW * 3), LABW), (255, 255, 255))
        sd = ImageDraw.Draw(strip)
        bb = sd.textbbox((0, 0), name, font=FB)
        sd.text(((strip.width - (bb[2] - bb[0])) / 2, LABW * 0.08), name, font=FB, fill=(20, 20, 20))
        if sub:
            b2 = sd.textbbox((0, 0), sub, font=FR)
            sd.text(((strip.width - (b2[2] - b2[0])) / 2, LABW * 0.08 + (bb[3] - bb[1]) + LABW * 0.22), sub, font=FR, fill=(80, 80, 80))
        rot = strip.rotate(90, expand=True)
        canvas.paste(rot, (0, y + (row[0].height - rot.height) // 2))
    y += row[0].height + GAP
canvas.save(a.out)
print("saved", a.out, canvas.size)
