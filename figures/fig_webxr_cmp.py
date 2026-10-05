"""Compose the client-decode comparison figure: two rows (uncompressed field / our streamed
package) x three viewpoints, from frames captured by shoot_viewer.py through the WebGL client.
Usage: fig_webxr_cmp.py [--crop 0.30] """
import argparse, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VD = f"{ROOT}/viewer_webxr"
ap = argparse.ArgumentParser()
ap.add_argument("--rows", default="full:Uncompressed field;ours:Ours, streamed package")
ap.add_argument("--subs", default="225.6k Gaussians, 17.2 MB;40k Gaussians, 3.1 MB")
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--keepv", type=float, default=1.0)      # keep this bottom fraction of the content box (trims empty headroom)
ap.add_argument("--keeptop", type=float, default=1.0)    # keep this top fraction (focus on the upper body)
ap.add_argument("--out", default=f"{ROOT}/paper_vr/picture/webxr_decode.png")
a = ap.parse_args()

rows = [r.split(":") for r in a.rows.split(";")]
subs = a.subs.split(";")
imgs = [[Image.open(f"{VD}/cap_{tag}_{k}.png").convert("RGB") for k in range(a.n)] for tag, _ in rows]

# common tight crop around the union of non-white content
def bbox(im):
    g = np.asarray(im.convert("L"), np.uint8)
    ys, xs = np.where(g < 245)
    return (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else (0, 0, im.width - 1, im.height - 1)
bs = [bbox(im) for row in imgs for im in row]
x0 = max(0, min(b[0] for b in bs) - 12); y0 = max(0, min(b[1] for b in bs) - 12)
x1 = min(imgs[0][0].width, max(b[2] for b in bs) + 12); y1 = min(imgs[0][0].height, max(b[3] for b in bs) + 12)
if a.keepv < 1.0:
    y0 = int(y1 - (y1 - y0) * a.keepv)
if a.keeptop < 1.0:
    y1 = int(y0 + (y1 - y0) * a.keeptop)
imgs = [[im.crop((x0, y0, x1, y1)) for im in row] for row in imgs]
W, H = imgs[0][0].size
SCALE = 620 / W
W, H = int(W * SCALE), int(H * SCALE)
imgs = [[im.resize((W, H), Image.LANCZOS) for im in row] for row in imgs]

try:
    FB = ImageFont.truetype("/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf", int(W * 0.062))
    FR = ImageFont.truetype("/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf", int(W * 0.054))
except Exception:
    FB = FR = ImageFont.load_default()
LAB = int(W * 0.165)                       # per-row label strip on the left
GAP = 8
R = len(rows)
canvas = Image.new("RGB", (LAB + a.n * W + (a.n - 1) * GAP, R * H + (R - 1) * GAP), (255, 255, 255))
d = ImageDraw.Draw(canvas)
for r, ((tag, name), sub) in enumerate(zip(rows, subs)):
    y = r * (H + GAP)
    for k in range(a.n):
        canvas.paste(imgs[r][k], (LAB + k * (W + GAP), y))
    strip = Image.new("RGB", (H, LAB), (255, 255, 255))     # rotated row label
    sd = ImageDraw.Draw(strip)
    bb = sd.textbbox((0, 0), name, font=FB)
    sd.text(((H - (bb[2] - bb[0])) / 2, LAB * 0.10), name, font=FB, fill=(20, 20, 20))
    bb2 = sd.textbbox((0, 0), sub, font=FR)
    sd.text(((H - (bb2[2] - bb2[0])) / 2, LAB * 0.10 + (bb[3] - bb[1]) + LAB * 0.30), sub, font=FR, fill=(90, 90, 90))
    canvas.paste(strip.rotate(90, expand=True), (0, y))
canvas.save(a.out)
print("saved", a.out, canvas.size)
