"""Architecture data-flow diagram for the method doc (self-contained matplotlib)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fig, ax = plt.subplots(figsize=(13.5, 7.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 76); ax.axis("off")

def box(x, y, w, h, title, lines, fc, ec, tc="black", fs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.4",
                                fc=fc, ec=ec, lw=1.5))
    ax.text(x + w/2, y + h - 2.7, title, ha="center", va="top", fontsize=fs+1.1, weight="bold", color=tc)
    ax.text(x + w/2, y + h - 6.2, "\n".join(lines), ha="center", va="top", fontsize=fs, color=tc, family="monospace")

def arrow(x1, y1, x2, y2, txt="", col="#333", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>,head_width=3.2,head_length=5",
                                 lw=1.6, color=col, ls=ls, mutation_scale=1))
    if txt: ax.text((x1+x2)/2, (y1+y2)/2 + 1.4, txt, ha="center", fontsize=7.2, color=col, style="italic")

BLU, GRN, ORA, PUR, GRY, TEA = "#dbe9fb", "#dcf0dc", "#fde8d0", "#ece0f5", "#eeeeee", "#d5f2f2"

# --- Stage A: input + backbone + decoder (top row) ---
box(1, 58, 15, 15, "INPUT 1 (feed-fwd)", ["cam0 MONOCULAR", "9 frames RGB", "[9,3,Hin,Win]", "Hin,Win<=256", "-> front init only"], GRY, "#888")
box(20, 58, 17, 15, "VGGT-Omega 1B", ["FROZEN encoder", "pose-free geom.", "alt. frame/global", "attention", "-> patch tokens"], BLU, "#4a86c8")
box(41, 58, 17, 15, "patch tokens", ["fin[0,:,ps:,:]", "[9, Hin/16 *", "  Win/16, 2048]", "grid_sample", "-> [9, 64*88, 2048]"], BLU, "#4a86c8")
box(62, 58, 19, 15, "GroundedDecoderK", ["mlp(2048,256,46):", "Lin2048->256 GELU", "Lin256->256  GELU", "Lin256->46", "reshape [N,K=2,23]"], GRN, "#4aa84a")
box(84, 58, 15, 15, "attr. heads", ["depth,color,op,", "scale,quat,", "v1,v2,v3,", "tau_l,tau_r", "(PDIM=23 x K)"], GRN, "#4aa84a")
arrow(16, 65.5, 20, 65.5); arrow(37, 65.5, 41, 65.5); arrow(58, 65.5, 62, 65.5); arrow(81, 65.5, 84, 65.5)

# --- ray grounding + mirror (middle) ---
box(62, 38, 19, 15, "ray grounding", ["depth->softplus>0", "layer-accumulate K", "xyz = ro + d*rd", "(cam0 pixel rays)", "-> ~101k front G"], ORA, "#e08a30")
box(84, 38, 15, 15, "MIRROR back", ["person G (mask)", "reflect thru", "centroid plane", "_|_ cam0 view", "+40k back G"], TEA, "#3ab0b0")
arrow(91.5, 58, 91.5, 53, "", "#3ab0b0"); arrow(84, 45.5, 81, 45.5, "merge", "#3ab0b0")
arrow(74, 58, 72, 53, "activate")

# --- to free params ---
box(40, 38, 18, 15, "free 4D field", ["means,scales,quats", "opac,sh0[N,1,3],", "shN[N,15,3],mu_t,", "log_tl/tr,v1,v2,v3", "N~141k -> ~56k"], PUR, "#9060c0")
arrow(62, 45.5, 58, 45.5, "to nn.Param")

# --- INPUT 2: the multi-view supervision (feeds the optimizer, NOT the feed-forward net) ---
box(1, 1, 21, 12.5, "INPUT 2 (supervision)", ["32-cam MULTI-VIEW", "capture, 5 train t", "[32,5,Hr,Wr,3]", "+ FG masks = 160 imgs"], "#f6e6e6", "#c0392b", fs=8.0)
arrow(11.5, 13.5, 14, 17, "real target", "#c0392b")

# --- Stage D: optimization (bottom) ---
box(6, 17, 22, 15, "few-step optimize", ["Adam, per-G LRs", "fit to INPUT 2", "(person on black,", "32cam x 5t=160)", "FG-weighted MSE", "grad-clip 1.0"], PUR, "#9060c0")
box(31, 17, 21, 15, "densify + FLOOR", ["every 300 it:", "clone hi-grad,", "prune op<0.005,", "BUT keep top-k", "min_N = 50000", "24k iters ~2.5min"], PUR, "#9060c0")
box(55, 17, 20, 15, "(opt.) RIFE pseudo", ["interp TRAIN frames", "matte on black", "midpts=held t", "zero leakage", "-> +time superv."], TEA, "#3ab0b0")
arrow(45, 38, 20, 32, "", "#9060c0"); arrow(28, 24.5, 31, 24.5); arrow(55, 24.5, 52, 24.5, "mix pool", "#3ab0b0")

# --- output ---
box(79, 15, 19, 19, "OUTPUT", ["continuous-time", "4D Gaussian field", "render any (view,t)", "advect cubic +", "short-tail alpha(t)", "walk 24.5 / bear 26.8", "~0.6dB from teacher"], GRN, "#2a8a2a", fs=8.5)
arrow(48, 16.3, 78.5, 15.5, "converged", "#2a8a2a")

ax.text(50, 74.6, "Hybrid Amortized Continuous-Time 4D: input -> layers -> output",
        ha="center", fontsize=12.5, weight="bold")
ax.text(1, 51.4, "Stage A  feed-forward init (frozen backbone + trained decoder, one pass)", fontsize=8.6, style="italic", color="#4a6a8a")
ax.text(1, 34.4, "Stage B/C/D  mirror back  +  few-step fit to real capture  +  capacity floor", fontsize=8.6, style="italic", color="#7050a0")

plt.tight_layout()
for ext in ("pdf", "png"): plt.savefig(f"{ROOT}/figures/fig_architecture.{ext}", dpi=150, bbox_inches="tight")
print("saved fig_architecture.{pdf,png}")
