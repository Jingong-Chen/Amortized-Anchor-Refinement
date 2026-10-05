"""Fig.1 Motivation (3 panels, each a QUANTITATIVE diagnostic, per the paper template):
(a) predicting != optimizing; (b) the false fast-vs-heavy dichotomy; (c) our tunable, cheap front."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.1))

# (a) predict != optimize  (single-scene ceilings, walk)
labs = ["free-gaussian\npredictor\n(DETR)", "grounded\nsingle-forward", "per-scene\noptimization", "teacher"]
vals = [9.6, 16.5, 21.94, 25.12]; cols = ["#c0392b", "#e08a30", "#9060c0", "#2a8a2a"]
ax[0].bar(range(4), vals, color=cols)
for i, v in enumerate(vals): ax[0].text(i, v+0.4, f"{v}", ha="center", fontsize=9)
ax[0].axhspan(16.5, 21.94, color="gray", alpha=.12)
ax[0].text(1.5, 19.0, "gap prediction\ncannot cross", ha="center", fontsize=8.5, style="italic", color="#555")
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(labs, fontsize=7.8)
ax[0].set_ylabel("both = held view $\\times$ time PSNR (dB)"); ax[0].set_ylim(0, 28)
ax[0].set_title("(a)  Predicting $\\neq$ optimizing", fontsize=11)

# (b) the false dichotomy: fast-but-flat vs slow-but-heavy vs ours
pts = [("MoVieS\n(feed-forward)", 0.93, 11.9, 176, "#e08a30"),
       ("RetimeGS\n(per-scene)", 1200, 25.12, 60, "#9060c0"),
       ("Ours\n(hybrid)", 300, 25.27, 5.6, "#1f77b4")]
for lab, t, q, gs, c in pts:
    ax[1].scatter(t, q, s=gs*7, color=c, alpha=.7, edgecolors="k", zorder=3)
    ax[1].annotate(lab, (t, q), textcoords="offset points", xytext=(0, -34 if lab.startswith("Ours") else 12),
                   ha="center", fontsize=8.5, weight="bold")
ax[1].set_xscale("log"); ax[1].set_xlabel("per-scene time (s, log)")
ax[1].set_ylabel("free-view $\\times$ time PSNR (dB)"); ax[1].set_ylim(8, 28); ax[1].set_xlim(0.3, 4000)
ax[1].text(300, 15.5, "bubble = #gaussians\n(5.6 / 60 / 176 ×10k)", fontsize=7.5, ha="center", color="#555")
ax[1].grid(alpha=.25); ax[1].set_title("(b)  The false fast-vs-heavy dichotomy", fontsize=11)

# (c) ours: quality is tunable and insensitive to gaussian budget
N = [5.3, 16.5, 33, 56, 91, 137]; q = [21.87, 22.71, 22.55, 23.03, 22.94, 23.21]
ax[2].plot(N, q, "s-", color="#1f77b4", lw=2.2, ms=7)
ax[2].axhline(25.12, color="green", ls=":", lw=1.3); ax[2].text(137, 25.25, "teacher 25.12", fontsize=8, color="green", ha="right")
ax[2].annotate("$\\times$26 gaussians\n$\\to$ only +1.3 dB", (56, 22.3), fontsize=8.5, ha="center", color="#555", style="italic")
ax[2].set_xlabel("# gaussians ($\\times10^4$, time fixed)"); ax[2].set_ylabel("both PSNR (dB)")
ax[2].set_ylim(20.5, 26); ax[2].grid(alpha=.25)
ax[2].set_title("(c)  Ours: cheap & tunable, cuts to $\\sim$5k GS", fontsize=11)

plt.suptitle("High-quality 4D is not fast-but-flat prediction OR slow-but-heavy optimization, but a cascade "
             "that anchors, completes, and refines — amortizing the optimizer on a consumer GPU.", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.96])
for e in ("pdf", "png"): plt.savefig(f"{ROOT}/figures/fig_motivation.{e}", dpi=140, bbox_inches="tight")
print("saved fig_motivation")
