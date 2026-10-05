"""Three-way comparison: RetimeGS (per-scene opt) vs MoVieS (feed-forward) vs Ours (hybrid),
on Stage-Capture, all measured on our consumer GPU (RTX 6000)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

methods = ["RetimeGS\n(per-scene opt)", "MoVieS\n(feed-forward)", "Ours\n(hybrid)"]
col = ["#9060c0", "#e08a30", "#1f77b4"]
# metrics (measured this session unless noted)
gs = [600, 1762, 56]                       # k gaussians
fps = [664, 394, 1058]                      # render fps @512, same 3DGS pipeline
tsec = [1200, 0.93, 150]                    # per-scene time (s); MoVieS=paper A100 0.93s (12s on our GPU)
qual = [25.1, 11.9, 25.5]                   # free-view x time FG-PSNR (both); MoVieS=mono never-seen-view

fig, ax = plt.subplots(1, 4, figsize=(15, 4.3))
y = np.arange(3)

def barpanel(a, vals, title, better, logx=False, notes=None):
    a.barh(y, vals, color=col, height=0.62)
    a.set_yticks(y); a.set_yticklabels(methods, fontsize=9)
    a.invert_yaxis(); a.set_title(title, fontsize=10.5)
    if logx: a.set_xscale("log")
    for i, v in enumerate(vals):
        lab = notes[i] if notes else (f"{v:g}")
        a.text(v*(1.05 if not logx else 1.15), i, lab, va="center", fontsize=8.5)
    a.text(0.98, -0.14, better, transform=a.transAxes, ha="right", fontsize=8, style="italic", color="#555")
    a.grid(axis="x", alpha=.25)

barpanel(ax[0], gs, "Gaussians (thousands)", "↓ fewer is better (VR)", notes=["600k","1.76M","56k ★"])
ax[0].set_xlim(0, 2000)
barpanel(ax[1], fps, "Render fps @512", "↑ higher is better", notes=["664", "394", "1058 ★"])
ax[1].set_xlim(0, 1200)
barpanel(ax[2], tsec, "Per-scene time (s, log)", "↓ lower is better",
         logx=True, notes=["20 min", "0.93s*(A100)", "2.5 min"])
ax[2].set_xlim(0.3, 4000)
barpanel(ax[3], qual, "Free-view × time PSNR", "↑ higher is better",
         notes=["25.1", "11.9†", "25.5 ★"])
ax[3].set_xlim(0, 30)

fig.suptitle("RetimeGS vs MoVieS vs Ours on Stage-Capture (all render-fps on one consumer RTX 6000).  "
             "Ours: fewest gaussians, fastest render, single consumer GPU, quality on par with 20-min per-scene optimization.",
             fontsize=9.6)
fig.text(0.5, 0.005, "*MoVieS feed-forward is 0.93s on an A100 (12s on our RTX 6000); it needs large-cluster pretraining (170k scenes / 26.6M frames). "
         "†MoVieS takes MONOCULAR input → its free-viewpoint is true extrapolation (2.5D wall); ours & RetimeGS fit multi-view capture. Input-view: MoVieS 31.3 / ours 27.2.",
         ha="center", fontsize=7.2, style="italic", color="#666")
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
for ext in ("png", "pdf"): plt.savefig(f"{ROOT}/figures/fig_compare_methods.{ext}", dpi=140, bbox_inches="tight")
print("saved fig_compare_methods.{png,pdf}")
