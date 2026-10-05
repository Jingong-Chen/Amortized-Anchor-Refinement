"""Pareto figure (Nature style), fresh current-method data (9-frame short clip, mirror+HD).
(a) quality vs refinement time (smooth). (b) quality vs #Gaussians (nearly flat -> few-Gaussian deployable)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import natstyle as ns
ns.set_style()
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RED, INK, GREEN = ns.RED, ns.INK, ns.GREEN

# refinement-length sweep (min_n=50k): (iters k, both)
IT = [6, 12, 24, 48]; IT_B = [21.26, 22.81, 22.86, 24.11]
# capacity sweep (iters=24k): (#G x1e4, both)
NG = [4.6, 7.3, 9.8, 13.0]; NG_B = [24.00, 23.66, 22.86, 22.98]

fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))

a = ax[0]
a.plot(IT, IT_B, "-o", color=RED, mfc=RED, mec="white", mew=0.8, ms=7, lw=1.2)
a.set_xscale("log"); a.set_xticks(IT); a.set_xticklabels([str(v) for v in IT]); a.minorticks_off()
a.set_xlabel("Refinement steps ($\\times10^{3}$)"); a.set_ylabel("Held-timestamp PSNR (dB)")
a.set_ylim(20.5, 25); a.set_title("(a) Smooth in refinement time", fontsize=9)
a.annotate("more steps\n$\\rightarrow$ higher quality", xy=(48, 24.11), xytext=(9, 24.4),
           fontsize=7, color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
ns.panel_label(a, "a")

b = ax[1]
b.axhspan(23.5, 24.2, color=GREEN, alpha=0.10, lw=0)
b.plot(NG, NG_B, "-o", color=RED, mfc=RED, mec="white", mew=0.8, ms=7, lw=1.2)
b.set_xlabel("Number of Gaussians ($\\times10^{4}$)"); b.set_ylabel("Held-timestamp PSNR (dB)")
b.set_ylim(20.5, 25); b.set_xlim(3, 14); b.set_title("(b) Nearly flat in Gaussian count", fontsize=9)
b.annotate("$4.6{\\times}10^4$ Gaussians\nalready near-best", xy=(4.6, 24.0), xytext=(6.5, 21.5),
           fontsize=7, color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
ns.panel_label(b, "b")

plt.tight_layout(w_pad=2.0)
for ext in ("pdf", "png"):
    plt.savefig(f"{ROOT}/paper_vr/picture/pareto.{ext}", bbox_inches="tight")
print("saved pareto.pdf/png (Nature style, current-method 9f data)")
