"""Efficiency figure (Nature style). Our contribution is the red accent.
(a) Quality vs #Gaussians (log): we sit in the low-Gaussian, near-teacher corner.
(b) Quality vs training time (log): far under the teacher, scaling from a 9-min
    short clip to a 42-min full action (both regimes, red)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import natstyle as ns
ns.set_style()
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INK, RED, BLUE, GRAY, GREEN = ns.INK, ns.RED, ns.BLUE, ns.GRAY, ns.GREEN

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))

# ---------------- (a) Quality vs #Gaussians ----------------
a = ax[0]
a.axvspan(1e-3, 40, color=GREEN, alpha=0.08, lw=0)
a.scatter(128, 26.84, s=46, marker="o", facecolor=BLUE, edgecolor="none", zorder=3)
a.scatter(176, 11.9,  s=40, marker="s", facecolor=GRAY, edgecolor="none", zorder=3)
a.annotate("RetimeGS (teacher)", (128, 26.84), xytext=(9, -1), textcoords="offset points", ha="left", va="center", fontsize=7, color=BLUE)
a.annotate("MoVieS", (176, 11.9), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=7, color=GRAY)
a.scatter(22.6, 24.39, s=230, marker="*", facecolor=RED, edgecolor="white", linewidth=0.8, zorder=5)
a.annotate("Ours", (22.6, 24.39), xytext=(0, -17), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", color=RED)
a.annotate("", xy=(22.6, 28.2), xytext=(128, 28.2), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
a.text(53.8, 28.7, r"$5.7\times$ fewer Gaussians", ha="center", fontsize=7.2)
a.text(11, 11.2, "consumer-deployable\n(few Gaussians)", fontsize=6.8, color=GREEN, style="italic", ha="center")
a.set_xscale("log"); a.set_xlim(3, 470); a.set_ylim(10.5, 30.5)
a.set_xlabel(r"Number of Gaussians ($\times10^{4}$)")
a.set_ylabel("Held-timestamp PSNR (dB)")
ns.panel_label(a, "a")

# ---------------- (b) Quality vs training time ----------------
b = ax[1]
b.scatter(95, 26.84, s=46, marker="o", facecolor=BLUE, edgecolor="none", zorder=3)
b.scatter(0.0155, 11.9, s=40, marker="s", facecolor=GRAY, edgecolor="none", zorder=3)
b.plot([9.1, 42], [24.00, 24.39], "--", color=RED, lw=1.0, zorder=4)
b.scatter([9.1, 42], [24.00, 24.39], s=[150, 230], marker="*", facecolor=RED, edgecolor="white", linewidth=0.8, zorder=5)
b.text(20, 25.5, "Ours", color=RED, fontweight="bold", fontsize=8.5, ha="center")
b.annotate("short clip\n(9 frames)", (9.1, 24.00), xytext=(-3, -21), textcoords="offset points", ha="right", fontsize=7, color=RED)
b.annotate("full action\n(33 frames)", (42, 24.39), xytext=(3, -21), textcoords="offset points", ha="left", fontsize=7.5, fontweight="bold", color=RED)
b.annotate("RetimeGS (teacher)", (95, 26.84), xytext=(-9, -1), textcoords="offset points", ha="right", va="center", fontsize=7, color=BLUE)
b.annotate("MoVieS\n(cluster-trained)", (0.0155, 11.9), xytext=(6, 3), textcoords="offset points", ha="left", fontsize=7, color=GRAY)
b.annotate("", xy=(42, 28.2), xytext=(95, 28.2), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
b.text(63, 28.7, r"$2.3\times$ faster", ha="center", fontsize=7.2)
b.set_xscale("log"); b.set_xlim(6e-3, 320); b.set_ylim(9, 30.5)
b.set_xlabel("Training time per scene (min)")
b.set_ylabel("Held-timestamp PSNR (dB)")
ns.panel_label(b, "b")

plt.tight_layout(w_pad=2.2)
for ext in ("pdf", "png"):
    plt.savefig(f"{ROOT}/paper_vr/picture/efficiency.{ext}", bbox_inches="tight")
print("saved efficiency.pdf/png (Nature style, fixed labels)")
