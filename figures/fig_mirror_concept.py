"""A dead-simple schematic (not real data) explaining what MIRROR back-side does:
top-down slice of a person; cam0 sees only the front surface -> only front gaussians exist;
reflect each front gaussian across the mid-plane -> new back-side gaussians."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Ellipse

fig, ax = plt.subplots(1, 2, figsize=(12, 5.6))
th = np.linspace(-np.pi/2.1, np.pi/2.1, 9)                                       # sample angles on the front arc
rx, ry = 0.55, 0.42                                                             # body cross-section radii

for k, showback in enumerate([False, True]):
    a = ax[k]
    a.add_patch(Ellipse((0, 0), 2*rx, 2*ry, fill=False, ls="-", ec="#999", lw=1.4))  # body cross-section (top-down)
    a.text(0, 0, "body\n(top-down)", ha="center", va="center", fontsize=8, color="#888")
    # cam0 below, looking up (+y)
    a.add_patch(FancyArrowPatch((0, -1.9), (0, -0.75), arrowstyle="-|>", color="green", lw=2.2, mutation_scale=18))
    a.text(0, -2.15, "cam0\n(the only input view)", ha="center", fontsize=9, color="green", weight="bold")
    # front gaussians = blue dots on the near (cam0-facing, y<0) surface
    fx, fy = rx*np.sin(th), -ry*np.cos(th)
    a.scatter(fx, fy, s=130, c="#1f77b4", edgecolors="k", zorder=5)
    a.text(0, -0.62, "front gaussians\n(cam0 can see these)", ha="center", fontsize=9, color="#1f77b4")
    if showback:
        # mirror plane through the centre, ⊥ to cam0's view direction (here a horizontal line)
        a.plot([-1.05, 1.05], [0, 0], "k--", lw=1.5)
        a.text(1.08, 0.02, "mirror plane\n(⊥ cam0 view,\nthru centre)", ha="left", va="center", fontsize=8.5)
        # each front dot reflected to the back (y -> -y): new gaussians
        a.scatter(fx, -fy, s=130, c="#ff7f0e", edgecolors="k", zorder=5)
        for i in range(len(fx)):
            a.add_patch(FancyArrowPatch((fx[i], fy[i]), (fx[i], -fy[i]), arrowstyle="-|>",
                        color="#ff7f0e", lw=1.0, alpha=.55, mutation_scale=9, connectionstyle="arc3,rad=0.25"))
        a.text(0, 0.66, "mirrored back gaussians\n(NEW — copied & flipped)", ha="center", fontsize=9, color="#d15a00")
    a.set_title(("(1) feed-forward: only FRONT gaussians\n(cam0 can't see the back → back is empty)" if not showback
                 else "(2) MIRROR: copy each front gaussian,\nflip it across the plane → back-side gaussians"), fontsize=10.5)
    a.set_xlim(-1.5, 1.7); a.set_ylim(-2.35, 1.1); a.set_aspect("equal"); a.axis("off")

plt.suptitle("What MIRROR back-side does (schematic). Input: positions of the front gaussians. "
             "Output: a new batch of gaussians (GS) placed on the back by mirror symmetry.", fontsize=10.5)
plt.tight_layout()
for ext in ("png", "pdf"): plt.savefig(f"/figures/fig_mirror_concept.{ext}", dpi=130, bbox_inches="tight")
print("saved fig_mirror_concept.{png,pdf}")
