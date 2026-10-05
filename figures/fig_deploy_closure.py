"""Figure: closing the deployment gap of hybrid amortized 4D (Exp-87/88/89).

Left  — back-side structure quality is a continuous lever (teacher supervision, walk):
        none / random / MIRROR (deployable) / teacher-proxy (oracle).
Right — fully DEPLOYABLE path (real sparse capture as target, walk):
        mirror vs teacher-back vs no-back, + bear cross-scene; refs.
Reads result_*.json written by hybrid_amort.py; Exp-85c/85d curves predate JSON
persistence and are transcribed from EXPERIMENTS.md.
"""
import os
import json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load(scene, fname):
    d = json.load(open(f"{ROOT}/experiments/outputs/hybrid_{scene}/{fname}"))
    return [c["it"] for c in d["curve"]], [c["both"] for c in d["curve"]]

# historical (teacher-sup, walk; EXPERIMENTS.md Exp-85/85c/85d)
H_IT  = [0, 250, 500, 1000, 2000, 4000]
H_TEA = [15.23, 17.59, 18.25, 20.08, 22.61, 24.25]
H_RND = [16.04, 16.94, 17.49, 18.09, 18.29, 18.36]
H_NOB = [16.03, 16.93, 17.48, 18.19, 18.46, 18.71]
CEIL, PSCENE, FF = 25.12, 21.94, 16.7

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.3))

ax = axes[0]
ax.plot(H_IT, H_TEA, "o-", color="#d62728", lw=2, label="teacher-proxy back (oracle, 85d): 24.25")
it, bo = load("walk", "result_feed-forward-init_densify_mirror-back.json")
ax.plot(it, bo, "s-", color="#1f77b4", lw=2.2, label=f"MIRROR back (deployable): {bo[-1]:.2f}")
ax.plot(H_IT, H_RND, "^--", color="#7f7f7f", lw=1.3, label="random back (85c): 18.36")
ax.plot(H_IT, H_NOB, "v--", color="#bcbd22", lw=1.3, label="no back (85): 18.71 (w/ densify: 18.53)")
ax.axhline(CEIL, color="green", ls=":", lw=1.2)
ax.text(60, CEIL + 0.13, f"per-scene teacher {CEIL}", fontsize=7, color="green")
ax.set_title("Back-side structure quality (teacher supervision, walk)", fontsize=10)
ax.set_xlabel("optimization steps")
ax.set_ylabel("both = held view $\\times$ held time FG-PSNR (dB)")
ax.legend(fontsize=7.5, loc="lower right"); ax.grid(alpha=0.3)

ax = axes[1]
it, bo = load("walk", "result_feed-forward-init_densify_mirror-back_gt-sup_12000.json")
ax.plot(it, bo, "s-", color="#1f77b4", lw=2.2, label=f"MIRROR back (FULL DEPLOY): {bo[-1]:.2f}")
it, bo = load("walk", "result_feed-forward-init_densify_teacher-back_gt-sup_4000.json")
ax.plot(it, bo, "o-", color="#d62728", lw=2, label=f"teacher-proxy back: {bo[-1]:.2f}")
it, bo = load("walk", "result_feed-forward-init_densify_gt-sup_4000.json")
ax.plot(it, bo, "v--", color="#bcbd22", lw=1.5, label=f"no back: {bo[-1]:.2f}")
it, bo = load("bear", "result_feed-forward-init_densify_mirror-back_gt-sup_4000.json")
ax.plot(it, bo, "s--", color="#17becf", lw=1.5, label=f"bear FULL DEPLOY: {bo[-1]:.2f}")
ax.axhline(CEIL, color="green", ls=":", lw=1.2)
ax.text(120, CEIL + 0.13, f"per-scene teacher {CEIL}", fontsize=7, color="green")
ax.axhline(PSCENE, color="purple", ls=":", lw=1.2)
ax.text(120, PSCENE + 0.13, f"per-scene free-gaussian {PSCENE}", fontsize=7, color="purple")
ax.axhline(FF, color="gray", ls=":", lw=1.2)
ax.text(120, FF + 0.13, f"pure feed-forward {FF}", fontsize=7, color="gray")
ax.set_title("Deployable: half capture (16 cams $\\times$ 5 t = 80 imgs)", fontsize=10)
ax.set_xlabel("optimization steps")
ax.legend(fontsize=7.5, loc="lower right"); ax.grid(alpha=0.3)

# ---- panel 3: full capture + capacity floor (Exp-92 operating point) ----
def loadv(scene, fname):
    d = json.load(open(f"{ROOT}/experiments/outputs/viz_hybrid_{scene}/{fname}"))
    return [c["it"] for c in d["curve"]], [c["both"] for c in d["curve"]]
ax = axes[2]
it, bo = loadv("walk", "result_feed-forward-init_densify-minn50k_mirror-back_gt-sup-allcam_24000.json")
ax.plot(it, bo, "s-", color="#1f77b4", lw=2.2, label=f"MIRROR back (FULL DEPLOY): {bo[-1]:.2f}")
it, bo = loadv("walk", "result_feed-forward-init_densify-minn50k_teacher-back_gt-sup-allcam_12000.json")
ax.plot(it, bo, "o-", color="#d62728", lw=2, label=f"teacher-proxy back: {bo[-1]:.2f}")
it, bo = loadv("walk", "result_feed-forward-init_densify-minn50k_gt-sup-allcam_12000.json")
ax.plot(it, bo, "v--", color="#bcbd22", lw=1.5, label=f"no back: {bo[-1]:.2f}")
it, bo = loadv("bear", "result_feed-forward-init_densify-minn50k_mirror-back_gt-sup-allcam_rife-dense_24000.json")
ax.plot(it, bo, "s--", color="#17becf", lw=1.5, label=f"bear FULL DEPLOY (+RIFE): {bo[-1]:.2f}")
ax.axhline(CEIL, color="green", ls=":", lw=1.2)
ax.text(300, CEIL + 0.13, f"per-scene teacher {CEIL} (20 min, same 160 imgs)", fontsize=7, color="green")
ax.axhline(PSCENE, color="purple", ls=":", lw=1.2)
ax.text(300, PSCENE + 0.13, f"per-scene free-gaussian {PSCENE}", fontsize=7, color="purple")
ax.axhline(FF, color="gray", ls=":", lw=1.2)
ax.text(300, FF + 0.13, f"pure feed-forward {FF}", fontsize=7, color="gray")
ax.set_title("Full capture + capacity floor (min-N 50k, Exp-92)", fontsize=10)
ax.set_xlabel("optimization steps")
ax.legend(fontsize=7.5, loc="lower right"); ax.grid(alpha=0.3)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{ROOT}/figures/fig_deploy_closure.{ext}", dpi=160, bbox_inches="tight")
print("saved figures/fig_deploy_closure.{pdf,png}")
