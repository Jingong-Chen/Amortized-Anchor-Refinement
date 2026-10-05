"""Rate-distortion figure for the streaming stage: nine-scene mean held PSNR and topology
preservation rate against the coded bitrate, for the two training-free criteria used by current
Gaussian compression systems and for the package our selector transmits.
Numbers come from the same runs as the streaming table.
Usage: fig_tfs_rd2.py [--out picture/tfs_rd.pdf]"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, os, argparse
import numpy as np
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{ROOT}/experiments/outputs"
SCENES = ["walk", "bear", "doll", "newundress", "opencase", "passdoll", "pickupdoll", "putcomputer", "stretch"]
BUDGETS = [80000, 40000, 20000, 10000, 5000]
MBPS = {80000: 33.4, 40000: 16.8, 20000: 8.4, 10000: 4.2, 5000: 2.1}
D = 0.01

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=f"{ROOT}/paper_vr/picture/tfs_rd.pdf")
a = ap.parse_args()

def base_run(scene, base, t):
    p = f"{OUT}/morse_{scene}{base}/morse_prune_{scene}.json"
    if not os.path.exists(p): return None
    r = [x for x in json.load(open(p))["runs"] if x["target"] == t]
    if not r: return None
    r = r[0]
    use = (r["topo_topo_rate"] - r["imp_topo_rate"]) >= D
    return (r["topo_both"] if use else r["imp_both"], r["topo_topo_rate"] if use else r["imp_topo_rate"])

def crit_run(scene, name, t):
    for f in (f"{OUT}/abl_prune_criteria_20000.json", f"{OUT}/abl_prune_criteria_80000-40000-10000-5000.json"):
        if not os.path.exists(f): continue
        sc = json.load(open(f))["scenes"].get(scene, {})
        for key in (name, f"{name}_{t}"):
            if key in sc and sc[key].get("budget", 20000) == t:
                return (sc[key]["both"], sc[key]["topo_rate"])
    return None

sel = json.load(open(f"{OUT}/select_selfref.json")) if os.path.exists(f"{OUT}/select_selfref.json") else {}
def ours(scene, t):
    pick = sel.get(f"{scene}@{t}", {}).get("pick")
    if pick in ("opacity", "mass"): return base_run(scene, "_opacity" if pick == "opacity" else "", t)
    if pick in ("contrib", "lightgs"): return crit_run(scene, pick, t)
    return base_run(scene, "_opacity", t)

series = {"LightGaussian": lambda s, t: crit_run(s, "lightgs", t),
          "RadSplat": lambda s, t: crit_run(s, "contrib", t),
          "Ours": ours}
data = {}
for name, fn in series.items():
    xs, ps, rs = [], [], []
    for t in BUDGETS:
        v = [fn(s, t) for s in SCENES]
        v = [x for x in v if x]
        if not v: continue
        xs.append(MBPS[t]); ps.append(np.mean([x[0] for x in v])); rs.append(np.mean([x[1] for x in v]))
    data[name] = (xs, ps, rs)
    print(f"{name:<14} " + "  ".join(f"{x:.1f}Mbps:{p:.2f}/{r:.2f}" for x, p, r in zip(xs, ps, rs)))

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7})
BLUE, GRAY, INK = "#1f6fb4", "#9a9a9a", "#222222"
fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.55), dpi=300)
style = {"LightGaussian": dict(color=GRAY, marker="s", ls="--", ms=2.6, lw=1.0),
         "RadSplat": dict(color=INK, marker="^", ls=":", ms=2.8, lw=1.0),
         "Ours": dict(color=BLUE, marker="o", ls="-", ms=3.0, lw=1.4)}
for k, (ax, idx, ylab) in enumerate([(axes[0], 1, "PSNR (dB)"), (axes[1], 2, "Topology preserved")]):
    for name, (xs, ps, rs) in data.items():
        y = ps if idx == 1 else rs
        ax.plot(xs, y, label=name, **style[name])
    ax.set_xscale("log"); ax.set_xticks([2.1, 4.2, 8.4, 16.8, 33.4])
    ax.set_xticklabels(["2.1", "4.2", "8.4", "16.8", "33.4"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["2.1", "4.2", "8.4", "16.8", "33.4"]))
    ax.set_xlabel("bitrate (Mbps)"); ax.set_ylabel(ylab)
    ax.grid(alpha=0.25, lw=0.5); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
axes[0].legend(frameon=False, fontsize=6.4, loc="lower right", handlelength=1.6,
               bbox_to_anchor=(1.04, -0.04))   # nudged clear of the lowest curve
fig.tight_layout(pad=0.35)
fig.savefig(a.out, bbox_inches="tight")
fig.savefig(a.out.replace(".pdf", ".png"), bbox_inches="tight")
print("saved", a.out)
