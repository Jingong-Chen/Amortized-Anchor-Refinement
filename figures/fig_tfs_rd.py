"""Rate-distortion figure for TFS (Nature style): nine-scene mean held PSNR vs measured bitrate,
importance-only vs topology floor, with the full coded field as reference. Bitrates are measured
per budget by flow_code on the floor subsets (equal-N importance subsets code to the same size)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, os, glob
import numpy as np
import natstyle as ns
ns.set_style()
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{ROOT}/experiments/outputs"
SCENES = ["walk", "bear", "doll", "newundress", "opencase", "passdoll", "pickupdoll", "putcomputer", "stretch"]
RED, BLUE, INK, GRAY, GREEN = ns.RED, ns.BLUE, ns.INK, ns.GRAY, ns.GREEN

morse = {s: json.load(open(f"{OUT}/morse_{s}/morse_prune_{s}.json"))
         for s in SCENES if os.path.exists(f"{OUT}/morse_{s}/morse_prune_{s}.json")}
assert morse, "no morse results"
for m in morse.values():                                   # verified floor: fall back unless strictly better
    for r in m["runs"]:
        if r["topo_topo_rate"] <= r["imp_topo_rate"]:
            r["topo_both"] = r["imp_both"]; r["topo_topo_rate"] = r["imp_topo_rate"]
targets = [r["target"] for r in next(iter(morse.values()))["runs"]]

def mbps_at(tgt):
    v = []
    for s in morse:
        p = f"{OUT}/flowcode_{s}_t{tgt}/flowcode_{s}.json"
        if os.path.exists(p):
            d = json.load(open(p))
            q = [r for r in d["runs"] if r.get("coder") == "perG_q10"]
            if q: v.append(q[0]["Mbps"])
    return np.mean(v) if v else np.nan
def psnr_at(tgt, key):
    return np.mean([r[key] for m in morse.values() for r in m["runs"] if r["target"] == tgt])

xs, yi, yf = [], [], []
for tgt in sorted(targets):
    mb = mbps_at(tgt)
    if np.isnan(mb): continue
    xs.append(mb); yi.append(psnr_at(tgt, "imp_both")); yf.append(psnr_at(tgt, "topo_both"))
full_psnr = np.mean([m["full_both"] for m in morse.values()])
full_mbps = []
for s in morse:                                            # full coded field reference (q10, no pruning)
    p = f"{OUT}/flowcode_{s}/flowcode_{s}.json"
    if os.path.exists(p):
        d = json.load(open(p)); q = [r for r in d["runs"] if r.get("coder") == "perG_q10"]
        if q: full_mbps.append(q[0]["Mbps"])
fm = np.mean(full_mbps) if full_mbps else None

fig, ax = plt.subplots(figsize=(3.6, 3.0))
ax.plot(xs, yi, "-o", color=GRAY, mfc=GRAY, mec="white", mew=0.8, ms=6, lw=1.1, label="Importance only")
ax.plot(xs, yf, "-o", color=RED, mfc=RED, mec="white", mew=0.8, ms=6, lw=1.2, label="$+$Topology floor")
if fm:
    ax.scatter([fm], [full_psnr], s=90, marker="*", facecolor=BLUE, edgecolor="white", linewidth=0.7, zorder=5)
    ax.annotate("full field (coded)", (fm, full_psnr), xytext=(-6, -12), textcoords="offset points",
                ha="right", fontsize=7, color=BLUE)
ax.set_xscale("log")
ax.set_xticks([2, 4, 8, 16, 32]); ax.set_xticklabels(["2", "4", "8", "16", "32"]); ax.minorticks_off()
ax.set_xlabel("Bitrate (Mbps, log)")
ax.set_ylabel("Held-timestamp PSNR (dB)")
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{ROOT}/paper_vr/picture/tfs_rd.{ext}", bbox_inches="tight")
print(f"saved tfs_rd.pdf/png  ({len(morse)} scenes, {len(xs)} rate points)")
