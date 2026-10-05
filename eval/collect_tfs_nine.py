"""Aggregate the nine-scene TFS pipeline: per-budget mean/std of PSNR + topology preservation
(importance vs floor), package sizes/bitrates at the 4e4/2e4 topo keeps, and TFS runtime.
Prints a table + ready-to-paste macro lines. Partial-safe: aggregates whatever scenes exist."""
import json, glob, os
import numpy as np
ROOT = os.path.join(os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experiments/outputs")
SCENES = ["walk", "bear", "doll", "newundress", "opencase", "passdoll", "pickupdoll", "putcomputer", "stretch"]

morse = {}
for s in SCENES:
    p = f"{ROOT}/morse_{s}/morse_prune_{s}.json"
    if os.path.exists(p): morse[s] = json.load(open(p))
print(f"scenes with morse results: {len(morse)}/{len(SCENES)}  ({', '.join(morse)})")
if not morse: raise SystemExit("nothing to aggregate yet")

targets = [r["target"] for r in next(iter(morse.values()))["runs"]]
def with_fallback(r):
    """Verified floor: keep the floor arm only if preservation strictly improved (see morse_prune)."""
    if r["topo_topo_rate"] > r["imp_topo_rate"]: return r
    q = dict(r); q["topo_both"] = r["imp_both"]; q["topo_topo_rate"] = r["imp_topo_rate"]; q["restored"] = 0
    return q
for m in morse.values():
    m["runs"] = [with_fallback(r) for r in m["runs"]]
print(f"\nfull-field both: mean {np.mean([m['full_both'] for m in morse.values()]):.2f} "
      f"± {np.std([m['full_both'] for m in morse.values()]):.2f}  "
      f"N mean {np.mean([m['N_full'] for m in morse.values()])/1e4:.1f}e4")
print(f"{'Ns':>7} | {'imp PSNR':>14} | {'floor PSNR':>14} | {'imp rate':>12} | {'floor rate':>12} | restored")
rows = {}
for tgt in targets:
    g = lambda k: np.array([r[k] for m in morse.values() for r in m["runs"] if r["target"] == tgt])
    ib, fb, ir, fr, rs = g("imp_both"), g("topo_both"), g("imp_topo_rate"), g("topo_topo_rate"), g("restored")
    fs = np.array([r.get("floor_sec", np.nan) for m in morse.values() for r in m["runs"] if r["target"] == tgt])
    rows[tgt] = (ib, fb, ir, fr)
    print(f"{tgt:>7} | {ib.mean():6.2f}±{ib.std():4.2f} | {fb.mean():6.2f}±{fb.std():4.2f} | "
          f"{ir.mean():5.2f}±{ir.std():4.2f} | {fr.mean():5.2f}±{fr.std():4.2f} | {rs.mean():7.0f}  ({np.nanmean(fs):5.1f}s)")
refsec = np.array([m["runs"][0].get("ref_sec", np.nan) for m in morse.values()])
print(f"reference-diagram cost: {np.nanmean(refsec):.1f}s/scene (one-time)")

# packages
for tgt in (40000, 20000):
    tots, mbps, psnr = [], [], []
    for s in SCENES:
        p = f"{ROOT}/flowcode_{s}_t{tgt}/flowcode_{s}.json"
        if not os.path.exists(p): continue
        d = json.load(open(p))
        q10 = [r for r in d["runs"] if r.get("coder") == "perG_q10"]
        if not q10: continue
        tots.append(q10[0]["total_MB"]); mbps.append(q10[0]["Mbps"]); psnr.append(q10[0]["both"])
    if tots:
        print(f"package t{tgt}: {len(tots)} scenes  {np.mean(tots):.1f}±{np.std(tots):.1f} MB  "
              f"{np.mean(mbps):.1f} Mbps  PSNR {np.mean(psnr):.2f}")

# macro lines (paste into macros.tex once all nine present)
if len(morse) == len(SCENES):
    ib, fb, ir, fr = rows[20000]
    print("\n% ---- paste into macros.tex (TFS nine-scene) ----")
    print(f"\\renewcommand{{\\topogain}}{{{fb.mean()-ib.mean():.1f}}}")
    print(f"\\renewcommand{{\\ratefloor}}{{{fr.mean():.2f}}}")
    print(f"\\renewcommand{{\\rateimp}}{{{ir.mean():.2f}}}")
    for tgt, name in ((40000, "forty"), (20000, "twenty")):
        tots, mbps = [], []
        for s in SCENES:
            p = f"{ROOT}/flowcode_{s}_t{tgt}/flowcode_{s}.json"
            if os.path.exists(p):
                d = json.load(open(p)); q10 = [r for r in d["runs"] if r.get("coder") == "perG_q10"][0]
                tots.append(q10["total_MB"]); mbps.append(q10["Mbps"])
        if tots:
            print(f"\\renewcommand{{\\tfs{name}mb}}{{{np.mean(tots):.1f}}}")
            print(f"\\renewcommand{{\\tfs{name}mbps}}{{{np.mean(mbps):.1f}}}")
