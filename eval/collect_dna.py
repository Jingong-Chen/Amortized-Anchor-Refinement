"""Collect the DNA-Rendering cross-dataset results into the LaTeX rows of the appendix table.
Ours comes from the hybrid_amort result/perceptual json pair, the baselines from the shared
evaluator's eval33.json, so every number in the table is produced by the same metric code.
Usage: collect_dna.py [--scenes 0008_01,0019_06,0047_12]
"""
import os
import json, glob, argparse
import numpy as np
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser()
ap.add_argument("--scenes", default="0008_01,0019_06,0047_12")
a = ap.parse_args()
scenes = a.scenes.split(",")

def ours(s):
    r = sorted(glob.glob(f"{ROOT}/experiments/outputs/dna_models/{s}/result_*.json"))
    if not r: return None
    d = json.load(open(r[0]))
    row = {"psnr": d["final"]["both"], "recon": d["final"]["train"],
           "interpA": d["final"]["retiming"], "interpB": d["final"]["novel"],
           "n": d["N_final"] / 1e4, "sec": d["opt_sec"]}
    f = d["final"]                                  # eval_perceptual writes the held-split metrics here
    row["ssim"] = f.get("both_ssim", float("nan"))
    row["lpips"] = f.get("both_lpips", float("nan"))
    row["peak"] = d.get("N_peak"); row["caps"] = d.get("cap_hits")
    return row

def baseline(s, pre):
    f = f"{ROOT}/experiments/outputs/{pre}_{s}/eval33.json"
    try: d = json.load(open(f))
    except FileNotFoundError: return None
    return {"psnr": d["fg_psnr"], "ssim": d["fg_ssim"], "lpips": d["fg_lpips"]}

rows = {"Deformable-3DGS~\\cite{yang2024deformable}": [baseline(s, "dnadgs") for s in scenes],
        "4D-GS~\\cite{wu20244d}": [baseline(s, "dna4dgs") for s in scenes],
        "OURS": [ours(s) for s in scenes]}

def fmt(v, nd=2):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"

# The rows are \input into a tabular, so the last one must not end in \\: that opens an empty
# row and the \bottomrule that follows lands inside it, which LaTeX reports as a misplaced \noalign.
out = []
for name, vals in rows.items():
    got = [v for v in vals if v]
    psnr = " & ".join(fmt(v["psnr"]) if v else "--" for v in vals)
    mean = fmt(np.mean([v["psnr"] for v in got])) if got else "--"
    ss = fmt(np.mean([v["ssim"] for v in got]), 3) if got else "--"
    lp = fmt(np.mean([v["lpips"] for v in got]), 3) if got else "--"
    label = "\\textbf{\\textcolor{linkblue}{Ours}}" if name == "OURS" else name
    cells = f"{psnr} & {mean} & {ss} & {lp}"
    if name == "OURS":
        cells = " & ".join("\\textbf{\\textcolor{linkblue}{%s}}" % c.strip() for c in cells.split("&"))
    out.append(f"{label} & {cells}")

print("\\\\\n".join(out))

o = [v for v in rows["OURS"] if v]
if o:
    print("%% ours: #G mean %.1f x1e4 (peak %s), cap hits %s, fit mean %.1f min"
          % (np.mean([v["n"] for v in o]), [v["peak"] for v in o], [v["caps"] for v in o],
             np.mean([v["sec"] for v in o]) / 60))
    print("%% ours per-scene #G x1e4: %s" % ["%.1f" % v["n"] for v in o])
