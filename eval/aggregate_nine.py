"""Aggregate the 9-scene run: per-scene table + mean/std for PSNR (both/novel/retiming)
and perceptual (SSIM/LPIPS). Usage: aggregate_nine.py [outputs/nine_quality]"""
import json, glob, os, sys, statistics as st
ROOT = os.environ.get("AAR_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/experiments/outputs/nine_quality"
SCENES = ["walk","bear","doll","newundress","opencase","passdoll","pickupdoll","putcomputer","stretch"]
rows, keys = [], ["both","novel","retiming","both_ssim","both_lpips","novel_ssim","novel_lpips"]
print(f"{'scene':12s} {'both':>6s} {'novel':>6s} {'retim':>6s} {'b_ssim':>7s} {'b_lpips':>8s} {'min':>5s} {'N':>8s}")
agg = {k: [] for k in keys}
for s in SCENES:
    fs = glob.glob(f"{D}/{s}/result_*.json")
    if not fs: print(f"{s:12s}  --- (missing / OOM) ---"); continue
    d = json.load(open(fs[0])); fi = d["final"]
    for k in keys:
        if fi.get(k) is not None: agg[k].append(fi[k])
    print(f"{s:12s} {fi['both']:6.2f} {fi['novel']:6.2f} {fi['retiming']:6.2f} "
          f"{fi.get('both_ssim',float('nan')):7.3f} {fi.get('both_lpips',float('nan')):8.3f} "
          f"{d.get('opt_sec',0)/60:5.0f} {d.get('N_final',0):8d}")
n = len(agg["both"])
print(f"\n=== {n}/9 scenes ===")
def ms(k):
    v = agg[k]
    if not v: return "n/a"
    return f"{st.mean(v):.3f} ± {st.pstdev(v):.3f}" if len(v)>1 else f"{v[0]:.3f}"
print(f"both      : {ms('both')}   (min {min(agg['both']):.2f})" if agg['both'] else "both: n/a")
print(f"novel     : {ms('novel')}")
print(f"retiming  : {ms('retiming')}")
print(f"both SSIM : {ms('both_ssim')}")
print(f"both LPIPS: {ms('both_lpips')}")
print(f"novel SSIM: {ms('novel_ssim')}   novel LPIPS: {ms('novel_lpips')}")
print(f"\n-> macros: \\mainmean {st.mean(agg['both']):.2f}  \\mainmin {min(agg['both']):.2f}" if agg['both'] else "")
