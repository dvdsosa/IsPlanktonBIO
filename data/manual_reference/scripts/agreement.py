"""Agreement of every segmentation mask with the manual reference outlines (paper Table 7: IoU with stratified-
bootstrap CIs, weighted IoU, area ratio, boundary distance; paired IoU differences; per-group IoU; group areas
against PSSdb). Run make_masks.py first. Optional: --pssdb <PSSdb Scanner_1a_Size-distribution_v2024-04.csv>.

  python data/manual_reference/scripts/agreement.py [--pssdb path]
"""
import argparse, os, sys
import numpy as np, pandas as pd
from PIL import Image
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from common import MASKS, RESULTS, MM2, TEST_N, load_outlines, reference_mask, assd

ap = argparse.ArgumentParser(); ap.add_argument("--pssdb"); ap.add_argument("--masks", default=str(MASKS))
args = ap.parse_args()
METHODS = {"Deterministic": "deterministic", "SAM 2, mesh points (paper)": "paper_mesh", "SAM 2, bounding box": "box",
           "SAM 2, box + interior point": "box_center", "SAM 2, interior point": "center_point",
           "SAM 2, five mesh points": "mesh5_points", "SAM 2, mesh, largest mask": "paper_mesh_largest"}
rng = np.random.default_rng(2026)


def edges_touched(m):
    return int(m[0, :].any()) + int(m[-1, :].any()) + int(m[:, 0].any()) + int(m[:, -1].any())


rows = []
for img in load_outlines("outlines_main.json"):
    stem = os.path.splitext(img["dyb_filename"])[0]
    ref = reference_mask(img); a_ref = int(ref.sum())
    base = dict(file=img["key"], group=img["functional_group"], cls=img["dyb_folder"], ref_px=a_ref,
                ref_poly_px=img["area_px_polygon"], ref_edges=edges_touched(ref), time_s=img["time_s"])
    for lab, var in METHODS.items():
        f = f'{args.masks}/{img["dyb_folder"]}__{stem}__{var}.png'
        if not os.path.exists(f):
            continue
        m = np.array(Image.open(f)) > 0
        assert m.shape == ref.shape, (f, m.shape, ref.shape)
        inter, union = int((m & ref).sum()), int((m | ref).sum())
        rows.append(dict(base, method=lab, px=int(m.sum()), iou=inter / union, dice=2 * inter / (m.sum() + a_ref),
                         area_ratio=m.sum() / a_ref, assd_px=assd(m, ref), edges=edges_touched(m)))
df = pd.DataFrame(rows)
df["truncated_ref"] = df.ref_edges >= 2
RESULTS.mkdir(exist_ok=True)
df.to_csv(RESULTS / "per_image_metrics.csv", index=False)


def boot_ci(x, groups=None, B=10000, stat=np.mean):
    x = np.asarray(x)
    if groups is None:
        bs = [stat(rng.choice(x, len(x))) for _ in range(B)]
    else:
        idx = [np.flatnonzero(groups == g) for g in np.unique(groups)]
        bs = [stat(np.concatenate([x[rng.choice(i, len(i))] for i in idx])) for _ in range(B)]
    return np.percentile(bs, [2.5, 97.5])


def weighted(sub, col):
    g = sub.groupby("group")[col].mean()
    w = pd.Series(TEST_N).reindex(g.index)
    return float((g * w).sum() / w.sum())


summ = []
for subset, sel in [("all", df), ("excluding truncated", df[~df.truncated_ref])]:
    for lab in METHODS:
        s = sel[sel.method == lab]
        if s.empty:
            continue
        lo, hi = boot_ci(s.iou.to_numpy(), s.group.to_numpy())
        summ.append(dict(subset=subset, method=lab, n=len(s), IoU_mean=s.iou.mean(), IoU_ci_lo=lo, IoU_ci_hi=hi,
                         IoU_median=s.iou.median(), IoU_weighted=weighted(s, "iou"), Dice_mean=s.dice.mean(),
                         area_ratio_median=s.area_ratio.median(),
                         abs_area_err_median_pct=100 * np.median(np.abs(s.area_ratio - 1)),
                         ASSD_px_median=s.assd_px.median()))
S = pd.DataFrame(summ); S.to_csv(RESULTS / "summary.csv", index=False)
pd.set_option("display.width", 250); print(S.round(3).to_string(index=False))

det = df[df.method == "Deterministic"].set_index("file")
pair_rows = []
for lab in METHODS:
    if lab == "Deterministic":
        continue
    o = df[df.method == lab].set_index("file"); common = det.index.intersection(o.index)
    diff = det.loc[common, "iou"] - o.loc[common, "iou"]
    lo, hi = boot_ci(diff.to_numpy(), det.loc[common, "group"].to_numpy())
    pair_rows.append(dict(method=lab, n=len(common), mean_diff=diff.mean(), ci_lo=lo, ci_hi=hi,
                          det_better=int((diff > 0).sum()),
                          wilcoxon_p=stats.wilcoxon(det.loc[common, "iou"], o.loc[common, "iou"]).pvalue))
P = pd.DataFrame(pair_rows); P.to_csv(RESULTS / "paired_iou.csv", index=False)
print("\nPaired IoU differences (deterministic minus SAM 2 variant):\n" + P.round(4).to_string(index=False))

G = (df[df.method.isin(["Deterministic", "SAM 2, mesh points (paper)", "SAM 2, box + interior point"])]
     .pivot_table(index="group", columns="method", values="iou", aggfunc="mean"))
G["n"] = det.groupby("group").size(); G.to_csv(RESULTS / "iou_by_group.csv")

gm = pd.DataFrame({"manual_mm2": det.groupby("group").ref_px.mean() * MM2})
for lab in ["Deterministic", "SAM 2, mesh points (paper)"]:
    gm[lab + "_mm2"] = df[df.method == lab].groupby("group").px.mean() * MM2
if args.pssdb:
    ps = pd.read_csv(args.pssdb)
    ps["area"] = np.pi * (ps.equivalent_circular_diameter_mean / 2) ** 2 / 1e6
    gm["PSSdb_mm2"] = ps.groupby("Taxon").area.mean().reindex(gm.index)
    for col in ["manual_mm2", "Deterministic_mm2", "SAM 2, mesh points (paper)_mm2"]:
        r = stats.linregress(np.log10(gm.PSSdb_mm2), np.log10(gm[col]))
        print(f"log-log fit vs PSSdb, {col:32s}: R2 = {r.rvalue**2:.2f}, slope = {r.slope:.2f}")
gm.to_csv(RESULTS / "group_areas.csv")
