"""Area error with the area the deterministic pipeline reports (contour polygon, Green's theorem), against the mask
pixel count and every SAM 2 variant; medians and means with stratified-bootstrap CIs, weighted differences
(paper Table 7, area columns). Run make_masks.py and agreement.py first.

  python data/manual_reference/scripts/area_definition.py
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, TEST_N

W = pd.Series(TEST_N) / sum(TEST_N.values())
rng = np.random.default_rng(2026)
df = pd.read_csv(RESULTS / "per_image_metrics.csv")
ds = pd.read_csv(RESULTS / "deterministic_areas.csv")
ds["file"] = ds.dyb_folder + "__" + ds.dyb_filename
det = df[df.method == "Deterministic"].merge(ds[["file", "area_poly_px", "area_pix_px"]], on="file", how="left")
assert det.area_poly_px.notna().all() and (det.px == det.area_pix_px).all(), "mask pixel counts must match"
poly = det.drop(columns=["area_poly_px", "area_pix_px"]).assign(
    method="Deterministic, contour-polygon area", px=det.area_poly_px, area_ratio=det.area_poly_px / det.ref_px)
df = pd.concat([df, poly], ignore_index=True)
df["abs_err"] = (df.area_ratio - 1).abs()
print(f"contour-polygon / mask pixel count: median {np.median(det.area_poly_px / det.px):.4f}")


def wmean(x, col):
    g = x.groupby("g")[col].mean()
    return float((g * W.reindex(g.index)).sum() / W.reindex(g.index).sum())


rows = []
base = df[df.method == "Deterministic, contour-polygon area"].set_index("file")
for lab in df.method.unique():
    o = df[df.method == lab].set_index("file")
    gi = [np.flatnonzero(o.group.to_numpy() == g) for g in o.group.unique()]
    e = o.abs_err.to_numpy()
    bs = np.array([(np.median(s), s.mean()) for s in
                   (e[np.concatenate([rng.choice(ix, len(ix)) for ix in gi])] for _ in range(10000))])
    (med_lo, mean_lo), (med_hi, mean_hi) = np.percentile(bs, [2.5, 97.5], axis=0)
    r = dict(method=lab, n=len(o), abs_err_med_pct=100 * o.abs_err.median(), abs_err_mean_pct=100 * o.abs_err.mean(),
             med_lo=100 * med_lo, med_hi=100 * med_hi, mean_lo=100 * mean_lo, mean_hi=100 * mean_hi,
             area_ratio_med=o.area_ratio.median(), w_abs_err_pct=100 * wmean(o.assign(g=o.group), "abs_err"),
             crust_abs_err_med_pct=100 * o[o.group.isin(["Copepoda", "Crustacea_others"])].abs_err.median())
    if not lab.startswith("Deterministic, contour"):
        c = base.index.intersection(o.index)
        d = pd.DataFrame({"g": base.loc[c, "group"], "derr": base.loc[c, "abs_err"] - o.loc[c, "abs_err"]})
        idx = {g: d.index[d.g == g] for g in d.g.unique()}
        bsd = [wmean(d.loc[np.concatenate([rng.choice(ix, len(ix)) for ix in idx.values()])], "derr") for _ in range(10000)]
        r.update(paired_n=len(c), polygon_minus_other_med_pct=100 * d.derr.median(),
                 wilcoxon_p=stats.wilcoxon(base.loc[c, "abs_err"], o.loc[c, "abs_err"]).pvalue,
                 w_diff_pct=100 * wmean(d, "derr"), w_diff_lo=100 * np.percentile(bsd, 2.5), w_diff_hi=100 * np.percentile(bsd, 97.5))
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv(RESULTS / "area_definition.csv", index=False)
pd.set_option("display.width", 250); print(R.round(4).to_string(index=False))
