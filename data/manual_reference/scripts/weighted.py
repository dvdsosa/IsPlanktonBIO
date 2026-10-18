"""Post-stratified (test-set weighted) paired differences with stratified-bootstrap CIs, and paired tests on the
absolute area error of the masks. Run agreement.py first.

  python data/manual_reference/scripts/weighted.py
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, TEST_N

df = pd.read_csv(RESULTS / "per_image_metrics.csv")
W = pd.Series(TEST_N) / sum(TEST_N.values())
rng = np.random.default_rng(2026)
df["abs_err"] = (df.area_ratio - 1).abs()
det = df[df.method == "Deterministic"].set_index("file")
out = []
for lab in ["SAM 2, mesh points (paper)", "SAM 2, bounding box", "SAM 2, box + interior point", "SAM 2, five mesh points"]:
    o = df[df.method == lab].set_index("file"); c = det.index.intersection(o.index)
    d = pd.DataFrame({"g": det.loc[c, "group"], "diou": det.loc[c, "iou"] - o.loc[c, "iou"],
                      "derr": det.loc[c, "abs_err"] - o.loc[c, "abs_err"]})
    wmean = lambda x, col: float((x.groupby("g")[col].mean() * W).dropna().sum() / W[x.g.unique()].sum())
    bs_i, bs_e = [], []
    idx = {g: d.index[d.g == g] for g in d.g.unique()}
    for _ in range(10000):
        s = d.loc[np.concatenate([rng.choice(ix, len(ix)) for ix in idx.values()])]
        bs_i.append(wmean(s, "diou")); bs_e.append(wmean(s, "derr"))
    out.append(dict(vs=lab, n=len(c),
        wIoU_diff=wmean(d, "diou"), wIoU_lo=np.percentile(bs_i, 2.5), wIoU_hi=np.percentile(bs_i, 97.5),
        absErr_det_med=100 * det.loc[c, "abs_err"].median(), absErr_other_med=100 * o.loc[c, "abs_err"].median(),
        absErr_wilcoxon_p=stats.wilcoxon(det.loc[c, "abs_err"], o.loc[c, "abs_err"]).pvalue,
        wAbsErr_diff_pct=100 * wmean(d, "derr"), wAbsErr_lo=100 * np.percentile(bs_e, 2.5), wAbsErr_hi=100 * np.percentile(bs_e, 97.5)))
R = pd.DataFrame(out); R.to_csv(RESULTS / "weighted_differences.csv", index=False)
pd.set_option("display.width", 250); print(R.round(4).to_string(index=False))
