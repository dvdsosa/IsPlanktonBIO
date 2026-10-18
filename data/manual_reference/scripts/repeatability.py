"""Intra-annotator repeatability: second outline of 15 images (anonymised R01-R15, separate session) against the
first outline of the same images (paper Table 7, last row). Needs only the two outline files.

  python data/manual_reference/scripts/repeatability.py
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, load_outlines, reference_mask, assd

first = {i["key"]: i for i in load_outlines("outlines_main.json")}
rows = []
for r in load_outlines("outlines_repeat.json"):
    a = first[r["key"]]
    assert (a["width"], a["height"]) == (r["width"], r["height"]), r["repeat_id"]
    m1, m2 = reference_mask(a), reference_mask(r)
    inter, union = (m1 & m2).sum(), (m1 | m2).sum()
    rows.append(dict(repeat=r["repeat_id"], group=r["functional_group"], cls=r["dyb_folder"], iou=inter / union,
                     dice=2 * inter / (m1.sum() + m2.sum()), area_diff_pct=100 * (m2.sum() - m1.sum()) / m1.sum(),
                     assd_px=assd(m1, m2)))
df = pd.DataFrame(rows)
RESULTS.mkdir(exist_ok=True); df.to_csv(RESULTS / "repeatability.csv", index=False)
print(df.round(3).to_string(index=False))
print(f"\nIoU mean {df.iou.mean():.3f}; |area difference| median {df.area_diff_pct.abs().median():.1f}%, "
      f"mean {df.area_diff_pct.abs().mean():.1f}%; ASSD median {df.assd_px.median():.2f} px")
