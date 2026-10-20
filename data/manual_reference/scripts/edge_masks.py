"""Deterministic masks of the 120 outlined images as the deployed KV260 computes them, scored against the manual
reference (paper Section 4.3 and Table 2, row Determ./Edge).

The KV260 application (kria/resnet50_cosine_faiss_turbojpeg_IVFPQ.cpp) decodes JPEGs in BGR order (TurboJPEG,
TJPF_BGR) and segments them with segment_and_crop(). This script ports that function line by line and applies it
  - to the image in BGR order (cv2.imread, libjpeg-turbo): the edge masks;
  - to the image in RGB order (PIL): a control that reproduces the PC masks of make_masks.py exactly.
The PC data loader delivers RGB images to the same BGR-to-grayscale conversion, which swaps the red and blue
weights; the KV260 applies the standard weights. kria/tools/edge_mask_dump.cpp runs the deployed code itself on the
board; its masks are identical to the edge masks written here (--kria-masks checks this).

  python data/manual_reference/scripts/edge_masks.py --dataset /path/to/DYB-PlanktonNet \
      [--masks data/manual_reference/masks] [--kria-masks /path/to/edge_mask_dump/out]
"""
import argparse, os, sys
import numpy as np, pandas as pd, cv2
from PIL import Image
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from common import MASKS, RESULTS, TEST_N, load_outlines, reference_mask, assd

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, help="folder with the DYB-PlanktonNet class folders")
ap.add_argument("--masks", default=str(MASKS), help="PC masks from make_masks.py, for the RGB control (optional)")
ap.add_argument("--kria-masks", help="output folder of kria/tools/edge_mask_dump on the board (optional)")
args = ap.parse_args()
rng = np.random.default_rng(2026)


def segment_and_crop(img):
    """Port of segment_and_crop() in the KV260 application; returns (contour-polygon area, filled mask) or None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adj = cv2.convertScaleAbs(gray, alpha=200 / 100.0, beta=(150 - 100) * 2.55)
    _, otsu = cv2.threshold(adj, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    k_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    k_ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ero = cv2.erode(cv2.dilate(otsu, k_cross, iterations=1), k_ell, iterations=1)
    _, binv = cv2.threshold(ero, 100, 255, cv2.THRESH_BINARY_INV)
    bordered = cv2.copyMakeBorder(binv, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    cs, _ = cv2.findContours(bordered, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if len(cs) <= 1:
        return None
    big = int(np.argmax([cv2.contourArea(c) for c in cs]))
    org = max([c for i, c in enumerate(cs) if i != big], key=cv2.contourArea)
    area = float(cv2.contourArea(org))                  # reported area, before the -10 px shift
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [np.maximum(org - 10, 0)], -1, 255, cv2.FILLED)
    return area, mask


MASKS.mkdir(exist_ok=True)
rows, control = [], []
for img in load_outlines("outlines_main.json"):
    stem = os.path.splitext(img["dyb_filename"])[0]
    path = os.path.join(args.dataset, img["dyb_folder"], img["dyb_filename"])
    ref = reference_mask(img); a_ref = int(ref.sum())
    out = segment_and_crop(cv2.imread(path, cv2.IMREAD_COLOR))                      # BGR, as on the KV260
    pc_file = f'{args.masks}/{img["dyb_folder"]}__{stem}__deterministic.png'
    if os.path.exists(pc_file):                                                      # RGB control = PC pipeline
        ctl = segment_and_crop(np.array(Image.open(path).convert("RGB")))
        control.append(ctl is not None and bool(((ctl[1] > 0) == (np.array(Image.open(pc_file)) > 0)).all()))
    if out is None:
        continue
    area, m = out
    cv2.imwrite(str(MASKS / f'{img["dyb_folder"]}__{stem}__deterministic_edge.png'), m)
    if args.kria_masks:
        k = np.array(Image.open(f'{args.kria_masks}/{img["dyb_folder"]}__{stem}.png')) > 0
        assert ((m > 0) == k).all(), f"KV260 mask differs: {img['key']}"
    mb = m > 0
    inter, union = int((mb & ref).sum()), int((mb | ref).sum())
    rows.append(dict(file=img["key"], group=img["functional_group"], ref_px=a_ref, px=int(mb.sum()), area_poly_px=area,
                     iou=inter / union, area_ratio_poly=area / a_ref, area_ratio_mask=mb.sum() / a_ref,
                     assd_px=assd(mb, ref)))
df = pd.DataFrame(rows)
df.to_csv(RESULTS / "edge_per_image_metrics.csv", index=False)
if control:
    print(f"RGB control reproduces the PC masks: {sum(control)}/{len(control)}")
if args.kria_masks:
    print(f"KV260 masks identical to the edge masks written here: {len(df)}/{len(df)}")


def boot_ci(x, groups, B=10000, stat=np.mean):
    idx = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    bs = [stat(np.concatenate([x[rng.choice(i, len(i))] for i in idx])) for _ in range(B)]
    return np.percentile(bs, [2.5, 97.5])


def weighted(sub, col):
    g = sub.groupby("group")[col].mean()
    w = pd.Series(TEST_N).reindex(g.index)
    return float((g * w).sum() / w.sum())


# paired comparisons with the PC deterministic pipeline and with SAM 2 (results of agreement.py / make_masks.py)
err = np.abs(df.area_ratio_poly - 1)
pc_area = pd.read_csv(RESULTS / "deterministic_areas.csv")
pc_area["file"] = pc_area.dyb_folder + "__" + pc_area.dyb_filename
m = df.merge(pc_area[["file", "area_poly_px"]], on="file", suffixes=("", "_pc"))
err_pc = np.abs(m.area_poly_px_pc / m.ref_px - 1)
per_image = pd.read_csv(RESULTS / "per_image_metrics.csv")
lo, hi = boot_ci(df.iou.to_numpy(), df.group.to_numpy())
elo, ehi = boot_ci(err.to_numpy(), df.group.to_numpy(), stat=np.median)
summary = dict(n=len(df), IoU_mean=df.iou.mean(), IoU_ci_lo=lo, IoU_ci_hi=hi, IoU_weighted=weighted(df, "iou"),
               abs_area_err_median_pct=100 * err.median(), abs_area_err_ci_lo=100 * elo, abs_area_err_ci_hi=100 * ehi,
               abs_area_err_mask_median_pct=100 * np.abs(df.area_ratio_mask - 1).median(),
               ASSD_px_median=df.assd_px.median(),
               edge_over_pc_area_median=float(np.median(m.area_poly_px / m.area_poly_px_pc)),
               abs_area_err_pc_median_pct=100 * err_pc.median(),
               p_area_err_vs_pc=stats.wilcoxon(np.abs(m.area_poly_px / m.ref_px - 1), err_pc).pvalue)
pc_iou = per_image[per_image.method == "Deterministic"].set_index("file").iou
summary["p_iou_vs_pc"] = stats.wilcoxon(df.set_index("file").iou, pc_iou.reindex(df.file)).pvalue
for lab, tag in [("SAM 2, mesh points (paper)", "sam2_mesh"), ("SAM 2, bounding box", "sam2_box")]:
    s = per_image[per_image.method == lab].set_index("file").reindex(df.file)
    s_err = np.abs(s.area_ratio.to_numpy() - 1)
    summary[f"abs_area_err_{tag}_median_pct"] = 100 * np.median(s_err)
    summary[f"p_area_err_vs_{tag}"] = stats.wilcoxon(err.to_numpy(), s_err).pvalue
    summary[f"p_iou_vs_{tag}"] = stats.wilcoxon(df.iou.to_numpy(), s.iou.to_numpy()).pvalue
pd.DataFrame([summary]).to_csv(RESULTS / "edge_summary.csv", index=False)
print(pd.Series(summary).round(4).to_string())
