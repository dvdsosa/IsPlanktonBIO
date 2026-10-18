"""Regenerate the segmentation masks of the 120 outlined images (needs the DYB-PlanktonNet images, a CUDA GPU and the
SAM 2.1 Hiera-Large checkpoint; see the repository README). Writes masks/<dyb_folder>__<stem>__<variant>.png and
results/deterministic_areas.csv (contour-polygon area reported by the pipeline and mask pixel count).

Variants: deterministic; SAM 2 with the paper's mesh points (paper_mesh), bounding box (box), box plus the deepest
interior point (box_center), deepest interior point (center_point), five evenly spaced mesh points (mesh5_points), and
the paper's mesh points keeping the largest of the three candidate masks (paper_mesh_largest). Prompts derive from
the shared preprocessing contour; settings are those of the released SAM2Segmenter.

Run from the repository root:
  python data/manual_reference/scripts/make_masks.py --dataset /path/to/DYB-PlanktonNet \
      --checkpoint data/models/weights/sam2/sam2.1_hiera_large.pt
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))            # repository root, for src/
import src.segmenters as sg
import src.image_utils as iu
for mod in (sg, iu):
    for fn in ("print_normal", "print_yellow", "print_red", "print_green"):
        if hasattr(mod, fn): setattr(mod, fn, lambda *a, **k: None)
from src.image_utils import preprocess_image, get_area_proxy_method

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, help="folder with the DYB-PlanktonNet class folders")
ap.add_argument("--checkpoint", default="data/models/weights/sam2/sam2.1_hiera_large.pt")
args = ap.parse_args()
out = ROOT / "masks"; out.mkdir(exist_ok=True)

seg = sg.SAM2Segmenter(model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml", checkpoint=args.checkpoint,
                       prompt_method="mesh", points_density=500)
pred = seg._predictor


def organism_contour(pre):
    b = cv2.copyMakeBorder(pre, 10, 10, 10, 10, borderType=cv2.BORDER_CONSTANT, value=255)
    cs, _ = cv2.findContours(b, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if len(cs) < 2: return None
    a = [cv2.contourArea(c) for c in cs]; i = int(np.argmax(a))
    return max([c for j, c in enumerate(cs) if j != i], key=cv2.contourArea) - 10


areas = []
for r in csv.DictReader(open(ROOT / "manifest.csv")):
    c, fname = r["dyb_folder"], r["dyb_filename"]
    img = np.array(Image.open(Path(args.dataset) / c / fname).convert("RGB"))   # RGB, as the released pipeline
    pre = preprocess_image(img)
    stem = Path(fname).stem
    ok, poly_area, mdet = get_area_proxy_method(img, pre)
    if mdet is not None:
        cv2.imwrite(str(out / f"{c}__{stem}__deterministic.png"), mdet)
        areas.append(dict(dyb_folder=c, dyb_filename=fname, area_poly_px=poly_area, area_pix_px=int((mdet > 0).sum())))
    mesh_pts, center = seg._get_prompt_points(pre)
    cont = organism_contour(pre)
    if cont is None or (not mesh_pts and not center): continue
    x, y, bw, bh = cv2.boundingRect(cont); box = np.array([x, y, x + bw - 1, y + bh - 1], dtype=float)
    dc = sg._get_safe_center(cont + 10, (img.shape[0] + 20, img.shape[1] + 20), 0)
    dc = None if dc is None else (dc[0] - 10, dc[1] - 10)
    pts = list(mesh_pts) if mesh_pts else [center]
    prompts = {"paper_mesh": dict(point_coords=np.array(pts), point_labels=np.ones(len(pts))), "box": dict(box=box)}
    if dc is not None:
        prompts["center_point"] = dict(point_coords=np.array([dc]), point_labels=np.ones(1))
        prompts["box_center"] = dict(point_coords=np.array([dc]), point_labels=np.ones(1), box=box)
    if mesh_pts and len(mesh_pts) > 5:
        idx = np.linspace(0, len(mesh_pts) - 1, 5).round().astype(int)
        prompts["mesh5_points"] = dict(point_coords=np.array([mesh_pts[i] for i in idx]), point_labels=np.ones(5))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred.set_image(img)
        for name, kw in prompts.items():
            masks, scores, _ = pred.predict(multimask_output=True, **kw)
            best = masks[int(np.argmax(scores))] > 0
            cv2.imwrite(str(out / f"{c}__{stem}__{name}.png"), best.astype(np.uint8) * 255)
            if name == "paper_mesh":
                big = masks[int(np.argmax([m.sum() for m in masks]))] > 0
                cv2.imwrite(str(out / f"{c}__{stem}__paper_mesh_largest.png"), big.astype(np.uint8) * 255)
(ROOT / "results").mkdir(exist_ok=True)
with open(ROOT / "results" / "deterministic_areas.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(areas[0])); w.writeheader(); w.writerows(areas)
print(f"masks written to {out}")
