"""Shared paths and helpers for the manual-reference analyses (rasterisation and boundary distance)."""
import json
from pathlib import Path
import numpy as np
from matplotlib.path import Path as MPath
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]          # data/manual_reference
MASKS = ROOT / "masks"                              # written by make_masks.py (not distributed)
RESULTS = ROOT / "results"
MM2 = (7.38e-3) ** 2                                # mm^2 per pixel
# test-set images per functional group (the nine PSSdb groups), for post-stratified (weighted) means
TEST_N = {"Actinopterygii": 57, "Chaetognatha": 99, "Cnidaria_others": 89, "Copepoda": 1300, "Crustacea_others": 1670,
          "Dinophyceae": 14, "Ostracoda": 7, "Other_zooplankton": 103, "Tunicata": 111}


def load_outlines(name):
    """outlines_main.json or outlines_repeat.json; adds key = '<dyb_folder>__<dyb_filename>'."""
    d = json.load(open(ROOT / name))
    for im in d["images"]:
        im["key"] = f'{im["dyb_folder"]}__{im["dyb_filename"]}'
    return d["images"]


def reference_mask(img):
    """A pixel belongs to the reference when its centre lies inside a non-hole polygon and outside every hole
    (pixel (i, j) spans [i, i+1) x [j, j+1))."""
    h, w = img["height"], img["width"]
    yy, xx = np.mgrid[0:h, 0:w]
    centres = np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5])
    m = np.zeros(h * w, bool)
    for p in img["polygons"]:
        if not p["hole"]:
            m |= MPath(p["points"]).contains_points(centres)
    for p in img["polygons"]:
        if p["hole"]:
            m &= ~MPath(p["points"]).contains_points(centres)
    return m.reshape(h, w)


def boundary(m):
    return m & ~ndimage.binary_erosion(m, border_value=0)


def assd(a, b):
    """average symmetric boundary distance (pixels) between two masks"""
    ba, bb = boundary(a), boundary(b)
    if not ba.any() or not bb.any():
        return np.nan
    da = ndimage.distance_transform_edt(~bb)[ba]
    db = ndimage.distance_transform_edt(~ba)[bb]
    return (da.sum() + db.sum()) / (len(da) + len(db))
