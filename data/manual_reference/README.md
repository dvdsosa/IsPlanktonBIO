# Manual reference outlines

This folder holds the manual reference used to measure segmentation accuracy in the paper
*When Simpler Segmentation Measures Up: Geometric Fidelity and Energy Efficiency in Edge Plankton Biometry*
(Section 3.6, Section 4.3 and Table 7). It contains outlines only: the images belong to DYB-PlanktonNet and are not
redistributed here.

[← Back to the main README](../../README.md)

## Contents

| File | What it is |
|---|---|
| `outlines_main.json` | Outlines of 120 test images (first pass) |
| `outlines_repeat.json` | Second outline of 15 of those images, drawn in a separate session under anonymised names (`repeat_id` R01–R15), for intra-annotator repeatability |
| `manifest.csv` | Functional group, DYB-PlanktonNet class folder and file name of the 120 images |
| `results/` | Per-image metrics and the summary tables behind Table 7 (see below) |
| `scripts/` | Code that regenerates the pipeline masks and every number in `results/` |

## The images

Download DYB-PlanktonNet from IEEE DataPort ([doi:10.21227/875n-f104](https://doi.org/10.21227/875n-f104),
CC BY-NC-ND 4.0). Each outline names its image by `dyb_folder` (the class folder) and `dyb_filename`, which locate
the file in the original dataset. All 120 images belong to the test split of `data/splits/test_files.csv`.

## The sample

We drew 120 of the 2,685 mapped test specimens that both segmentation pipelines validate, with a fixed seed, so that
every image has a mask from each method. Each of the nine PSSdb functional groups received 15 images, sampled by
cycling through its classes. *Dinophyceae* and *Ostracoda* contribute all their 8 and 5 eligible specimens, and
*Copepoda* and Crustacea (others) received 16 each. The allocation is equal rather than proportional, so that every
group can be evaluated; the scripts report unweighted means and means weighted by group frequency in the test set.
Five *Creseis acicula* images in the first draw, a taxon excluded from the analysis, were replaced by other eligible
Other-zooplankton specimens drawn with a second fixed seed.

## Annotation protocol

One author outlined every organism with [MorphoTrace](https://github.com/dvdsosa/morphotrace) v0.2.5 (tag `v0.2.5`;
run it online at https://dvdsosa.github.io/morphotrace/; Zenodo [doi:10.5281/zenodo.23454366](https://doi.org/10.5281/zenodo.23454366)), blind to the
pipeline masks.

- Each outline encloses the whole visible silhouette, including antennae and appendages but excluding setae thinner
  than one pixel. This follows the area definition behind the allometric coefficients of Lehette and
  Hernández-León (2009).
- One polygon per organism, with no holes for internal or transparent regions.
- Short stretches of faint boundary are closed by following the curvature of the visible outline. An image with more
  than a quarter of its boundary indistinguishable would have been flagged as unusable; none was.

## File format

Both JSON files share one structure. The header gives the tool and version, export time, annotator initials,
scale (`um_per_px` = 7.38) and the coordinate convention. Each entry of `images` holds:

| Field | Meaning |
|---|---|
| `dyb_folder`, `dyb_filename` | Image in DYB-PlanktonNet |
| `functional_group` | PSSdb functional group used in the paper |
| `width`, `height` | Image size in pixels |
| `polygons` | List of `{label, hole, points}`; `points` are `[x, y]` vertices |
| `area_px_polygon`, `area_mm2_polygon`, `perimeter_px` | Polygon area and perimeter |
| `time_s` | Annotation time in seconds |
| `status`, `unusable_reason` | All entries are `annotated` |

Coordinates are continuous image-pixel coordinates with the origin at the top-left corner; pixel (i, j) spans
[i, i+1) × [j, j+1). To rasterise an outline, a pixel belongs to the reference when its centre lies inside a non-hole
polygon and outside every hole (`scripts/common.py`, `reference_mask`).

## Reproducing Table 7

Run from the repository root, with the environment of the main README:

```bash
# 1) Masks of every pipeline for the 120 images (GPU and SAM 2.1 checkpoint needed; writes masks/, not tracked)
python data/manual_reference/scripts/make_masks.py --dataset /path/to/DYB-PlanktonNet \
    --checkpoint data/models/weights/sam2/sam2.1_hiera_large.pt
# 2) Agreement with the outlines (IoU, boundary distance, paired tests, group areas; --pssdb is optional)
python data/manual_reference/scripts/agreement.py --pssdb /path/to/Scanner_1a_Size-distribution_v2024-04.csv
# 3) Weighted differences, area error with the reported area, repeatability
python data/manual_reference/scripts/weighted.py
python data/manual_reference/scripts/area_definition.py
python data/manual_reference/scripts/repeatability.py   # needs only the two outline files
```

Step 1 also writes `results/deterministic_areas.csv`: the contour-polygon area that the deterministic pipeline
reports (Green's theorem) and the pixel count of its mask. Steps 2 and 3 regenerate the tables in `results/`, which
are the exact files used in the paper:

| File | Content |
|---|---|
| `per_image_metrics.csv` | IoU, Dice, area ratio and boundary distance of every mask; `file` = `<dyb_folder>__<dyb_filename>` |
| `summary.csv` | Mean IoU with stratified-bootstrap CI, weighted IoU, median area ratio and error, boundary distance, with and without outlines that touch two or more image edges |
| `paired_iou.csv` | Paired IoU differences against the deterministic pipeline (bootstrap CI, Wilcoxon test) |
| `weighted_differences.csv` | Test-set-weighted paired differences in IoU and absolute area error |
| `area_definition.csv` | Area error with the reported contour-polygon area, the mask pixel count and every SAM 2 variant; medians and means with CIs |
| `repeatability.csv` | Second against first outline for the 15 repeated images |
| `iou_by_group.csv`, `group_areas.csv` | Mean IoU and mean areas per functional group |
| `deterministic_areas.csv` | Reported area and mask pixel count of the deterministic pipeline |
| `edge_per_image_metrics.csv`, `edge_summary.csv` | Deterministic masks of the deployed KV260 against the outlines (see below) |

Regenerated masks match the paper's exactly for the deterministic pipeline and for the mesh-point, five-point and largest-mask variants. A few box and interior-point masks can differ between runs because SAM 2 runs in bfloat16 on the GPU, which is not bit-reproducible (16 of 836 masks in our check, on an RTX 3060); the effect on Table 7 is at most 0.002 in mean IoU and 0.3 points in median area error.

Mask variants written by `make_masks.py`: `deterministic`; SAM 2 with the paper's mesh points (`paper_mesh`),
bounding box (`box`), box plus the deepest interior point (`box_center`), deepest interior point (`center_point`),
five evenly spaced mesh points (`mesh5_points`), and the paper's mesh points keeping the largest of the three
candidate masks (`paper_mesh_largest`). All prompts derive from the shared preprocessing contour.

## Deployed KV260 masks

The paper's PC pipeline and the deployed KV260 application segment the same way but read the images differently.
On the PC, the data loader delivers RGB images to a grayscale conversion written for OpenCV's BGR order, so the red
and blue weights are swapped; the KV260 decodes JPEGs in BGR order (TurboJPEG) and applies the standard weights.
Table 2 of the paper therefore also scores the deterministic masks of the KV260 against the outlines (row
Determ./Edge): mean IoU 0.79 and median absolute area error 12.5%, against 0.79 and 11.2% on the PC.

```bash
# 1) On the PC: port of the KV260 segment_and_crop() applied to the images in BGR order
python data/manual_reference/scripts/edge_masks.py --dataset /path/to/DYB-PlanktonNet
# 2) Optional, on the KV260: the deployed code itself (kria/tools/edge_mask_dump.cpp, build line in its header)
./edge_mask_dump list.txt out          # list.txt: the 120 image paths of manifest.csv
python data/manual_reference/scripts/edge_masks.py --dataset /path/to/DYB-PlanktonNet --kria-masks out
```

`edge_masks.py` also feeds the same port the images in RGB order and checks that it reproduces the PC masks of
`make_masks.py` (`--masks`, default `masks/`). In our run both checks hold for all 120 images: the RGB control
matches the PC masks and the KV260 masks match the BGR port, pixel for pixel.

## License

The outlines and the files in `results/` are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The code in `scripts/` is covered by the repository
license. The DYB-PlanktonNet images keep their own license.
