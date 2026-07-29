"""
segmenters.py — Pluggable Stage-2 segmentation backends for IsPlanktonBIO.

The Stage-2 morphology step needs a binary mask isolating the specimen from the
background so its planar area (mm²) can be measured and the cropped region
re-embedded for verification. Two interchangeable backends are provided:

    * OpenCVSegmenter — the default, dependency-free heuristic pipeline
      (Otsu threshold + morphology + connected-components), identical to the
      historical behaviour of the project.
    * SAM2Segmenter — Segment Anything Model 2 (Meta) prompted with automatic
      foreground points derived from the OpenCV preprocessing. Higher-quality
      masks at the cost of a heavy optional dependency and GPU weights.

Both backends implement the same tiny contract::

    success, mask_area, mask_contour = segmenter.segment(stage2_image)

where ``stage2_image`` is a uint8 RGB image (H, W, 3), ``mask_area`` is the
specimen area in **pixels**, and ``mask_contour`` is a uint8 mask (0/255) with
the same H×W as the input (or ``None`` when segmentation fails).

The concrete backend is chosen at runtime by :func:`build_segmenter` from the
``segmentation`` block of the YAML config, so the rest of the pipeline
(`main.py`, `pipeline_worker.py`) stays completely backend-agnostic.

IMPORTANT: SAM2 is an *optional* dependency. It is imported lazily inside
``SAM2Segmenter.__init__`` — never at module import time — so a user who only
runs the OpenCV backend does not need to install SAM2 at all.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.image_utils import get_area_proxy_method, preprocess_image
from src.logger_utils import print_normal, print_red, print_yellow

# Public return type: (success, mask_area_pixels, mask_contour_uint8_or_None)
SegmentationResult = Tuple[bool, float, Optional[np.ndarray]]


class Segmenter(ABC):
    """Abstract base class for a Stage-2 segmentation backend."""

    @abstractmethod
    def segment(self, stage2_image: np.ndarray) -> SegmentationResult:
        """
        Segment the specimen in a Stage-2 image.

        Args:
            stage2_image (np.ndarray): uint8 RGB image, shape (H, W, 3).

        Returns:
            Tuple[bool, float, Optional[np.ndarray]]:
                - success: True if a valid mask was produced.
                - mask_area: specimen area in pixels (0.0 on failure).
                - mask_contour: uint8 mask (0/255), same H×W as input, or None.
        """
        raise NotImplementedError


class OpenCVSegmenter(Segmenter):
    """
    Default heuristic segmenter (Otsu + morphology + connected components).

    This is a thin wrapper around the existing :func:`preprocess_image` and
    :func:`get_area_proxy_method` helpers, preserving the historical behaviour
    of the pipeline exactly. It has no dependencies beyond OpenCV/NumPy.
    """

    def segment(self, stage2_image: np.ndarray) -> SegmentationResult:
        preprocessed_image = preprocess_image(stage2_image)
        success, mask_area, mask_contour = get_area_proxy_method(
            stage2_image, preprocessed_image
        )
        return success, float(mask_area) if mask_area is not None else 0.0, mask_contour


# ---------------------------------------------------------------------------
# SAM2 prompt-point helpers (adapted from the historical sam2-pipeline_v2.py).
# Kept as module-level functions so SAM2Segmenter stays readable; none of them
# import sam2, so importing this module never pulls in the optional dependency.
# ---------------------------------------------------------------------------


def _get_safe_center(
    contour: np.ndarray, img_shape: Tuple[int, int], keep_distance: int = 5
) -> Optional[Tuple[int, int]]:
    """Computes the deepest interior point of a contour via a distance transform."""
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    _, max_dist, _, max_loc = cv2.minMaxLoc(dist_map)
    if max_dist < keep_distance:
        return None
    return max_loc  # (x, y)


def _generate_mesh_points_in_contour(
    contour: np.ndarray,
    border_distance: int,
    points_density: int,
    img_shape: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """
    Generates a regular grid of points and keeps only those strictly inside the
    contour and at least ``border_distance`` pixels away from its boundary.
    """
    img_height, img_width = img_shape
    total_area = img_width * img_height
    spacing = np.sqrt(total_area / points_density)

    x_coords = np.arange(0, img_width, spacing)
    y_coords = np.arange(0, img_height, spacing)

    mesh_points: List[Tuple[int, int]] = []
    for y in y_coords:
        for x in x_coords:
            point = (int(x), int(y))
            dist = cv2.pointPolygonTest(contour, point, True)
            if dist >= border_distance:
                mesh_points.append(point)

    return mesh_points


class SAM2Segmenter(Segmenter):
    """
    Segment Anything Model 2 backend prompted with automatic foreground points.

    The OpenCV preprocessing (:func:`preprocess_image`) is reused to obtain a
    binary silhouette of the specimen, from which foreground prompt points are
    derived (a mesh of interior points, or a single safe-center point). SAM2 is
    then prompted with those points to produce a high-quality mask.

    SAM2 and its dependencies are imported lazily in ``__init__`` so that this
    module can be imported (and the OpenCV backend used) without SAM2 present.
    """

    def __init__(
        self,
        model_cfg: str,
        checkpoint: str,
        prompt_method: str = "mesh",
        points_density: int = 500,
    ) -> None:
        """
        Args:
            model_cfg (str): SAM2 Hydra config name bundled with the ``sam2``
                package, e.g. ``configs/sam2.1/sam2.1_hiera_l.yaml``.
            checkpoint (str): Path to the SAM2 ``.pt`` checkpoint file.
            prompt_method (str): ``"mesh"`` (default) or ``"center"``.
            points_density (int): Target number of grid points for ``"mesh"``.
        """
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:  # pragma: no cover - exercised only w/o SAM2
            raise ImportError(
                "SAM2 backend requested (segmentation.method: sam2) but the "
                "'sam2' package is not installed.\n"
                "Install the optional dependency with:\n"
                "    pip install -r requirements-sam2.txt\n"
                "and download the checkpoint (see README, 'Optional: SAM2 "
                "segmentation'). To run without SAM2, set "
                "segmentation.method: opencv in your config."
            ) from exc

        self._torch = torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print_normal(f"SAM2Segmenter: using device {device}")

        sam2_model = build_sam2(
            model_cfg, checkpoint, device=device, apply_postprocessing=False
        )
        self._predictor = SAM2ImagePredictor(
            sam_model=sam2_model,
            mask_threshold=0,
            max_hole_area=0,
            max_sprinkle_area=0,
        )
        self._device = device
        self._default_prompt_method = prompt_method
        self._points_density = int(points_density)

    def _get_prompt_points(
        self, preprocessed_image: np.ndarray
    ) -> Tuple[Optional[List[Tuple[int, int]]], Optional[Tuple[int, int]]]:
        """
        Derives SAM2 foreground prompt points from a binary silhouette.

        A 10-pixel border is added so appendages touching the frame do not merge
        with the image edge; returned coordinates are shifted back by 10 px. The
        largest contour (image frame) is discarded and the next-largest contour
        (assumed to be the organism) is used. Falls back from ``mesh`` to
        ``center`` and progressively relaxes the border distance if no valid
        points are found.

        Returns:
            (points, center): exactly one of the two is non-None on success;
            both are None if no valid prompt could be generated.
        """
        bordered_image = cv2.copyMakeBorder(
            preprocessed_image, 10, 10, 10, 10,
            borderType=cv2.BORDER_CONSTANT, value=255,
        )

        contours, _ = cv2.findContours(
            bordered_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return None, None

        areas = [cv2.contourArea(c) for c in contours]
        largest_idx = int(np.argmax(areas))
        contours = [c for i, c in enumerate(contours) if i != largest_idx]
        if not contours:
            return None, None

        largest_contour = max(contours, key=cv2.contourArea)

        # Local (non-global) fallback state — thread-safe, unlike the historical
        # implementation which mutated a module-level PROMPT_POINTS_METHOD.
        method = self._default_prompt_method
        border_distance = 5

        while border_distance >= 0:
            if method == "mesh":
                generated = _generate_mesh_points_in_contour(
                    largest_contour,
                    border_distance=border_distance,
                    points_density=self._points_density,
                    img_shape=bordered_image.shape,
                )
                if generated:
                    points = [(p[0] - 10, p[1] - 10) for p in generated]
                    return points, None
            else:
                center = _get_safe_center(
                    largest_contour, bordered_image.shape, border_distance
                )
                if center:
                    return None, (center[0] - 10, center[1] - 10)

            if border_distance > 0:
                border_distance -= 1
                print_yellow(
                    f"{method.upper()}: no valid points, retrying with "
                    f"border_distance={border_distance}"
                )
            elif method == "mesh":
                method = "center"
                border_distance = 10
                print_yellow("MESH failed, switching to CENTER prompt method")
            else:
                print_red("No valid center or mesh points found after all attempts.")
                return None, None

        return None, None

    def _segment_with_coordinates(
        self, image: np.ndarray, coordinates: List[Tuple[int, int]]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Runs SAM2 point-prompted prediction; returns masks/scores sorted by score."""
        torch = self._torch
        with torch.inference_mode(), torch.autocast(
            self._device.type, dtype=torch.bfloat16
        ):
            self._predictor.set_image(image)
            input_point = np.array(coordinates)
            input_label = np.array([1] * len(coordinates))
            masks, scores, _ = self._predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            order = np.argsort(scores)[::-1]
            return masks[order], scores[order]

    def segment(self, stage2_image: np.ndarray) -> SegmentationResult:
        try:
            preprocessed_image = preprocess_image(stage2_image)
            points, center = self._get_prompt_points(preprocessed_image)

            coordinates: List[Tuple[int, int]] = []
            if points:
                coordinates.extend(points)
            if center:
                coordinates.append(center)

            if not coordinates:
                print_yellow("SAM2: no valid prompt points, segmentation failed.")
                return False, 0.0, None

            masks, _ = self._segment_with_coordinates(stage2_image, coordinates)
            if masks is None or len(masks) == 0:
                print_yellow("SAM2: predictor returned no masks.")
                return False, 0.0, None

            best_mask = masks[0]
            # Area is the count of foreground pixels (SAM2 masks are boolean/0-1).
            mask_area = float(np.count_nonzero(best_mask))
            # Emit a 0/255 uint8 mask to match the OpenCV backend's convention,
            # so all downstream consumers (crop, border check, plotting) behave
            # identically regardless of the chosen backend.
            mask_contour = (best_mask.astype(np.uint8) * 255)
            return True, mask_area, mask_contour

        except Exception as exc:  # keep the pipeline alive on per-image failures
            print_red(f"SAM2 segmentation error: {exc}")
            return False, 0.0, None


def build_segmenter(config: dict) -> Segmenter:
    """
    Factory that builds the segmentation backend selected in the YAML config.

    Expected config shape::

        segmentation:
          method: opencv            # 'opencv' (default) | 'sam2'
          sam2:
            model_cfg: 'configs/sam2.1/sam2.1_hiera_l.yaml'
            checkpoint: 'data/models/weights/sam2/sam2.1_hiera_large.pt'
            prompt_method: 'mesh'   # 'mesh' | 'center'
            points_density: 500

    Args:
        config (dict): Parsed YAML configuration.

    Returns:
        Segmenter: The instantiated backend.

    Raises:
        ValueError: If ``segmentation.method`` is not recognised.
        KeyError: If the SAM2 backend is selected without a ``checkpoint``.
    """
    seg_cfg = config.get("segmentation") or {}
    method = str(seg_cfg.get("method", "opencv")).lower()

    if method == "opencv":
        print_normal("Segmentation backend: OpenCV (heuristic)")
        return OpenCVSegmenter()

    if method == "sam2":
        sam2_cfg = seg_cfg.get("sam2") or {}
        checkpoint = sam2_cfg.get("checkpoint")
        if not checkpoint:
            raise KeyError(
                "segmentation.method is 'sam2' but segmentation.sam2.checkpoint "
                "is not set in the config."
            )
        print_normal(f"Segmentation backend: SAM2 (checkpoint={checkpoint})")
        return SAM2Segmenter(
            model_cfg=sam2_cfg.get(
                "model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml"
            ),
            checkpoint=checkpoint,
            prompt_method=sam2_cfg.get("prompt_method", "mesh"),
            points_density=int(sam2_cfg.get("points_density", 500)),
        )

    raise ValueError(
        f"Unknown segmentation.method: {method!r} (expected 'opencv' or 'sam2')."
    )
