"""
pipeline_worker.py — 4-stage threaded producer-consumer pipeline for IsPlanktonBIO.

Mirrors the C++ resnet50_cosine_faiss_turbojpeg_IVFPQ.cpp architecture:

  T1 data_feed_worker   → raw_queue (maxsize=20)   →
  T2 preprocess_worker  → preproc_queue (maxsize=4) →
  T3 inference_worker   → infer_queue (maxsize=4)   →
  T4 output_worker

EOF propagation: each stage puts the None sentinel into its output queue when it is
done (or on error), allowing downstream threads to drain and exit cleanly.

Usage (from main.py):
    raw_queue     = queue.Queue(maxsize=20)
    preproc_queue = queue.Queue(maxsize=4)
    infer_queue   = queue.Queue(maxsize=4)
    bar           = tqdm(total=len(stage1_loader.dataset), desc="Processing")

    threads = [
        threading.Thread(target=data_feed_worker,
                         args=(stage1_loader, stage2_loader, raw_queue)),
        threading.Thread(target=preprocess_worker,
                         args=(raw_queue, preproc_queue)),
        threading.Thread(target=inference_worker,
                         args=(preproc_queue, infer_queue, context)),
        threading.Thread(target=output_worker,
                         args=(infer_queue, context, bar)),
    ]
    for t in threads: t.start()
    for t in threads: t.join()
"""

import logging
import queue
import sqlite3
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.biomass_utils import compute_biomass
from src.dataset import apply_additional_transforms_stage2, loader_with_paths
from src.db_utils import get_label_from_db
from src.image_utils import (
    crop_image_with_mask,
    get_area_proxy_method,
    is_border_touch_acceptable,
    preprocess_image,
)
from src.io_utils import get_subtitles, save_species_info
from src.logger_utils import (
    print_green,
    print_log_red,
    print_log_yellow,
    print_normal,
)
from src.plot_helpers import PlotData, PlotPaths, plot_pipeline_steps

if TYPE_CHECKING:
    from src.pipeline_context import PipelineContext

# ---------------------------------------------------------------------------
# Sentinel value — poison pill propagated through queues to signal EOF / error
# ---------------------------------------------------------------------------
_SENTINEL = None


# ---------------------------------------------------------------------------
# Pipeline data-transfer objects
# ---------------------------------------------------------------------------


@dataclass
class RawItem:
    """
    Produced by T1 (data_feed_worker).
    Wraps a single image pair directly from the DataLoaders.
    """

    stage1_image: torch.Tensor  # shape (1, 3, 224, 224) — stage-1 normalized
    stage2_image: torch.Tensor  # shape (1, 3, H, W)    — ToTensor only
    ground_truth: str
    path: str


@dataclass
class PreprocessedItem:
    """
    Produced by T2 (preprocess_worker).
    Carries forward the stage-1 tensor and adds all CPU preprocessing outputs.
    """

    # --- forwarded from RawItem ---
    stage1_image: torch.Tensor
    ground_truth: str
    path: str
    # --- produced by T2 ---
    stage2_bgr: np.ndarray  # uint8 BGR, shape (H, W, 3) — for plotting
    preprocessed_image: Optional[np.ndarray]  # binary mask from preprocess_image()
    success: bool  # True if contour detection succeeded
    mask_area: float
    mask_contour: Optional[np.ndarray]
    acceptable_segmentation: bool  # True if border-touch check passed
    # (3, 224, 224) stage-2 normalized tensor ready for encoder; None when
    # segmentation failed, border touch rejected, or cropping returned empty
    cropped_tensor: Optional[torch.Tensor]


@dataclass
class InferenceItem:
    """
    Produced by T3 (inference_worker).
    Carries preprocessing metadata and FAISS IDs (label lookup deferred to T4).
    """

    # --- forwarded from PreprocessedItem ---
    stage2_bgr: np.ndarray
    ground_truth: str
    path: str
    success: bool
    mask_area: float
    mask_contour: Optional[np.ndarray]
    acceptable_segmentation: bool
    # --- Stage-1 results (always present) ---
    faiss_id_stage1: int
    dist_stage1: float
    # --- Stage-2 results (None when speculative path was skipped) ---
    faiss_id_stage2: Optional[int] = field(default=None)
    dist_stage2: Optional[float] = field(default=None)


# ---------------------------------------------------------------------------
# T1 — Data Feed Worker
# ---------------------------------------------------------------------------


def data_feed_worker(
    stage1_loader: torch.utils.data.DataLoader,
    stage2_loader: torch.utils.data.DataLoader,
    raw_queue: queue.Queue,
) -> None:
    """
    T1: Zips the two DataLoaders and feeds RawItems into raw_queue.

    Mirrors C++ imageProducer (thread t1).  The DataLoader's internal workers
    (num_workers=8) already handle disk I/O and JPEG decode in parallel, so T1
    is essentially free — it just coordinates the two iterators.

    Args:
        stage1_loader: DataLoader with stage-1 transforms applied.
        stage2_loader: DataLoader with stage-2 raw ToTensor transform.
        raw_queue: Bounded queue (maxsize=20) consumed by T2.
    """
    try:
        for (s1_img, gt, path), (s2_img, _, _) in zip(
            loader_with_paths(stage1_loader), loader_with_paths(stage2_loader)
        ):
            raw_queue.put(RawItem(s1_img, s2_img, gt, path))
    except Exception:
        logging.exception("[T1] Unhandled error in data_feed_worker")
    finally:
        raw_queue.put(_SENTINEL)  # always emit EOF sentinel


# ---------------------------------------------------------------------------
# T2 — CPU Preprocessing Worker
# ---------------------------------------------------------------------------


def preprocess_worker(
    raw_queue: queue.Queue,
    preproc_queue: queue.Queue,
    transform_cfg: Optional[dict] = None,
) -> None:
    """
    T2: Consumes RawItems, performs CPU-bound image preprocessing, and feeds
    PreprocessedItems into preproc_queue.

    Overlaps with T3 GPU inference because PyTorch CUDA operations release the
    GIL, and most OpenCV / NumPy functions do so as well.

    Mirrors C++ imageConsumer (thread t2).

    Args:
        raw_queue: Input queue fed by T1.
        preproc_queue: Output queue consumed by T3 (maxsize=4).
        transform_cfg: Optional dict with stage-2 mean/std from config["transforms"].
            Passed straight through to apply_additional_transforms_stage2().
    """
    try:
        while True:
            item = raw_queue.get()
            if item is _SENTINEL:
                break  # EOF received — propagate downstream and exit

            try:
                _preprocess_one(item, preproc_queue, transform_cfg)
            except Exception:
                logging.exception("[T2] Error preprocessing %s — skipping", item.path)
                # Do NOT emit sentinel on per-item errors; keep the pipeline alive
    except Exception:
        logging.exception("[T2] Unhandled error in preprocess_worker")
    finally:
        preproc_queue.put(_SENTINEL)


def _preprocess_one(
    item: RawItem,
    preproc_queue: queue.Queue,
    transform_cfg: Optional[dict] = None,
) -> None:
    """Runs the CPU preprocessing pipeline for a single image and enqueues result."""
    # ------------------------------------------------------------------
    # Convert stage-2 PyTorch tensor → uint8 BGR numpy for OpenCV
    # (mirrors main.py lines 149-154)
    # ------------------------------------------------------------------
    stage2_np = item.stage2_image.squeeze(0).numpy()
    stage2_np = np.transpose(stage2_np, (1, 2, 0))
    stage2_bgr: np.ndarray = cv2.normalize(
        stage2_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # ------------------------------------------------------------------
    # OpenCV preprocessing: grayscale → brightness/contrast → Otsu
    # → morphological ops → contour detection
    # ------------------------------------------------------------------
    preprocessed_image = preprocess_image(stage2_bgr)
    success, mask_area, mask_contour = get_area_proxy_method(stage2_bgr, preprocessed_image)

    acceptable_segmentation = False
    cropped_tensor: Optional[torch.Tensor] = None

    if success and mask_contour is not None:
        acceptable_segmentation = is_border_touch_acceptable(
            mask_contour, tolerance_percent=10.0
        )
        if acceptable_segmentation:
            cropped = crop_image_with_mask(stage2_bgr, mask_contour)
            # crop_image_with_mask returns a CHW torch.Tensor or None
            if cropped is not None and cropped.numel() > 0:
                # apply_additional_transforms_stage2 expects a tensor input here
                # and returns a (C, H, W) tensor (see dataset.py lines 154-162)
                cropped_tensor = apply_additional_transforms_stage2(
                    cropped, transform_cfg
                )
            # If cropped_tensor is still None after this, T3 will skip Stage-2

    preproc_queue.put(
        PreprocessedItem(
            stage1_image=item.stage1_image,
            ground_truth=item.ground_truth,
            path=item.path,
            stage2_bgr=stage2_bgr,
            preprocessed_image=preprocessed_image,
            success=success,
            mask_area=mask_area if mask_area is not None else 0.0,
            mask_contour=mask_contour,
            acceptable_segmentation=acceptable_segmentation,
            cropped_tensor=cropped_tensor,
        )
    )


# ---------------------------------------------------------------------------
# T3 — GPU Inference Worker
# ---------------------------------------------------------------------------


def inference_worker(
    preproc_queue: queue.Queue,
    infer_queue: queue.Queue,
    context: "PipelineContext",
) -> None:
    """
    T3: Consumes PreprocessedItems, runs GPU inference for Stage-1 (always)
    and Stage-2 (speculatively, when segmentation succeeded), performs FAISS
    nearest-neighbor search, and feeds InferenceItems into infer_queue.

    CUDA releases the GIL during device-side compute, which allows T2 to run
    CPU preprocessing concurrently.

    The Stage-2 path is speculative: it runs whenever `acceptable_segmentation`
    is True and `cropped_tensor` is not None, without waiting for Stage-1 to
    confirm a ground-truth match first.  T4 performs the conditional gating.
    This wastes ~20% of Stage-2 GPU calls (when Stage-1 mispredicts), which is
    acceptable given the system's >80% Stage-1 accuracy.

    Mirrors C++ dpuTaskMux (thread t3).

    Args:
        preproc_queue: Input queue fed by T2.
        infer_queue: Output queue consumed by T4 (maxsize=4).
        context: Shared PipelineContext (models + FAISS indexes are thread-safe
                 for concurrent reads in eval mode).
    """
    # Ensure the CUDA context is initialized in THIS thread.
    # PyTorch CUDA memory is per-process, but the runtime context must be
    # activated on each thread that issues CUDA calls.
    try:
        _warmup = torch.zeros(1, device="cuda")
        del _warmup
    except Exception:
        logging.warning("[T3] CUDA warm-up failed — GPU may be unavailable")

    try:
        while True:
            item = preproc_queue.get()
            if item is _SENTINEL:
                break

            try:
                _infer_one(item, infer_queue, context)
            except Exception:
                logging.exception("[T3] Error on %s — skipping", item.path)
    except Exception:
        logging.exception("[T3] Unhandled error in inference_worker")
    finally:
        infer_queue.put(_SENTINEL)


def _infer_one(
    item: PreprocessedItem,
    infer_queue: queue.Queue,
    context: "PipelineContext",
) -> None:
    """Runs Stage-1 (always) and speculative Stage-2 inference for one image."""
    with torch.no_grad():
        # ------------------------------------------------------------------
        # Stage-1: full-image embedding → FAISS k-NN (k=1)
        # ------------------------------------------------------------------
        s1_img = item.stage1_image.float().cuda(non_blocking=True)
        feat_s1 = context.model_stage1.encoder(s1_img)
        feat_s1 = F.normalize(feat_s1, p=2, dim=1)
        feat_s1_np = feat_s1.cpu().numpy().astype(np.float32)
        D_s1, I_s1 = context.faiss_index_stage1.search(feat_s1_np, 1)
        faiss_id_s1 = int(I_s1[0][0])
        dist_s1 = float(D_s1[0][0])

        # ------------------------------------------------------------------
        # Stage-2 (speculative): cropped-region embedding → FAISS k-NN (k=1)
        # Only runs when T2 produced a valid cropped tensor.
        # ------------------------------------------------------------------
        faiss_id_s2: Optional[int] = None
        dist_s2: Optional[float] = None

        if item.acceptable_segmentation and item.cropped_tensor is not None:
            s2_img = item.cropped_tensor.unsqueeze(0).cuda(non_blocking=True)
            feat_s2 = context.model_stage2.encoder(s2_img)
            feat_s2 = F.normalize(feat_s2, p=2, dim=1)
            feat_s2_np = feat_s2.cpu().numpy().astype(np.float32)
            D_s2, I_s2 = context.faiss_index_stage2.search(feat_s2_np, 1)
            faiss_id_s2 = int(I_s2[0][0])
            dist_s2 = float(D_s2[0][0])

    infer_queue.put(
        InferenceItem(
            stage2_bgr=item.stage2_bgr,
            ground_truth=item.ground_truth,
            path=item.path,
            success=item.success,
            mask_area=item.mask_area,
            mask_contour=item.mask_contour,
            acceptable_segmentation=item.acceptable_segmentation,
            faiss_id_stage1=faiss_id_s1,
            dist_stage1=dist_s1,
            faiss_id_stage2=faiss_id_s2,
            dist_stage2=dist_s2,
        )
    )


# ---------------------------------------------------------------------------
# T4 — Output Worker (DB lookup, biomass, JSON, plotting)
# ---------------------------------------------------------------------------


def output_worker(
    infer_queue: queue.Queue,
    context: "PipelineContext",
    progress_bar: Optional[tqdm] = None,
) -> None:
    """
    T4: Consumes InferenceItems and performs all downstream CPU + I/O work:
    SQLite label lookups, is_valid gating logic, biomass computation,
    JSON writes, and optional matplotlib plotting.

    Opens its OWN SQLite connections from the paths stored in context
    (sqlite3 cursors are NOT thread-safe and must not be shared across threads).

    Mirrors C++ faissTaskMux (thread t4).

    Args:
        infer_queue: Input queue fed by T3.
        context: Shared PipelineContext.
        progress_bar: Optional tqdm instance; updated once per completed image.
    """
    # T4 owns its own SQLite connections (not shared with main thread or T3)
    sql_s1 = sqlite3.connect(context.sqlite_db_path_stage1)
    sql_s2 = sqlite3.connect(context.sqlite_db_path_stage2)
    cursor_s1 = sql_s1.cursor()
    cursor_s2 = sql_s2.cursor()

    try:
        while True:
            item = infer_queue.get()
            if item is _SENTINEL:
                break

            try:
                _output_one(item, context, cursor_s1, cursor_s2)
            except Exception:
                logging.exception("[T4] Error on %s — skipping", item.path)
            finally:
                if progress_bar is not None:
                    progress_bar.update(1)
    except Exception:
        logging.exception("[T4] Unhandled error in output_worker")
    finally:
        sql_s1.close()
        sql_s2.close()


def _output_one(
    item: InferenceItem,
    context: "PipelineContext",
    cursor_s1: sqlite3.Cursor,
    cursor_s2: sqlite3.Cursor,
) -> None:
    """
    Reproduces the gating / output logic of process_single_image() for one image.
    Mirrors main.py lines 141-293.
    """
    config = context.config
    biomass_params = context.biomass_params
    pixel_to_mm2 = context.pixel_to_mm2_factor

    print_normal(f"Processing image: {item.path}.")

    # ------------------------------------------------------------------
    # Stage-1 DB lookup
    # ------------------------------------------------------------------
    result_query_stage1 = get_label_from_db(cursor_s1, item.faiss_id_stage1)
    result_query_stage2: Optional[str] = None
    is_valid = "unmatched"
    biomass_val = None

    print_green(
        f"Real image class: {item.ground_truth} - Queried image class: {result_query_stage1}"
    )

    class_id, image_id = get_subtitles(item.path)

    if result_query_stage1 == item.ground_truth:
        # Stage-1 matched ground truth
        if item.success and item.mask_contour is not None:
            if item.acceptable_segmentation:
                # Stage-2 inference was run speculatively in T3; check result
                if item.faiss_id_stage2 is not None:
                    result_query_stage2 = get_label_from_db(cursor_s2, item.faiss_id_stage2)

                    print_green(
                        f"Real image class: {item.ground_truth} - "
                        f"OpenCV image class: {result_query_stage2}"
                    )

                    if result_query_stage2 == result_query_stage1:
                        is_valid = "matched"

                        biomass_val = compute_biomass(
                            species_label=result_query_stage2,
                            area_pixels=item.mask_area,
                            biomass_params=biomass_params,
                            pixel_to_mm2_factor=pixel_to_mm2,
                        )
                        if biomass_val is not None:
                            print_green(
                                f"Estimated Biomass for {result_query_stage2}: "
                                f"{biomass_val:.4f} mg"
                            )
                        else:
                            print_log_yellow(
                                f"No biomass parameters found in JSON for "
                                f"{result_query_stage2}"
                            )
                    else:
                        is_valid = "unmatched"

                    save_species_info(
                        class_id,
                        image_id,
                        config["paths"]["species_json"],
                        is_valid,
                        result_query_stage2,
                        biomass_val,
                    )
                else:
                    # T3 skipped Stage-2 (cropped_tensor was None despite acceptable seg.)
                    print_log_yellow(f"Failed cropping: {item.path}")
                    is_valid = "failed-cropping"
                    save_species_info(
                        class_id,
                        image_id,
                        config["paths"]["species_json"],
                        is_valid,
                        "none",
                        None,
                    )
            else:
                print_log_yellow(f"The specimen touches border: {item.path}")
                is_valid = "touches-border"
                save_species_info(
                    class_id,
                    image_id,
                    config["paths"]["species_json"],
                    is_valid,
                    result_query_stage1,
                    None,
                )
        else:
            print_log_red(f"Failed segmentation: {item.path}")
            is_valid = "failed-segmentation"
            save_species_info(
                class_id,
                image_id,
                config["paths"]["species_json"],
                is_valid,
                "none",
                None,
            )
    else:
        print_log_red(
            f"Stage-1 CNN prediction does not match ground truth: {item.path}"
        )
        is_valid = "unmatched-stage1"
        save_species_info(
            class_id,
            image_id,
            config["paths"]["species_json"],
            is_valid,
            result_query_stage1,
            None,
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    plot_data = PlotData(
        stage2_image=item.stage2_bgr,
        preprocessed_image_color=None,
        mask_contour=item.mask_contour,
        mask_area=item.mask_area,
        path=item.path,
        ground_truth=item.ground_truth,
        result_query_stage1=result_query_stage1,
        result_query_stage2=result_query_stage2,
        is_valid=is_valid,
        acceptable_segmentation=item.acceptable_segmentation,
    )
    plot_paths = PlotPaths(
        AREAS_JSON_PATH=config["paths"]["areas_json"],
        PLOTS_SAVE_PATH=config["paths"]["plots_save_path"],
        PLOTS_SAVE_PATH_FAILED=config["paths"]["plots_save_failed"],
        MASK_SAVE_PATH=config["paths"]["mask_save_path"],
        MASK_SAVE_PATH_FAILED=config["paths"]["mask_save_failed"],
        SAVE_PLOT_FIGURES=config.get("runtime", {}).get("save_plot_figures", True),
    )
    plot_pipeline_steps(plot_data, plot_paths)
