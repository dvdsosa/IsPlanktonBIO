"""
IsPlanktonBIO: A two-stage pipeline for plankton taxonomy and morphological trait extraction.

This module implements a comprehensive computer vision pipeline combining Supervised
Contrastive Learning (SupCon) embeddings with classical image processing techniques.
The execution is divided into two main components:

    1. Stage-1 (Taxonomy): An Information Retrieval (IR) system utilizing a FAISS
       index for highly accurate, initial species classification.
    2. Stage-2 (Morphology & Verification): An automated segmentation protocol
       to isolate specimens, extract their planar area in mm² (as a preparatory
       step for downstream biomass estimation), and perform a secondary embedding
       verification on the cropped region. The segmentation backend is pluggable
       and selected via the 'segmentation.method' config field: 'opencv' (default
       heuristic — Otsu + morphology) or 'sam2' (Segment Anything Model 2, an
       optional GPU-backed dependency). See src/segmenters.py.

Configuration is handled via a YAML file (e.g., 'configs/default_config.yaml').
Please refer to the README.md file at the repository root for detailed execution
instructions, dataset structure guidelines, and reproducible environment setups.
"""

import argparse
import json
import logging
import os
import queue
import sqlite3
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from torchvision import datasets, transforms
from tqdm import tqdm

from data.models.resnet_big import SupConResNet
from src.biomass_utils import compute_biomass
from src.dataset import (
    apply_additional_transforms_stage2,
    loader_with_paths,
    set_loader,
)
from src.db_utils import get_label_from_db
from src.image_utils import (
    add_scale_bar,
    adjust_brightness_contrast_cv,
    convert_area_pixels_to_mm2,
    crop_image_with_mask,
    is_border_touch_acceptable,
    otsu_normal,
)
from src.io_utils import accumulate_result, dump_results_json, get_subtitles
from src.logger_utils import (
    ENABLE_PLOT_DISPLAY,
    ENABLE_PRINTING,
    ENABLE_PRINTING_RED,
    ENABLE_PRINTING_YELLOW,
    print_green,
    print_log_red,
    print_log_yellow,
    print_normal,
    print_red,
    print_yellow,
)
from src.models_utils import load_faiss_index, load_model
from src.pipeline_context import PipelineContext
from src.segmenters import build_segmenter
from src.pipeline_worker import (
    data_feed_worker,
    inference_worker,
    output_worker,
    preprocess_worker,
)
from src.plot_helpers import (
    PlotData,
    PlotPaths,
    plot_pipeline_steps,
    visualize_pipeline_images,
)


def parse_args():
    """
    Parses CLI arguments required to run the main application.

    Returns:
        argparse.Namespace: Command-line arguments populated by the user or default values.
    """
    parser = argparse.ArgumentParser(
        description="IsPlanktonBIO: A two-stage pipeline for plankton taxonomy and morphological trait extraction."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--threaded",
        action="store_true",
        default=False,
        help=(
            "Run the 4-stage threaded pipeline (T1→T2→T3→T4) instead of the "
            "default sequential loop.  Overlaps CPU preprocessing (T2), GPU "
            "inference (T3), and file I/O (T4) for higher throughput."
        ),
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help=(
            "Apply torch.compile() to both encoder models before inference. "
            "Requires PyTorch >= 2.0.  Uses mode='reduce-overhead', which "
            "minimises Python-dispatcher overhead per call — the dominant cost "
            "at batch_size=1.  Expect a slow first 1-2 images while the "
            "TorchInductor kernels are traced and cached."
        ),
    )
    return parser.parse_args()


def process_single_image(
    stage1_image,
    ground_truth,
    path,
    stage2_image,
    context: PipelineContext,
):
    """
    Executes the entire end-to-end evaluation pipeline on a single query image instance.
    This includes feature extraction via a Stage-1 model, finding nearest neighbors,
    running image segmentation checks, executing Stage-2 analysis on valid candidates,
    and managing the output telemetry and plots.

    Args:
        stage1_image (torch.Tensor): Image prepared for Stage-1 evaluation.
        ground_truth (str): Ground-truth class label.
        path (str): File system path of the current source image.
        stage2_image (torch.Tensor): Image prepared for Stage-2 evaluation.
        context (PipelineContext): Model and configuration context for the pipeline.

    Returns:
        str: The final biological prediction returned by the most restrictive pipeline stage.
    """
    is_valid = "unmatched"

    model_stage1 = context.model_stage1
    model_stage2 = context.model_stage2
    faiss_index_stage1 = context.faiss_index_stage1
    faiss_index_stage2 = context.faiss_index_stage2
    sql_cursor_stage1 = context.sql_cursor_stage1
    sql_cursor_stage2 = context.sql_cursor_stage2
    config = context.config
    biomass_params = context.biomass_params
    pixel_to_mm2 = context.pixel_to_mm2_factor
    threshold_stage1 = context.cosine_threshold_stage1
    threshold_stage2 = context.cosine_threshold_stage2

    print_normal(f"Processing image: {path}.")
    stage1_image = stage1_image.float().cuda()
    features_stage1 = model_stage1.encoder(stage1_image)
    features_stage1 = torch.nn.functional.normalize(features_stage1, p=2, dim=1)
    feature_stage1_np = features_stage1.cpu().numpy().astype(np.float32)
    D_stage1, I_stage1 = faiss_index_stage1.search(feature_stage1_np, 1)
    faiss_id_stage1 = int(I_stage1[0][0])
    result_query_stage1 = get_label_from_db(sql_cursor_stage1, faiss_id_stage1)
    confidence_score_stage1 = float(D_stage1[0][0])

    result_query_stage2 = None
    confidence_score_stage2 = None
    mask_contour = None
    # 0.0 (not None) so the plotting / area-saving step never multiplies None
    # when Stage-1 is OOD-rejected before segmentation runs (matches the
    # threaded path, where T2 also defaults mask_area to 0.0).
    mask_area = 0.0
    acceptable_segmentation = False
    biomass_val = None

    _, image_id = get_subtitles(path)

    print_green(
        f"Real image class: {ground_truth} - Queried image class: {result_query_stage1} "
        f"(similarity_s1={confidence_score_stage1:.4f})"
    )

    # Convert from PyTorch tensor to numpy array for STAGE-2 processing and
    # plotting. Done unconditionally (even when Stage-1 is OOD-rejected) so the
    # plotting block always receives a numpy image.
    # Undo ToTensor() by multiplying by 255 — NORM_MINMAX is FORBIDDEN here,
    # since it would rescale each image's dynamic range and corrupt the
    # brightness-dependent segmentation.
    stage2_image = stage2_image.squeeze(0).numpy()
    stage2_image = np.transpose(stage2_image, (1, 2, 0))
    stage2_image = (stage2_image * 255.0).clip(0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Stage-1 OOD gate: reject when the nearest-neighbour cosine similarity
    # falls below the calibrated threshold. Unlike the previous behaviour,
    # the ground truth is NOT used to gate the pipeline — it only serves as
    # the JSON grouping key and for logging/plotting.
    # ------------------------------------------------------------------
    if confidence_score_stage1 < threshold_stage1:
        print_log_red(
            f"Stage-1 rejected by cosine threshold (similarity={confidence_score_stage1:.4f} "
            f"< {threshold_stage1}): {path}"
        )
        is_valid = "ood_sample_stage1"
    else:
        # Segment the specimen using the configured backend (OpenCV or SAM2).
        success, mask_area, mask_contour = context.segmenter.segment(stage2_image)

        if not (success and mask_contour is not None):
            print_log_red(f"Failed segmentation: {path}")
            is_valid = "failed-segmentation"
        else:
            acceptable_segmentation = is_border_touch_acceptable(
                mask_contour, tolerance_percent=10.0
            )
            if not acceptable_segmentation:
                print_log_yellow(f"The specimen touches border: {path}")
                is_valid = "touches-border"
            else:
                cropped_image = crop_image_with_mask(stage2_image, mask_contour)
                cropped_image = apply_additional_transforms_stage2(
                    cropped_image, config.get("transforms")
                )

                if cropped_image is None or cropped_image.numel() == 0:
                    print_log_yellow(f"Failed cropping: {path}")
                    is_valid = "failed-cropping"
                else:
                    transformed_img = cropped_image.unsqueeze(0).to(stage1_image.device)
                    features_stage2 = model_stage2.encoder(transformed_img)
                    features_stage2 = torch.nn.functional.normalize(
                        features_stage2, p=2, dim=1
                    )
                    feature_stage2_np = features_stage2.cpu().numpy().astype(np.float32)
                    D_stage2, I_stage2 = faiss_index_stage2.search(feature_stage2_np, 1)
                    faiss_id_stage2 = int(I_stage2[0][0])
                    result_query_stage2 = get_label_from_db(
                        sql_cursor_stage2, faiss_id_stage2
                    )
                    confidence_score_stage2 = float(D_stage2[0][0])

                    print_green(
                        f"Real image class: {ground_truth} - OpenCV image class: "
                        f"{result_query_stage2} (similarity_s2={confidence_score_stage2:.4f})"
                    )

                    # Stage-2 OOD gate + Stage-1/Stage-2 agreement check
                    if confidence_score_stage2 < threshold_stage2:
                        print_log_yellow(
                            f"Stage-2 rejected by cosine threshold "
                            f"(similarity={confidence_score_stage2:.4f} < {threshold_stage2}): {path}"
                        )
                        is_valid = "ood_sample_stage2"
                    elif result_query_stage2 == result_query_stage1:
                        is_valid = "matched"

                        # --- BIOMASS COMPUTATION ---
                        # Pixel length for DYB-PlanktonNet is 7.38 μm.
                        # Area per pixel in mm^2 = (7.38 * 10^-3)^2
                        biomass_val = compute_biomass(
                            species_label=result_query_stage2,
                            area_pixels=mask_area,
                            biomass_params=biomass_params,
                            pixel_to_mm2_factor=pixel_to_mm2,
                        )

                        if biomass_val is not None:
                            print_green(
                                f"Estimated Biomass for {result_query_stage2}: {biomass_val:.4f} mg"
                            )
                        else:
                            print_log_yellow(
                                f"No biomass parameters found in JSON for {result_query_stage2}"
                            )
                        # ----------------------------------------
                    else:
                        is_valid = "unmatched"

    accumulate_result(
        context.results_dict,
        ground_truth,
        image_id,
        result_query_stage1,
        confidence_score_stage1,
        result_query_stage2,
        confidence_score_stage2,
        is_valid,
        biomass_mg=biomass_val,
    )

    plot_data = PlotData(
        stage2_image=stage2_image,
        preprocessed_image_color=None,
        mask_contour=mask_contour,
        mask_area=mask_area,
        path=path,
        ground_truth=ground_truth,
        result_query_stage1=result_query_stage1,
        result_query_stage2=result_query_stage2,
        is_valid=is_valid,
        acceptable_segmentation=acceptable_segmentation,
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


def main():
    """
    Main entry point for executing the entire processing pipeline.
    Reads configurations, instantiates directories, initializes dataloaders, databases,
    and hardware models, iterates through the test suite, and computes operational runtimes.
    """
    # Suppress verbose logging from external libraries to keep tqdm progress bar visible
    logging.getLogger("sam2").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("torchvision").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    # Suppress root logger INFO messages from any library
    logging.getLogger().setLevel(logging.WARNING)

    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Global pipeline parameters
    pixel_to_mm2 = config.get("pixel_to_mm2", 0.0000544644)

    # FAISS cosine-similarity thresholds for OOD rejection. A sample is rejected
    # when its nearest-neighbour cosine similarity falls below the threshold.
    # A very negative default (-9999.0) disables the gate (never rejects).
    cosine_threshold_stage1 = float(config.get("cosine_threshold_stage_1", -9999.0))
    cosine_threshold_stage2 = float(config.get("cosine_threshold_stage_2", -9999.0))

    # Determine segmentation method to select appropriate stage-2 checkpoints
    segmentation_method = config.get("segmentation", {}).get("method", "opencv")

    FAISS_INDEX_PATH_STAGE_1 = config["paths"]["faiss_index_stage_1"]
    SQLITE_DB_PATH_STAGE_1 = config["paths"]["sqlite_db_stage_1"]
    CKPT_PATH_STAGE_1 = config["paths"]["ckpt_stage_1"]

    # Select stage-2 paths based on segmentation method
    if segmentation_method == "sam2":
        FAISS_INDEX_PATH_STAGE_2 = config["paths"]["faiss_index_stage_2_sam2"]
        SQLITE_DB_PATH_STAGE_2 = config["paths"]["sqlite_db_stage_2_sam2"]
        CKPT_PATH_STAGE_2 = config["paths"]["ckpt_stage_2_sam2"]
    else:  # opencv (default)
        FAISS_INDEX_PATH_STAGE_2 = config["paths"]["faiss_index_stage_2_opencv"]
        SQLITE_DB_PATH_STAGE_2 = config["paths"]["sqlite_db_stage_2_opencv"]
        CKPT_PATH_STAGE_2 = config["paths"]["ckpt_stage_2_opencv"]

    MODEL_NAME = config["model"]["name"]

    faiss_index_stage1 = load_faiss_index(FAISS_INDEX_PATH_STAGE_1)
    sql_stage1 = sqlite3.connect(SQLITE_DB_PATH_STAGE_1)
    sql_cursor_stage1 = sql_stage1.cursor()
    faiss_index_stage2 = load_faiss_index(FAISS_INDEX_PATH_STAGE_2)
    sql_stage2 = sqlite3.connect(SQLITE_DB_PATH_STAGE_2)
    sql_cursor_stage2 = sql_stage2.cursor()
    model_stage1 = load_model(CKPT_PATH_STAGE_1, MODEL_NAME)
    model_stage2 = load_model(CKPT_PATH_STAGE_2, MODEL_NAME)

    # Build the Stage-2 segmentation backend (OpenCV by default, SAM2 optional).
    # SAM2 is imported lazily inside its constructor, so the OpenCV path never
    # requires the optional 'sam2' dependency.
    segmenter = build_segmenter(config)

    stage1_loader, stage2_loader = set_loader(
        dataset_root_path=config["paths"]["dataset_root_path"],
        transform_cfg=config.get("transforms"),
    )

    model_stage1.eval()
    model_stage2.eval()

    # ------------------------------------------------------------------
    # Optional torch.compile — activate with --compile or
    # runtime.compile_models: true in the YAML config.
    #
    # We compile model.encoder (not the top-level wrapper) because all
    # inference calls go through model.encoder(image).  Compiling only
    # the wrapper's forward() would leave the encoder path unoptimised.
    #
    # mode="reduce-overhead" is best for batch_size=1: it traces a CUDA
    # graph so each forward skips Python-level dispatch entirely.
    # Trade-off: the first ~2 images are slow while Inductor traces and
    # caches the kernels.  Every subsequent call is significantly faster.
    #
    # Bonus: enable TF32 for free throughput on Ampere/Ada/Hopper GPUs
    # (no accuracy loss for float32 convolutions in practice).
    # ------------------------------------------------------------------
    use_compile = args.compile or config.get("runtime", {}).get("compile_models", False)
    if use_compile:
        if hasattr(torch, "compile"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("=> Applying torch.compile(mode='reduce-overhead') to both encoders...")
            model_stage1.encoder = torch.compile(
                model_stage1.encoder, mode="reduce-overhead"
            )
            model_stage2.encoder = torch.compile(
                model_stage2.encoder, mode="reduce-overhead"
            )
            print("   Encoders compiled. First 1-2 images will be slow (kernel tracing).")
        else:
            print(
                "WARNING: --compile requested but torch.compile is unavailable "
                "(requires PyTorch >= 2.0). Continuing without compilation."
            )

    # Load biomass parameters
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lehette_path = os.path.join(
        project_root, "data", "databases", "lehette_parameters.json"
    )
    with open(lehette_path, "r", encoding="utf-8") as bf:
        biomass_params = json.load(bf)

    context = PipelineContext(
        model_stage1=model_stage1,
        model_stage2=model_stage2,
        faiss_index_stage1=faiss_index_stage1,
        faiss_index_stage2=faiss_index_stage2,
        sql_cursor_stage1=sql_cursor_stage1,
        sql_cursor_stage2=sql_cursor_stage2,
        config=config,
        biomass_params=biomass_params,
        pixel_to_mm2_factor=pixel_to_mm2,
        # Paths for T4 to open its own SQLite connections (cursors are not thread-safe)
        sqlite_db_path_stage1=SQLITE_DB_PATH_STAGE_1,
        sqlite_db_path_stage2=SQLITE_DB_PATH_STAGE_2,
        cosine_threshold_stage1=cosine_threshold_stage1,
        cosine_threshold_stage2=cosine_threshold_stage2,
        segmenter=segmenter,
    )

    start_time = time.time()  # Start timing

    if args.threaded:
        # ------------------------------------------------------------------
        # 4-stage threaded pipeline: T1→T2→T3→T4
        # Overlaps DataLoader iteration (T1), CPU preprocessing (T2), GPU
        # inference + FAISS search (T3), and DB/JSON/plot I/O (T4).
        # Activate with: python -m src.main --threaded
        # ------------------------------------------------------------------
        raw_queue = queue.Queue(maxsize=20)
        preproc_queue = queue.Queue(maxsize=4)
        infer_queue = queue.Queue(maxsize=4)
        bar = tqdm(total=len(stage1_loader.dataset), desc="Processing [threaded]")

        threads = [
            threading.Thread(
                target=data_feed_worker,
                args=(stage1_loader, stage2_loader, raw_queue),
                name="T1-DataFeed",
                daemon=True,
            ),
            threading.Thread(
                target=preprocess_worker,
                args=(
                    raw_queue,
                    preproc_queue,
                    config.get("transforms"),
                    context.segmenter,
                ),
                name="T2-Preprocess",
                daemon=True,
            ),
            threading.Thread(
                target=inference_worker,
                args=(preproc_queue, infer_queue, context),
                name="T3-Inference",
                daemon=True,
            ),
            threading.Thread(
                target=output_worker,
                args=(infer_queue, context, bar),
                name="T4-Output",
                daemon=True,
            ),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        bar.close()
    else:
        # ------------------------------------------------------------------
        # Original sequential path (default) — preserved for reference and
        # as a correctness baseline for --threaded verification.
        # ------------------------------------------------------------------
        with torch.no_grad():
            for (stage1_image, ground_truth, path), (stage2_image, _, _) in tqdm(
                zip(loader_with_paths(stage1_loader), loader_with_paths(stage2_loader)),
                desc="Processing",
            ):
                process_single_image(
                    stage1_image,
                    ground_truth,
                    path,
                    stage2_image,
                    context,
                )

    sql_stage1.close()
    sql_stage2.close()
    end_time = time.time()  # End timing

    # Write the fully accumulated inference results in a single I/O operation.
    dump_results_json(context.results_dict, config["paths"]["species_json"])

    print(f"Total number of images in test set: {len(stage1_loader.dataset)}")
    print(f"Total elapsed time: {end_time - start_time:.2f} seconds")

    total_images = len(stage1_loader.dataset)
    mean_time_per_image = (end_time - start_time) / total_images
    print(f"Mean processing time per image: {mean_time_per_image:.4f} seconds")


if __name__ == "__main__":
    main()
