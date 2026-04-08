"""
IsPlanktonBIO: A two-stage pipeline for plankton taxonomy and morphological trait extraction.

This module implements a comprehensive computer vision pipeline combining Supervised
Contrastive Learning (SupCon) embeddings with classical image processing techniques.
The execution is divided into two main components:

    1. Stage-1 (Taxonomy): An Information Retrieval (IR) system utilizing a FAISS
       index for highly accurate, initial species classification.
    2. Stage-2 (Morphology & Verification): An automated segmentation protocol
       using OpenCV heuristics to isolate specimens, extract their planar area in mm²
       (as a preparatory step for downstream biomass estimation), and perform a
       secondary embedding verification on the cropped region.

Configuration is handled via a YAML file (e.g., 'configs/default_config.yaml').
Please refer to the README.md file at the repository root for detailed execution
instructions, dataset structure guidelines, and reproducible environment setups.
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
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
    get_area_proxy_method,
    is_border_touch_acceptable,
    otsu_normal,
    preprocess_image,
)
from src.io_utils import get_subtitles, save_mask_area, save_species_info
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

    print_normal(f"Processing image: {path}.")
    stage1_image = stage1_image.float().cuda()
    features_stage1 = model_stage1.encoder(stage1_image)
    features_stage1 = torch.nn.functional.normalize(features_stage1, p=2, dim=1)
    feature_stage1_np = features_stage1.cpu().numpy().astype(np.float32)
    D_stage1, I_stage1 = faiss_index_stage1.search(feature_stage1_np, 1)
    faiss_id_stage1 = int(I_stage1[0][0])
    result_query_stage1 = get_label_from_db(sql_cursor_stage1, faiss_id_stage1)
    result_query_stage2 = None

    print_green(
        f"Real image class: {ground_truth} - Queried image class: {result_query_stage1}"
    )

    # Convert from PyTorch tensor to numpy array for STAGE-2 processing
    stage2_image = stage2_image.squeeze(0).numpy()
    stage2_image = np.transpose(stage2_image, (1, 2, 0))
    # Normalize to [0, 255] uint8 range from float32
    stage2_image = cv2.normalize(
        stage2_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # Preprocess the image
    preprocessed_image = preprocess_image(stage2_image)
    success, mask_area, mask_contour = get_area_proxy_method(
        stage2_image, preprocessed_image
    )
    acceptable_segmentation = False

    class_id, image_id = get_subtitles(path)

    if result_query_stage1 == ground_truth:
        if success and mask_contour is not None:
            acceptable_segmentation = is_border_touch_acceptable(
                mask_contour, tolerance_percent=10.0
            )
            if acceptable_segmentation:
                cropped_image = crop_image_with_mask(stage2_image, mask_contour)
                cropped_image = apply_additional_transforms_stage2(
                    cropped_image, config.get("transforms")
                )

                if cropped_image is not None and cropped_image.numel() > 0:
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

                    print_green(
                        f"Real image class: {ground_truth} - OpenCV image class: {result_query_stage2}"
                    )

                    if result_query_stage2 == result_query_stage1:
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
                        biomass_val = None

                    save_species_info(
                        class_id,
                        image_id,
                        config["paths"]["species_json"],
                        is_valid,
                        result_query_stage2,
                        biomass_val,
                    )
                else:
                    print_log_yellow(f"Failed cropping: {path}")
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
                print_log_yellow(f"The specimen touches border: {path}")
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
            print_log_red(f"Failed segmentation: {path}")
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
        print_log_red(f"Stage-1 CNN prediction does not match ground truth: {path}")
        is_valid = "unmatched-stage1"
        save_species_info(
            class_id,
            image_id,
            config["paths"]["species_json"],
            is_valid,
            result_query_stage1,
            None,
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
    )
    plot_pipeline_steps(plot_data, plot_paths)


def main():
    """
    Main entry point for executing the entire processing pipeline.
    Reads configurations, instantiates directories, initializes dataloaders, databases,
    and hardware models, iterates through the test suite, and computes operational runtimes.
    """
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Global pipeline parameters
    pixel_to_mm2 = config.get("pixel_to_mm2", 0.0000544644)

    FAISS_INDEX_PATH_STAGE_1 = config["paths"]["faiss_index_stage_1"]
    FAISS_INDEX_PATH_STAGE_2 = config["paths"]["faiss_index_stage_2"]
    SQLITE_DB_PATH_STAGE_1 = config["paths"]["sqlite_db_stage_1"]
    SQLITE_DB_PATH_STAGE_2 = config["paths"]["sqlite_db_stage_2"]
    CKPT_PATH_STAGE_1 = config["paths"]["ckpt_stage_1"]
    CKPT_PATH_STAGE_2 = config["paths"]["ckpt_stage_2"]
    MODEL_NAME = config["model"]["name"]

    faiss_index_stage1 = load_faiss_index(FAISS_INDEX_PATH_STAGE_1)
    sql_stage1 = sqlite3.connect(SQLITE_DB_PATH_STAGE_1)
    sql_cursor_stage1 = sql_stage1.cursor()
    faiss_index_stage2 = load_faiss_index(FAISS_INDEX_PATH_STAGE_2)
    sql_stage2 = sqlite3.connect(SQLITE_DB_PATH_STAGE_2)
    sql_cursor_stage2 = sql_stage2.cursor()
    model_stage1 = load_model(CKPT_PATH_STAGE_1, MODEL_NAME)
    model_stage2 = load_model(CKPT_PATH_STAGE_2, MODEL_NAME)

    stage1_loader, stage2_loader = set_loader(
        dataset_root_path=config["paths"]["dataset_root_path"],
        transform_cfg=config.get("transforms"),
    )

    model_stage1.eval()
    model_stage2.eval()

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
    )

    start_time = time.time()  # Start timing

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

    print(f"Total number of images in test set: {len(stage1_loader.dataset)}")
    print(f"Total elapsed time: {end_time - start_time:.2f} seconds")

    total_images = len(stage1_loader.dataset)
    mean_time_per_image = (end_time - start_time) / total_images
    print(f"Mean processing time per image: {mean_time_per_image:.4f} seconds")


if __name__ == "__main__":
    main()
