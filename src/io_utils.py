import json
import os
from typing import Tuple

from src.logger_utils import print_normal


def get_subtitles(image_path: str) -> Tuple[str, str]:
    """
    Extracts the class path and filename from the given image path.
    Args:
        image_path (str): The path to the image file.
    Returns:
        tuple: A tuple containing the class path (str) and the filename (str).
    """
    # Normalize the path to remove '..'
    normalized_path = os.path.normpath(image_path)
    # Get the filename
    local_filename = os.path.basename(normalized_path)
    # Get the last subdirectory
    local_class_path = os.path.basename(os.path.dirname(normalized_path))

    return local_class_path, local_filename


def save_mask_area(
    class_id: str, image_id: str, mask_area: float, json_path: str
) -> None:
    """
    Save the mask area to a JSON file.

    Args:
        class_id (str): The class ID to use as the parent key
        image_id (str): The image ID to use as the child key
        mask_area (int): The computed area of the mask
        json_path (str): Path to the JSON file where the area will be saved

    Returns:
        None
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # Load existing data if the file exists
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    # Update the data with the new mask area
    if class_id not in data:
        data[class_id] = {}

    # Store mask area as string to match the expected format
    data[class_id][image_id] = f"{mask_area:.3f}"

    # Save the updated data back to the file
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    print_normal(f"Saved mask area {int(mask_area)} for {class_id}/{image_id}")


def accumulate_result(
    results_dict: dict,
    ground_truth: str,
    filename: str,
    stage_1_prediction,
    stage_1_confidence,
    stage_2_prediction,
    stage_2_confidence,
    pipeline_status: str,
    biomass_mg=None,
) -> None:
    """
    Appends one inference entry to the in-memory results dict.

    The parent key is always ``ground_truth`` (class name); the pipeline status
    is stored inside each entry instead of grouping the top-level dict by status.
    This keeps both the Stage-1 and Stage-2 predictions (with their confidences)
    for every image and avoids the status-based grouping that leaked information
    into downstream evaluation. Results are collected in memory and written once
    at the end of the run via :func:`dump_results_json`.

    Args:
        results_dict: Mapping of ground_truth -> list[entry_dict] accumulated in place.
        ground_truth: True class label (used as parent key).
        filename: Image filename.
        stage_1_prediction: Predicted class from Stage 1 (None if not available).
        stage_1_confidence: Stage-1 FAISS cosine similarity (None if not available).
        stage_2_prediction: Predicted class from Stage 2 (None if Stage 2 did not run).
        stage_2_confidence: Stage-2 FAISS cosine similarity (None if Stage 2 did not run).
        pipeline_status: Final pipeline outcome ('matched', 'unmatched',
            'ood_sample_stage1', 'ood_sample_stage2', 'failed-segmentation',
            'touches-border', 'failed-cropping', ...).
        biomass_mg: Estimated biomass in mg (None if not computed).
    """
    entry = {
        "filename": filename,
        "stage_1_prediction": stage_1_prediction,
        "stage_1_confidence": round(float(stage_1_confidence), 6)
        if stage_1_confidence is not None
        else None,
        "stage_2_prediction": stage_2_prediction,
        "stage_2_confidence": round(float(stage_2_confidence), 6)
        if stage_2_confidence is not None
        else None,
        "pipeline_status": pipeline_status,
        "biomass_mg": round(float(biomass_mg), 6) if biomass_mg is not None else None,
    }

    bucket = results_dict.setdefault(ground_truth, [])
    if not any(item["filename"] == filename for item in bucket):
        bucket.append(entry)


def dump_results_json(results_dict: dict, json_path: str) -> None:
    """
    Writes a fully accumulated inference-results dict to disk in one I/O operation.

    Designed to be called once after the full inference loop instead of writing
    after every image, eliminating the per-image read-modify-write bottleneck.

    Args:
        results_dict: Mapping of ground_truth -> list[entry_dict].
        json_path: Destination file path (created if absent).
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results_dict, fh, indent=4)
    total = sum(len(v) for v in results_dict.values())
    print_normal(f"Saved {total} inference results -> {json_path}")
