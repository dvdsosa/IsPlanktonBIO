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


def save_species_info(
    class_id, image_id, json_path, is_valid, result_inference, biomass_val=None
):
    """
    Logs species classification inferences to a structured JSON file mapping pipeline statuses.

    Args:
        class_id (str): The ground-truth class directory name.
        image_id (str): The processed image filename.
        json_path (str): System path for the output JSON serialization.
        is_valid (str): The categorical status constraint (e.g., 'matched', 'touches-border').
        result_inference (str): The final predicted class label string.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # Load existing data if the file exists
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError:
            data = {
                "matched": {},
                "unmatched": {},
                "failed-cropping": {},
                "touches-border": {},
                "failed-segmentation": {},
                "unmatched-stage1": {},
            }
    else:
        data = {
            "matched": {},
            "unmatched": {},
            "failed-cropping": {},
            "touches-border": {},
            "failed-segmentation": {},
            "unmatched-stage1": {},
        }

    if class_id not in data[is_valid]:
        data[is_valid][class_id] = []

    # Create entry with file, predicted, and status information
    entry = {
        "file": image_id,
        "predicted": result_inference if result_inference != "none" else None,
    }
    if biomass_val is not None:
        # Format the biomass nicely
        entry["biomass"] = round(biomass_val, 6)

    # Add entry to the list if not already present (check by filename)
    if not any(item.get("file") == image_id for item in data[is_valid][class_id]):
        data[is_valid][class_id].append(entry)

    # Save the updated data back to the file
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    print_normal(f"Saved species info for {class_id}/{image_id} as {is_valid}")
