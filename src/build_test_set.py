"""Utility script to build a test set from a reference CSV.

This script copies files listed in 'data/splits/test_files.csv' from a local
downloaded copy of the original dataset into a local 'data/raw_files/test_set'
folder keeping the original class folder structure.

Usage: run this script from the repository root or call it as a module:

    python -m src.build_test_set

Before running, update 'source_dataset_dir' below to point to your local copy
of the original DYB-PlanktonNet dataset.
"""

import csv
import os
import shutil

def main():
    """Build the test set by copying files declared in the CSV to a target folder.

    The function expects a CSV with headers 'folder' and 'filename' located at
    'data/splits/test_files.csv' relative to the project root.
    """

    # 1. Paths configuration
    # Path where the user has downloaded and extracted the original dataset
    source_dataset_dir = "/home/dsosatr/tesis/DYB-PlanktonNetV1.1/DYB-PlanktonNet"

    # Determine project root dynamically to avoid relative path issues
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    csv_file_path = os.path.join(project_root, "data", "splits", "test_files.csv")
    dest_dir = os.path.join(project_root, "data", "raw_files", "test_set")

    # Preliminary checks
    if not os.path.exists(source_dataset_dir):
        print(f"Error: original dataset not found at: {source_dataset_dir}")
        print(
            "Please update the `source_dataset_dir` variable in this script with your local path."
        )
        return

    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at: {csv_file_path}")
        return

    print("Starting test set construction...")
    print(f"Source: {source_dataset_dir}")
    print(f"Destination: {dest_dir}\n")

    # Counters for the final summary
    copied_files = 0
    missing_files = 0

    # 2. Read CSV and copy images
    with open(csv_file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            folder_name = row["folder"]
            file_name = row["filename"]

            # Build source image path
            src_image_path = os.path.join(source_dataset_dir, folder_name, file_name)

            # Build destination path (preserve class folder structure)
            dest_image_folder = os.path.join(dest_dir, folder_name)
            dest_image_path = os.path.join(dest_image_folder, file_name)

            # If the original image exists, copy it
            if os.path.exists(src_image_path):
                os.makedirs(dest_image_folder, exist_ok=True)
                shutil.copy2(src_image_path, dest_image_path)
                copied_files += 1
            else:
                print(f"Warning: File not found - {src_image_path}")
                missing_files += 1

    # 3. Final summary
    print("\n--- Process Completed ---")
    print(f"Images successfully copied: {copied_files}")
    if missing_files > 0:
        print(f"Images not found (please check paths): {missing_files}")
    else:
        print("The Test Set has been generated 100% according to the original split.")
    print(f"You can find the ready test set at: {dest_dir}")


if __name__ == "__main__":
    main()
