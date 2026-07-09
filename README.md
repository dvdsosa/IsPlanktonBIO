# IsPlanktonBIO: A Two-Stage Pipeline for Plankton Taxonomy and Morphological Trait Extraction

[![Journal](https://img.shields.io/badge/Journal-IEEE_JOURNAL_OF_OCEANIC_ENGINEERING-blue.svg)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=48)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official codebase for the manuscript submitted to the ***IEEE JOURNAL OF OCEANIC ENGINEERING***. 

**IsPlanktonBIO** implements a comprehensive computer vision pipeline combining Supervised Contrastive Learning (SupCon) embeddings with classical image processing techniques. The execution is twofold:
1. **Stage-1 (Taxonomy):** Integrates the [IsPlanktonIR](https://doi.org/10.1093/icesjms/fsac198) Information Retrieval system, utilizing a FAISS index for high-accuracy initial species classification.
2. **Stage-2 (Morphology & Verification):** Employs automated OpenCV-based segmentation to isolate specimens, calculates planar area for biomass estimation, and cross-verifies the initial prediction using secondary morphological embeddings.

### How the pipeline decides (per image)

The FAISS indices use Inner Product over L2-normalised embeddings, so each nearest-neighbour search returns a **cosine similarity** in `[-1, 1]` (higher = more similar). The pipeline does **not** use the ground-truth label to decide anything; the ground truth is only recorded as the grouping key in the output and used for logging. Each image flows through the following gates:

1. **Stage-1 OOD gate** — if the Stage-1 cosine similarity is **below** `cosine_threshold_stage_1`, the image is rejected as `ood_sample_stage1`.
2. **Segmentation** — if OpenCV cannot isolate a specimen contour → `failed-segmentation`.
3. **Border check** — if the specimen touches the image border beyond tolerance → `touches-border`.
4. **Cropping** — if the masked crop is empty → `failed-cropping`.
5. **Stage-2 OOD gate** — if the Stage-2 cosine similarity is **below** `cosine_threshold_stage_2`, the image is rejected as `ood_sample_stage2`.
6. **Agreement** — if Stage-1 and Stage-2 predictions agree → `matched` (and biomass is computed); otherwise → `unmatched`.

> **OOD thresholds.** A sample is rejected when its similarity falls *below* the threshold. A very negative value (the shipped default `-9999.0`) **disables** the gate, so nothing is ever rejected and every image proceeds to segmentation. Replace `cosine_threshold_stage_1` / `cosine_threshold_stage_2` in the config with values calibrated on an In-Distribution validation set to enable OOD rejection.

### Output format (`output/inference_results.json`)

Results are accumulated in memory and written once at the end of the run. The JSON is keyed by **ground-truth class name**; each image is one entry that keeps both stage predictions, their similarities, the final pipeline status, and the estimated biomass:

```json
{
    "Copepoda": [
        {
            "filename": "img_00123.jpg",
            "stage_1_prediction": "Copepoda",
            "stage_1_confidence": 0.8421,
            "stage_2_prediction": "Copepoda",
            "stage_2_confidence": 0.7965,
            "pipeline_status": "matched",
            "biomass_mg": 0.004512
        }
    ]
}
```

`stage_2_*` and `biomass_mg` are `null` when the image did not reach the corresponding step.

---

## ⚙️ Installation & Requirements

First, clone the repository to your local machine:
```bash
git clone https://github.com/your-username/IsPlanktonBIO.git
cd IsPlanktonBIO
```

To set up the environment and install all necessary dependencies for deep learning inference (`torch`, `torchvision`), vector embeddings (`faiss-cpu`), and data processing, run:

```bash
conda create -n paper2 python=3.12.3 -y
conda activate paper2
pip install -r requirements.txt
```

---

## 📥 Pre-trained Models and Database Files

Due to file size limits, the large model weights (`.pth`), FAISS indices (`.bin`), and SQLite databases (`.sqlite`) are hosted externally on **Zenodo** at the following DOI: [10.5281/zenodo.19162092](https://doi.org/10.5281/zenodo.19162092).

Please download the following files from Zenodo and place them in their respective directories within the `data/` folder before running the pipeline:

- **`data/models/weights/`**
  - `resnet50_stage1.pth`
  - `resnet50_stage2_opencv.pth`
- **`data/databases/vector/`**
  - `faiss_index_stage1_pruned_089.bin`
  - `faiss_index_stage2_opencv_pruned_091.bin`
- **`data/databases/sql/`**
  - `plankton_db_stage1_pruned_089.sqlite`
  - `plankton_db_stage2_opencv_pruned_091.sqlite`

---

## 🚀 Usage (Running the Full Pipeline)

Once the environment and the dataset are ready (see the *"Dataset Access"* section below), you can launch the complete two-stage inference and morphological extraction pipeline.

**Load the test set into RAM first.** The embedded Kria target loads the dataset from a RAM-backed filesystem, so for a fair, apples-to-apples timing comparison we run the desktop pipeline the same way. Copy the test set into `/dev/shm` (tmpfs) — this matches the `dataset_root_path` in `configs/default_config.yaml`:

```bash
cp -r /home/dsosatr/tesis/DYB-VOS/test /dev/shm/test_dataset_ram
```

Then run the pipeline:

```bash
python -m src.main --config configs/default_config.yaml
```

When you are done, erase the dataset from RAM to free the tmpfs space:

```bash
rm -rf /dev/shm/test_dataset_ram
```

The pipeline will read the configurations, load models and FAISS indices, evaluate the test set, calculate biomass, and output the tracking results and telemetry.

---

## 📊 Quick Evaluation (Pre-computed Results)

If you wish to verify the metrics reported in the paper (Accuracy, F1-Score, etc.) without downloading the large image dataset or running the full deep learning inference, we provide the raw prediction outputs in a JSON file.

You can generate the final tables and comprehensive classification reports instantly by running:

```bash
python -m tests.evaluate_metrics --json_path output/inference_results.json
```

This script reads the true labels and both stage predictions from the JSON file and reports three views of performance:

- **Pipeline Acceptance / Rejection Analysis** — the fraction of images accepted (`matched`), rejected as OOD at each stage, rejected for touching the border, or left `unmatched` (Stage-1 ≠ Stage-2).
- **Global Classifier Cascade (End-to-End)** — accuracy/precision/recall/F1 using the best available prediction (Stage-2 when present, else Stage-1).
- **Stage Metrics** — the raw classifier accuracy of each backbone, without OOD gating.

A copy of the printed report is also written to `tests/output/<method>_acc.txt` (the path is auto-derived from the JSON filename, or set explicitly with `--output`).

---

## 📂 Dataset Access and Reproducibility

Due to licensing restrictions, the original DYB-PlanktonNet dataset cannot be hosted directly in this repository. However, to ensure full reproducibility of the results presented in our paper, we provide the exact data splits used during our experiments.

To run the evaluation pipeline from scratch, please follow these steps:

### 1. Download the Dataset
Obtain the original images by downloading the dataset directly from IEEE Dataport: [DYB-PlanktonNet](https://ieee-dataport.org/documents/dyb-planktonnet).

### 2. Recreate the Test Partition
We have provided CSV files detailing the exact images used for our splits in the `data/splits/` directory. Each file contains `folder,filename` pairs.
To evaluate the models, open `data/splits/test_files.csv` and copy the specified images from the downloaded dataset into a dedicated testing directory (e.g., `data/raw_files/test_set/`).

#### Automated Setup Script
To facilitate the test set creation and ensure full reproducibility, we provide a Python script that automatically reads the CSV and copies the exact same images used in our study. This guarantees the models are evaluated on the identical data split, reproducing the exact results published in the paper while preserving the required class-folder structure.

Open `src/build_test_set.py` and ensure the `source_dataset_dir` variable points to your downloaded dataset. Then, run:
```bash
python -m src/build_test_set.py
```
The script uses only standard Python libraries and will automatically generate the `data/raw_files/test_set/` directory for you.

### 3. Update Configuration
Before running `src.main`, you must update the configuration file to point to your newly created test set. Open `configs/default_config.yaml` and modify the `dataset_root_path` under the `paths` section with the absolute path to your test directory:
```yaml
paths:
  # Replace this with the actual absolute path on your local machine
  dataset_root_path: '/path/to/your/repository/data/raw_files/test_set/'
```
*Note: The `train_files.csv` and `val_files.csv` are also provided in the `data/splits/` folder for complete transparency and in case you wish to retrain the models from scratch.*

---
