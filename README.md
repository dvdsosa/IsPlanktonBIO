# IsPlanktonBIO: A Two-Stage Pipeline for Plankton Taxonomy and Morphological Trait Extraction

[![Journal](https://img.shields.io/badge/Journal-IEEE_JOURNAL_OF_OCEANIC_ENGINEERING-blue.svg)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=48)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official codebase for the manuscript submitted to the ***IEEE JOURNAL OF OCEANIC ENGINEERING***. 

**IsPlanktonBIO** implements a comprehensive computer vision pipeline combining Supervised Contrastive Learning (SupCon) embeddings with classical image processing techniques. The execution is twofold:
1. **Stage-1 (Taxonomy):** Integrates the [IsPlanktonIR](https://doi.org/10.1093/icesjms/fsac198) Information Retrieval system, utilizing a FAISS index for high-accuracy initial species classification.
2. **Stage-2 (Morphology & Verification):** Employs automated OpenCV-based segmentation to isolate specimens, calculates planar area for biomass estimation, and cross-verifies the initial prediction using secondary morphological embeddings.
---

## ⚙️ Installation & Requirements

First, clone the repository to your local machine:
```bash
git clone https://github.com/your-username/IsPlanktonBIO.git
cd IsPlanktonBIO
```

To set up the environment and install all necessary dependencies for deep learning inference (`torch`, `torchvision`), vector embeddings (`faiss-cpu`), and data processing, run:

```bash
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

Once the environment and the dataset are ready (see the *"Dataset Access"* section below), you can launch the complete two-stage inference and morphological extraction pipeline by running:

```bash
python -m src.main --config configs/default_config.yaml
```

The pipeline will read the configurations, load models and FAISS indices, evaluate the test set, calculate biomass, and output the tracking results and telemetry.

---

## 📊 Quick Evaluation (Pre-computed Results)

If you wish to verify the metrics reported in the paper (Accuracy, F1-Score, etc.) without downloading the large image dataset or running the full deep learning inference, we provide the raw prediction outputs in a JSON file.

You can generate the final tables and comprehensive classification reports instantly by running:

```bash
python -m src.evaluate_metrics
```

This script reads the true labels and model predictions from the JSON file and calculates all the exact figures presented in the manuscript.

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
