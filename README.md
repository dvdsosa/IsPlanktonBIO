# IsPlanktonBIO: A Two-Stage Pipeline for Plankton Taxonomy and Morphological Trait Extraction

[![Journal](https://img.shields.io/badge/Journal-Results_in_Engineering-blue.svg)](https://www.sciencedirect.com/journal/results-in-engineering)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD 2-Clause](https://img.shields.io/badge/License-BSD_2--Clause-yellow.svg)](https://opensource.org/licenses/BSD-2-Clause)

This repository contains the official codebase for "When Simpler Segmentation Wins: Geometric Fidelity and Energy Efficiency in Edge Plankton Biometry," submitted to ***Results in Engineering*** (Elsevier).

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
git clone https://github.com/dvdsosa/IsPlanktonBIO.git
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

Due to file size limits, the large model weights (`.pth`), FAISS indices (`.bin`), and SQLite databases (`.sqlite`) are hosted externally on **Zenodo** at the following DOI: [10.5281/zenodo.21676347](https://doi.org/10.5281/zenodo.21676347).

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

The files above cover Stage-1 and the default **`opencv`** Stage-2 backend. If you plan to run the optional **`sam2`** Stage-2 backend instead (see *"Optional: SAM2 Segmentation Backend"* below), also download:

- **`data/models/weights/`**
  - `resnet50_stage2_sam2.pth`
- **`data/databases/vector/`**
  - `faiss_index_stage2_pruned_093.bin`
- **`data/databases/sql/`**
  - `plankton_db_stage2_pruned_093.sqlite`

> Note: the SAM2 *checkpoint* itself (`sam2.1_hiera_large.pt`) is **not** part of this Zenodo record — it is downloaded separately from Meta's official release, as described in the SAM2 section below.

---

## 🚀 Usage (Running the Full Pipeline)

Once the environment and the dataset are ready (see the *"Dataset Access"* section below), you can launch the complete two-stage inference and morphological extraction pipeline.

**Load the test set into RAM first.** The embedded Kria target loads the dataset from a RAM-backed filesystem, so for a fair, apples-to-apples timing comparison we run the desktop pipeline the same way. Copy the test set into `/dev/shm` (tmpfs) — this matches the `dataset_root_path` in `configs/default_config.yaml`:

```bash
cp -r /home/dsosatr/tesis/DYB/test /dev/shm/test_dataset_ram
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

## 🧩 Optional: SAM2 Segmentation Backend

The Stage-2 morphology step supports two interchangeable segmentation backends,
selected by the `segmentation.method` field in `configs/default_config.yaml`:

- **`opencv`** (default): a dependency-free heuristic (Otsu threshold +
  morphology + connected components). This is what the paper reports and
  requires nothing beyond `requirements.txt`.
- **`sam2`**: [Segment Anything Model 2](https://github.com/facebookresearch/sam2)
  (Meta), automatically prompted with foreground points derived from the same
  preprocessing. It produces higher-quality masks at the cost of a heavy
  optional dependency and GPU weights.

SAM2 is imported lazily, so **you do not need to install anything extra to run
the default OpenCV backend.** To enable SAM2:

1. Install the optional dependency (requires a CUDA-capable GPU):

   ```bash
   pip install --no-build-isolation -r requirements-sam2.txt
   ```

   > **Note:** The `--no-build-isolation` flag is required to bypass SAM2's 
   > package name mismatch issue during build.

2. Download the checkpoint used in this project — SAM2.1 **Hiera Large** — directly
   from Meta's official SAM2 release (it is **not** part of this project's Zenodo
   archive) and place it at `data/models/weights/sam2/sam2.1_hiera_large.pt`:

   ```bash
   mkdir -p data/models/weights/sam2
   curl -L -o data/models/weights/sam2/sam2.1_hiera_large.pt \
       https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
   ```

   (Other variants — tiny/small/base_plus — and the official download script are
   listed at [facebookresearch/sam2](https://github.com/facebookresearch/sam2#getting-started).)

3. Set the backend in your config:

   ```yaml
   segmentation:
     method: 'sam2'
   ```

Then run the pipeline as usual (`python -m src.main --config configs/default_config.yaml`).
Both backends are compatible with the `--threaded` execution mode.

### SAM2 Installation Troubleshooting

SAM2 from GitHub has a package name mismatch during build (`sam-2` in metadata vs 
`sam2` in requirements). The `--no-build-isolation` flag bypasses this:

```bash
pip install --no-build-isolation -r requirements-sam2.txt
```

**If that still fails**, try installing the package directly:
```bash
pip install --no-build-isolation git+https://github.com/facebookresearch/sam2.git@main
```

After successful installation, verify it works:
```bash
python -c "from sam2.build_sam import build_sam2; print('SAM2 imported successfully')"
```

If you continue to experience issues, consider using the default **OpenCV backend** 
(`segmentation.method: 'opencv'`) instead, which requires no extra dependencies.

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

## ⏱️ Benchmarking (FPS & Power/Energy on a desktop GPU)

Two helper scripts benchmark the pipeline on a CUDA desktop and report, as
**mean ± std** over N repetitions: **FPS**, peak process RAM, GPU power/memory/utilization,
CPU utilization/temperature, CPU package power (via RAPL), and **energy per inference**
(J/image) for GPU, CPU, and total.

- **`benchmark_gpu.sh <backend> [runs]`** — benchmarks a single backend
  (`opencv` or `sam2`). Requires the test set already loaded in RAM.
- **`run_all_gpu.sh [runs]`** — sweeps **both** backends sequentially; it loads
  the test set into `/dev/shm` automatically and removes it afterwards.

Outputs land in `output/metrics/metrics_gpu_<backend>.txt` plus the renamed
`output/inference_results_gpu_<backend>.json` and `output/species_results_gpu_<backend>.json`.

### When do I need `sudo`?

The **only** measurement that requires elevated privileges is **CPU package power/energy**,
read from the kernel RAPL counter (`/sys/class/powercap/intel-rapl:0/energy_uj`), which is
root-only on recent kernels. Everything else (FPS, GPU power/memory, CPU utilization/temperature,
RAM) is captured without `sudo`.

| Goal | Command |
|------|---------|
| Everything **except** CPU power | run **without** `sudo` |
| **Also** CPU power/energy (RAPL) | run **with** `sudo -E` |

Running under `sudo -E` preserves your environment; the script hands the generated files
back to your user (`$SUDO_USER`) when it finishes.

### Copy-paste ready

Activate your environment first (the script auto-activates `paper2` only if `torch` is missing):

```bash
conda activate paper2
```

**Both backends, no CPU power** (loads/frees the RAM dataset automatically):

```bash
bash run_all_gpu.sh 10
```

**Both backends, WITH CPU power/energy** (RAPL → needs `sudo`):

```bash
sudo -E bash run_all_gpu.sh 10
```

**Single backend** (`benchmark_gpu.sh` does *not* manage the RAM dataset, so load it first):

```bash
# 1) Load the test set into RAM (matches dataset_root_path in the config)
cp -r /home/dsosatr/tesis/DYB-VOS/test /dev/shm/test_dataset_ram

# 2a) OpenCV backend — without CPU power
bash benchmark_gpu.sh opencv 10
# 2b) OpenCV backend — with CPU power (RAPL)
sudo -E bash benchmark_gpu.sh opencv 10

# 2c) SAM2 backend — without CPU power
bash benchmark_gpu.sh sam2 10
# 2d) SAM2 backend — with CPU power (RAPL)
sudo -E bash benchmark_gpu.sh sam2 10

# 3) Free the RAM when done
rm -rf /dev/shm/test_dataset_ram
```

The second argument (`10` above) is the number of repetitions; it defaults to `10` if omitted.

> **Notes.**
> - The `sam2` backend needs the optional dependency and checkpoint from the
>   *"Optional: SAM2 Segmentation Backend"* section above.
> - A single-instance lock prevents two benchmarks from running at once (they
>   share the same `output/*.json` files); wait for one to finish before starting another.
> - If a run fails, its `output/metrics/run_<backend>_<n>.log` is kept with the
>   error; on success the per-run log is removed.

---

## 🛰️ Edge Deployment Code (AMD Kria KV260)

The paper's edge numbers (0.337 J/inference, 19.33 FPS, 6.52 W) come from a
separate C++ implementation that runs on the board under Vitis AI, archived
verbatim (entry-point source renamed only) under [`kria/`](kria/):

```
kria/
├── ASSETS.md                                    # where each config-referenced file actually comes from
├── FAISS.md                                     # building FAISS/OpenBLAS on the board (prerequisite)
├── resnet50_cosine_faiss_turbojpeg_IVFPQ.cpp    # 4-stage threaded pipeline
├── configs/default_config_paper2.yaml
├── .vscode/tasks.json                           # build task — see FAISS.md first
├── models/*.xmodel                              # DPU-compiled Stage-1/Stage-2 backbones
├── monitor_power.sh
├── run_paper2_experiments.sh                    # → output/metrics/metrics_paper2.txt
└── run_paper2_idle_experiments.sh               # → output/metrics/metrics_paper2_idle.txt (source of the paper's 4.41 W idle figure)
```

Build FAISS/OpenBLAS first ([`kria/FAISS.md`](kria/FAISS.md)), then compile
with the only task in `.vscode/tasks.json`, using
`resnet50_cosine_faiss_turbojpeg_IVFPQ.cpp` as the active file. Fetch the
FAISS/SQLite/weight files the config points to from
[`kria/ASSETS.md`](kria/ASSETS.md).

```bash
cp -r ../DYB-linearHead/test /dev/shm/test_dataset_ram
./bin/resnet50_cosine_faiss_turbojpeg_IVFPQ --config configs/default_config_paper2.yaml
```

For the paper's benchmark table (mean ± std over 10 runs):
`./run_paper2_experiments.sh 10` (active inference) and
`./run_paper2_idle_experiments.sh 120` (idle-power baseline) — both use
`monitor_power.sh`, included alongside them.

> Targets the Kria KV260 under PetaLinux/Vitis AI; archived here for
> provenance, not for building on a desktop.

---
