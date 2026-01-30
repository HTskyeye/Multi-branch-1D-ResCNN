# Multi-branch 1D-ResCNN for Spectrum Prediction

This repository contains the full pipeline used in our paper: **data preprocessing → model training → result export** for a deep-learning-based spectral task.

- **Inputs** are generated from `100-IT.xls` → `ai_features.xlsx`
- **Labels** are generated from `spectrum_data.xlsx` → `spectrum_labels.xlsx` (**3 channels**, each resampled to **120 points**)
- Training code includes data loading, model definition, training loop, and export of predictions/checkpoints.

------

## 1. Model Overview (Brief)

The model is designed to predict **3-channel spectral labels** (each channel is a 120-point sequence) from processed input features.

**High-level architecture:**

- A **feature encoder** processes the input representation.
- A **multi-head / multi-branch predictor** outputs three spectral channels.
- Internally, we use a **1D CNN-based backbone** with residual blocks (stable for smooth spectral curves), and a lightweight projection head to output a 120-length sequence per channel.

**Training objective:**

- Regression loss between predicted and ground-truth spectra (L1).

(Exact architectural details and hyperparameters are specified in `configs/*.yaml` and `src/spectra_dl/models/`.)

## 2. Environment

We recommend using conda and pinning NumPy to 1.x to avoid binary compatibility issues with some compiled dependencies .

```bash
conda create -n spectra python=3.10 "numpy<2" -y
conda activate spectra
pip install -r requirements.txt
```

------

## 3. Repository Structure

```
project-root/
  configs/                    # YAML configs (paths, hyperparameters)
  data/
    raw/                      # Raw inputs (spectrum_data.xlsx, 100-IT.xls)
    processed/                # Preprocessed outputs (generated)
  scripts/                    # One-click reproduction scripts
  src/spectra_dl/             # Preprocess / dataset / model / training
  runs/                       # Outputs (auto-generated)
```

------

## 4. Data Preprocessing

### 4.1 Labels: `spectrum_labels.xlsx` (3 channels)

- Generated from `spectrum_data.xlsx`
- Labels contain **3 channels** (recommended sheets: `ch1`, `ch2`, `ch3`)
- Each channel is **smoothed/interpolated** and **uniformly resampled to 120 points**
- The sample axis ordering (row/column) follows the preprocessing scripts and is consistent with the dataset loader.

### 4.2 Inputs: `ai_features.xlsx`

- Generated from `100-IT.xls`
- Contains model input features aligned with the label samples (same sample ordering / count)

------

## 5. Quick Start (Recommended for Reviewers)

### 5.1 One-click reproduction

#### Windows (Anaconda Prompt)

From the project root:

```bat
scripts\reproduce.bat
```

#### Linux/macOS

```bash
bash scripts/reproduce.sh
```

This runs:

1. Spectrum preprocessing → `data/processed/spectrum_labels.xlsx`
2. Response preprocessing → `data/processed/ai_features.xlsx`
3. Training → outputs saved under `runs/<exp_name>/`

------

### 5.2 Step-by-step execution (more transparent)

#### (0) Go to project root and set module path

Windows:

```bat
cd C:\path\to\project-root
set PYTHONPATH=src
```

Linux/macOS:

```bash
cd /path/to/project-root
export PYTHONPATH=src
```

#### (1) Preprocess spectrum labels (3-channel, 120 points)

Windows:

```bat
python -m spectra_dl.preprocess.spectrum ^
  --input data\raw\spectrum_data.xlsx ^
  --output data\processed\spectrum_labels.xlsx ^
  --n_points 120
```

Linux/macOS:

```bash
python -m spectra_dl.preprocess.spectrum \
  --input data/raw/spectrum_data.xlsx \
  --output data/processed/spectrum_labels.xlsx \
  --n_points 120
```

#### (2) Preprocess response inputs

Windows:

```bat
python -m spectra_dl.preprocess.response ^
  --input data\raw\100-IT.xls ^
  --output data\processed\ai_features.xlsx
```

Linux/macOS:

```bash
python -m spectra_dl.preprocess.response \
  --input data/raw/100-IT.xls \
  --output data/processed/ai_features.xlsx
```

#### (3) Train the model

```bash
python -m spectra_dl.train --config configs/default.yaml
```

------

## 6. Outputs

Each run creates a directory under `runs/<exp_name>/`, typically containing:

- `config_resolved.yaml` — snapshot of the effective config
- `split.json` — train/test split indices (for reproducibility)
- `loss_history.csv` (or `.xlsx`) — training curves
- `checkpoints/` — saved model weights (best/last)
- `test_predictions_*.xlsx` — per-sample predictions (true vs pred) for 3 channels

------

## 7. License 

- License: see `LICENSE`

