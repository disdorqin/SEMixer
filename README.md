# SEMixer: Semantics Enhanced MLP-Mixer for Multiscale Mixing and Long-term Time Series Forecasting

[![arXiv](https://img.shields.io/badge/arXiv-2602.16220-red)](https://arxiv.org/pdf/2602.16220)

> **Published in:** ACM Web Conference 2026 (WWW 2026)

This repository contains the official implementation of **SEMixer**, a lightweight multiscale model for long-term time series forecasting. SEMixer enhances temporal representation learning through patch-level semantics using the **Random Attention Mechanism (RAM)** and the **Multiscale Progressive Mixing Chain (MPMC)**.

![Framework](repo/figure/framework_SEMixer.jpg)

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dataset Preparation](#dataset-preparation)
- [Reproduction Guide](#reproduction-guide)
  - [Script Overview](#script-overview)
  - [Quick Start](#quick-start)
  - [Advanced Usage](#advanced-usage)
- [Result Collection](#result-collection)
- [Citation](#citation)

---

## Requirements

- **OS:** Windows (recommended) / Linux
- **Python:** 3.11+
- **GPU:** NVIDIA GPU with CUDA support (recommended for faster training)
- **RAM:** 16GB+ recommended
- **Storage:** ~2GB for datasets and checkpoints

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/disdorqin/SEMixer.git
cd SEMixer
```

### 2. Create and Activate Virtual Environment

**Windows:**
```batch
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

> **Note:** A `requirements.txt` file is not included in the repository. You can install the core dependencies manually:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn matplotlib einops
```

If you encounter missing packages during runtime, install them as needed (e.g., `tqdm`, `psutil`, etc.).

### 4. Verify Environment

```bash
python run_reproduction.py check
```

This will print your Python path, PyTorch version, and CUDA availability.

---

## Project Structure

```
SEMixer/
├── repo/                           # Core model code
│   ├── run.py                      # Main training script (do not call directly)
│   ├── models/                     # Model implementations (SEMixer, PatchTST, etc.)
│   ├── layers/                     # Network layers and attention mechanisms
│   ├── exp/                        # Experiment wrappers
│   ├── data_provider/              # Data loaders
│   ├── utils/                      # Utility functions
│   ├── dataset/                    # Time series datasets (CSV files)
│   └── figure/                     # Figures for paper
├── LongTermTSF_SEMixer/            # Generated checkpoints & logs (not in git)
├── run_reproduction.py             # **Primary Python entry point for reproduction**
├── run_reproduction.bat            # Windows batch wrapper for easy mode
├── collect_results.py              # Result aggregation script
├── 运行_简单.bat                   # Run easy datasets (ETTh1, ETTh2)
├── 运行_中等.bat                   # Run medium datasets
├── 运行_困难.bat                   # Run hard datasets
├── 运行_全部.bat                   # Run all datasets
├── 运行脚本_README.md              # Original Chinese script documentation
├── SEMixer_论文复现完整指南.md      # Detailed reproduction guide (Chinese)
└── README.md                       # This file
```

---

## Dataset Preparation

The datasets used in the paper are public time series benchmarks. They should be placed in the `repo/dataset/` directory.

**Included datasets in `repo/dataset/`:**
- `ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv`
- `weather.csv`, `electricity.csv`, `exchange_rate.csv`, `national_illness.csv`

**Note:** `solar_AL.txt` and `traffic.csv` may need to be downloaded separately if not present. You can download them from [Google Drive](https://drive.google.com/drive/folders/1PPLsAoDbv4WcoXDp-mm4LFxoKwewnKxX).

---

## Reproduction Guide

### Script Overview

Below is a summary of every important script in this repository:

| Script | Purpose | When to Use |
|--------|---------|-------------|
| **`run_reproduction.py`** | **Main Python entry point.** Supports modes: `easy`, `medium`, `hard`, `all`, `custom`, `collect`, `check`. | **Use this for all reproduction tasks.** |
| **`run_reproduction.bat`** | Windows batch file that activates `.venv` and runs `run_reproduction.py easy --collect-after`. | Quick one-click run for beginners. |
| **`collect_results.py`** | Aggregates all experimental results from `LongTermTSF_SEMixer/` and generates `实验结果汇总.md` with MSE/MAE tables and paper comparison. | Run after experiments finish. |
| **`运行_简单.bat`** | Runs **easy** datasets (`ETTh1`, `ETTh2`). ~30 min, 40 training runs. | **Recommended first step.** |
| **`运行_中等.bat`** | Runs **medium** datasets (`ETTm1`, `ETTm2`, `exchange_rate`, `national_illness`). ~2 hours, 80 runs. | Run after easy mode succeeds. |
| **`运行_困难.bat`** | Runs **hard** datasets (`weather`, `electricity`, `traffic`, `solar_AL`). ~8 hours, needs good GPU. | For full reproduction. |
| **`运行_全部.bat`** | Runs **all 10 datasets**. ~10–15 hours. | For complete paper reproduction. |
| **`repo/run.py`** | Core training loop. Defines dataset-specific hyperparameters and iterates over seeds/prediction lengths. | Called internally by `run_reproduction.py`; avoid direct use unless you know the CLI. |

### Difficulty Classification

| Difficulty | Datasets | Variables | Total Runs | Est. Time |
|------------|----------|-----------|------------|-----------|
| **Easy** ⭐ | ETTh1, ETTh2 | 7 | 40 | ~30 min |
| **Medium** ⭐⭐ | ETTm1, ETTm2, exchange_rate, national_illness | 7–8 | 80 | ~2 hours |
| **Hard** ⭐⭐⭐ | weather, electricity, traffic, solar_AL | 21–862 | 80 | ~8 hours |

### Quick Start

#### Option A: One-Click Batch File (Windows)

Double-click the batch file corresponding to the difficulty you want:

```batch
运行_简单.bat        # Recommended for first try
运行_中等.bat
运行_困难.bat
运行_全部.bat
```

#### Option B: Command Line (Cross-Platform)

Make sure `.venv` is activated, then:

```bash
# Check environment
python run_reproduction.py check

# Run easy mode + auto-collect results
python run_reproduction.py easy --collect-after

# Run medium mode
python run_reproduction.py medium --collect-after

# Run all datasets
python run_reproduction.py all --collect-after
```

### Advanced Usage

#### Custom Single Experiment

Run a specific dataset, prediction length, and seed:

```bash
python run_reproduction.py custom --dataset ETTh1 --pred-len 96 --seed 0
```

#### Collect Results Manually

If you already ran experiments and want to regenerate the report:

```bash
python run_reproduction.py collect
# or directly:
python collect_results.py --output-file 实验结果汇总.md
```

#### Using `repo/run.py` Directly

> ⚠️ Only recommended for advanced users familiar with the code.

```bash
cd repo
python run.py easy      # easy | medium | hard | all
```

---

## Result Collection

After experiments complete, the `collect_results.py` script scans the `LongTermTSF_SEMixer/` directory and generates a Markdown report (`实验结果汇总.md`) containing:

1. **Per-dataset results** — MSE and MAE for each prediction length and seed.
2. **Average results** — Mean MSE/MAE per dataset and overall.
3. **Paper comparison** — Side-by-side comparison with the paper’s reported results.

**Checkpoints and logs** are saved under:

```
LongTermTSF_SEMixer/
├── ETTh1/
│   └── random_seed_0/
│       └── ETTh1_SEMixer_SeqLen1280_PredLen96_HiddenDim_128/
│           ├── checkpoint.pth
│           ├── record_args.json
│           ├── record_all_loss_train.json
│           ├── record_all_loss_val.json
│           └── record_all_loss_test.json
```

Each folder contains:
- `checkpoint.pth` — Best model weights.
- `record_args.json` — Hyperparameters used.
- `record_all_loss_*.json` — Training/validation/test loss curves.

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{zhang2026semixer,
  title={SEMixer: Semantics Enhanced MLP-Mixer for Multiscale Mixing and Long-term Time Series Forecasting},
  author={Zhang, Xu and Wang, Qitong and Wang, Peng and Wang, Wei},
  journal={arXiv preprint arXiv:2602.16220},
  year={2026}
}
```

```bibtex
@inproceedings{zhang2025lightweight,
  title={A lightweight sparse interaction network for time series forecasting},
  author={Zhang, Xu and Wang, Qitong and Wang, Peng and Wang, Wei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={12},
  pages={13304--13312},
  year={2025}
}
```

---

## License

See [repo/LICENSE](repo/LICENSE) for details.

## Contact

For questions or issues, please open a GitHub Issue in this repository.
