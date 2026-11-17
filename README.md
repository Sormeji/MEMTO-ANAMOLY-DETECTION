Here is the updated `README.md` including the full **SMD (Server Machine Dataset)** setup, expected `.npy` file shapes (`(708405, 38)` and `(708420, 38)`), and all three execution commands (`--input_c 38 --output_c 38 --anormly_ratio 0.5`):

```markdown
# MEMTO: Memory-guided Transformer for Multivariate Time-Series & Tabular Anomaly Detection

An adapted and extended implementation of **MEMTO** (*Memory-guided Transformer for Multivariate Time-Series Anomaly Detection*, NeurIPS 2023), supporting both standard continuous sensor telemetry benchmarks (**SMD**, **MSL**, **SMAP**) and tabular financial fraud detection (**Credit Card Transactions**).

---

## 📌 Overview

MEMTO combines a Transformer encoder-decoder with a gated memory module initialized via K-Means clustering to learn prototypical normal patterns and detect anomalies via reconstruction and latent memory deviations.

This repository includes:
- **Continuous Telemetry Pipelines:** Training and evaluation workflows for **SMD** (38 channels), **MSL** (55 channels), and **SMAP** (25 channels).
- **Tabular Financial Fraud Adaptation:** Custom preprocessing, feature engineering (Haversine distance, category-relative spending Z-scores, cyclic temporal encoding, `StandardScaler` normalization), and point-wise windowing (`win_size=1`) for **Priyam Choksi's Credit Card Transactions** dataset.
- **CPU-Compatible Execution:** Patched clustering and data loading workflows for running end-to-end on CPU environments.

---

## 📂 Project Structure

```text
MEMTO-main/
├── checkpoints/                  # Saved Phase 1 and Phase 2 (.pth) model weights
├── data/
│   ├── SMD/                      # Server Machine Dataset (38 features)
│   ├── MSL/                      # Mars Science Laboratory (55 features)
│   ├── SMAP/                     # Soil Moisture Active Passive (25 features)
│   └── CREDIT_CARD/              # Preprocessed Credit Card .npy arrays
├── data_factory/
│   └── data_loader.py            # Custom Dataset loaders (SMD, MSL, SMAP, CREDIT_CARD)
├── model/                        # TransformerVar & Gated Memory architecture
├── prepare_nasa.py               # Converts raw Kaggle NASA SMAP/MSL files into .npy arrays
├── preprocess_credit_card.py     # Feature engineering & normalization for Credit Card data
├── solver.py                     # Two-stage training, K-Means init, and evaluation logic
└── main.py                       # CLI entry point

```

---

## ⚙️ Installation & Requirements

* Python 3.10+ (Tested on Python 3.13)
* PyTorch
* NumPy, Pandas, Scikit-Learn, Matplotlib

```bash
pip install torch numpy pandas scikit-learn matplotlib

```

---

## 🚀 Usage

MEMTO uses a **two-stage training workflow** followed by evaluation:

1. **Phase 1 (`--mode train`):** Trains the base Transformer encoder-decoder (`first_train`).
2. **Phase 2 (`--mode memory_initial`):** Runs K-Means clustering on latent representations to initialize the 128 memory prototypes and fine-tunes the full network (`second_train`).
3. **Evaluation (`--mode test`):** Computes anomaly scores against the percentile threshold (`--anormly_ratio`).

---

### 1. Server Machine Dataset (`SMD` — `38` channels)

#### Expected Data Files (`./data/SMD/`)

* `SMD_train.npy` — shape `(708405, 38)`
* `SMD_test.npy` — shape `(708420, 38)`
* `SMD_test_label.npy` — shape `(708420,)`

#### Run SMD Pipeline

```bash
# Phase 1 Training
python main.py --mode train --dataset SMD --data_path ./data/SMD --model_save_path ./checkpoints --device cpu --input_c 38 --output_c 38 --win_size 100 --lr 1e-4 --num_epochs 10

# Memory Initialization & Phase 2 Training
python main.py --mode memory_initial --dataset SMD --data_path ./data/SMD --model_save_path ./checkpoints --device cpu --input_c 38 --output_c 38 --win_size 100 --lr 1e-4 --num_epochs 10 --memory_initial True --phase_type second_train

# Evaluation
python main.py --mode test --dataset SMD --data_path ./data/SMD --model_save_path ./checkpoints --device cpu --input_c 38 --output_c 38 --win_size 100 --anormly_ratio 0.5 --phase_type second_train

```

---

### 2. NASA Telemetry Benchmarks (`MSL` & `SMAP`)

#### Prepare Dataset

Download the [NASA SMAP/MSL Anomaly Detection Dataset](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl), extract it, and run:

```bash
python prepare_nasa.py

```

This generates:

* `data/MSL/`: `MSL_train.npy` `(58317, 55)`, `MSL_test.npy` `(73729, 55)`, `MSL_test_label.npy` `(73729,)`
* `data/SMAP/`: `SMAP_train.npy` `(135183, 25)`, `SMAP_test.npy` `(427617, 25)`, `SMAP_test_label.npy` `(427617,)`

#### Run MSL (`55` channels)

```bash
# Phase 1 Training
python main.py --mode train --dataset MSL --data_path ./data/MSL --model_save_path ./checkpoints --device cpu --input_c 55 --output_c 55 --win_size 100 --lr 1e-4 --num_epochs 10

# Memory Initialization & Phase 2 Training
python main.py --mode memory_initial --dataset MSL --data_path ./data/MSL --model_save_path ./checkpoints --device cpu --input_c 55 --output_c 55 --win_size 100 --lr 1e-4 --num_epochs 10 --memory_initial True --phase_type second_train

# Evaluation
python main.py --mode test --dataset MSL --data_path ./data/MSL --model_save_path ./checkpoints --device cpu --input_c 55 --output_c 55 --win_size 100 --anormly_ratio 1.0 --phase_type second_train

```

#### Run SMAP (`25` channels)

```bash
# Phase 1 Training
python main.py --mode train --dataset SMAP --data_path ./data/SMAP --model_save_path ./checkpoints --device cpu --input_c 25 --output_c 25 --win_size 100 --lr 1e-4 --num_epochs 10

# Memory Initialization & Phase 2 Training
python main.py --mode memory_initial --dataset SMAP --data_path ./data/SMAP --model_save_path ./checkpoints --device cpu --input_c 25 --output_c 25 --win_size 100 --lr 1e-4 --num_epochs 10 --memory_initial True --phase_type second_train

# Evaluation
python main.py --mode test --dataset SMAP --data_path ./data/SMAP --model_save_path ./checkpoints --device cpu --input_c 25 --output_c 25 --win_size 100 --anormly_ratio 1.0 --phase_type second_train

```

---

### 3. Credit Card Fraud Detection (`CREDIT_CARD`)

#### Prepare Dataset

Place the raw CSV at `./data/CREDIT_CARD/credit_card.csv` and run:

```bash
python preprocess_credit_card.py

```

Key preprocessing adaptations:

* Filters training data strictly to normal transactions (`is_fraud == 0`) for unsupervised prototype learning.
* Drops non-behavioral IDs (`cc_num`, `unix_time`, `trans_num`) and applies `StandardScaler` fit strictly on training data.
* Engineers domain-specific fraud signals (`distance_km`, `log_amt`, `amt_cat_zscore`, cyclic hour encodings, and frequency encodings).

#### Run Credit Card Pipeline (`--win_size 1`)

```bash
# Phase 1 Training
python main.py --mode train --dataset CREDIT_CARD --data_path ./data/CREDIT_CARD --model_save_path ./checkpoints --device cpu --input_c 16 --output_c 16 --win_size 1 --lr 1e-4 --num_epochs 5

# Memory Initialization & Phase 2 Training
python main.py --mode memory_initial --dataset CREDIT_CARD --data_path ./data/CREDIT_CARD --model_save_path ./checkpoints --device cpu --input_c 16 --output_c 16 --win_size 1 --lr 1e-4 --num_epochs 5 --memory_initial True --phase_type second_train

# Full Test Set Evaluation (259,335 rows)
python main.py --mode test --dataset CREDIT_CARD --data_path ./data/CREDIT_CARD --model_save_path ./checkpoints --device cpu --input_c 16 --output_c 16 --win_size 1 --batch_size 256 --anormly_ratio 1.0 --phase_type second_train

```

---

## 📊 Key Experimental Findings

| Dataset Type | Channels | Window Size | Evaluation Mode | Characteristics |
| --- | --- | --- | --- | --- |
| **SMD** | `38` | `100` | Segment Point-Adjustment | Server telemetry metrics (`~4.2%` anomaly rate); continuous multi-step outages. |
| **MSL** | `55` | `100` | Segment Point-Adjustment | Mars rover sensor channels (`~10.5%` anomaly rate); continuous trajectory anomalies. |
| **SMAP** | `25` | `100` | Segment Point-Adjustment | Satellite telemetry (`~12.8%` anomaly rate); periodic waveform disruptions. |
| **Credit Card Fraud** | `16` | `1` | Strict Point-Wise (`step=1`) | Isolated single-row events across interleaved users (`~0.52%` fraud rate); requires feature scaling and `win_size=1` to avoid cross-customer sequence mixing. |

---

## 📚 Citation & Acknowledgements

Based on the MEMTO paper:

> **MEMTO: Memory-guided Transformer for Multivariate Time-Series Anomaly Detection**
> Junho Song, Keonwoo Kim, Jeonglyul Oh, Sungzoon Cho — *NeurIPS 2023*

```

```
