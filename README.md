# ECG Myocardial Infarction Detector (MLOps)

Deep Learning pipeline for automated detection of Myocardial Infarction (MI) from 12-lead ECG signals, integrating a SQL-based data engineering layer with a high-performance 1D Convolutional Neural Network (CNN) and full experiment tracking via MLflow.

The dataset is derived from the **PTB-XL** clinical database, where raw voltage signals are mapped to diagnostic metadata stored in a relational SQLite structure. This project demonstrates a production-ready MLOps workflow: from raw signal extraction via SQL queries to model registry and deployment-ready artifacts.

---

## Background

Myocardial Infarction (MI) is a critical cardiovascular event where rapid and accurate ECG interpretation can be life-saving. While the PTB-XL dataset provides a rich ground truth, most implementations either ignore the complex relational metadata (SCP-codes) or fail to provide a scalable MLOps infrastructure.

This project was built to bridge that gap. We implement a **Stratified 10-Fold cross-validation strategy** (using folds 1-8 for training, 9 for validation, and 10 for independent testing) to ensure clinical generalizability. The core objective: automate the binary classification of MI with high sensitivity, as missing a positive case (False Negative) is the highest risk in clinical settings.

---

## Project Structure

```text
Project_SQL_MLOps_ECG/
│
├── README.md                           # Project Documentation
├── .gitignore                          # Excludes raw data/models
├── requirements.txt                    # Production dependencies
│
├── data/                               # Clinical Data (Local only)
│   ├── raw/                            # .dat / .hea PhysioNet files
│   └── processed/                      # ptbxl.db (SQLite Metadata)
│
├── sql/
│   └── schema.sql                      # DDL for the clinical database
│
├── src/
│   ├── build_database.py              # ETL: CSV/Records to SQLite
│   └── train_mi_detector.py           # Main DL & MLOps Pipeline
│
├── outputs/                            # Generated at runtime
│   ├── models/                         # Local .pth checkpoints
│   └── logs/                           # Training telemetry
│
├── notebooks/                          # Exploratory Signal Analysis
└── mlruns/                             # MLflow Tracking UI data
```

---

## Pipeline

```text
PTB-XL Raw Records (.dat) + Metadata (.csv)
    │
    ▼  src/build_database.py (ETL)
SQLite Database (ptbxl.db)
    │
    ▼  SQL Extraction (MI Diagnostic Mapping)
12-Lead Normalized Signal Tensors
    │
    ▼  src/train_mi_detector.py (1D-CNN)
─────────────────────────────────────────────────────────
MLflow Tracking Registry (Params, Metrics, Artifacts)
    │
    ▼  Best Model Checkpoint (.pth)
Clinical Evaluation (AUROC, Sensitivity, Specificity)
```

---

## Results Summary

### Performance Metrics (Held-out Test Set)

The model was evaluated on Fold 10 (2,158 unseen exams) after triggering **Early Stopping** at Epoch 31:

| Metric | Value |
|---|---|
| **AUROC** | **0.9242** |
| **Sensitivity (Recall)** | **86.91%** |
| **Specificity** | **80.35%** |
| **Accuracy** | **82.02%** |
| **F1-Score** | **71.13%** |

→ **High Clinical Sensitivity:** The model successfully identified 478 out of 550 real MI cases, crucial for screening applications.

### Confusion Matrix (Test Set)
| | Predicted Normal | Predicted MI |
|---|---|---|
| **Actual Normal** | 1292 (TN) | 316 (FP) |
| **Actual MI** | 72 (FN) | 478 (TP) |

---

## Key Design Choices (and Why)

**Why SQL for metadata?**
In a real hospital environment, ECG metadata isn't in a CSV; it's in a relational database. By using SQLite, we simulate a production data engineering layer, allowing for complex diagnostic filtering (e.g., isolating MI via SCP-code mappings) directly during the data loading phase.

**Why 4-Block 1D-CNN?**
ECG signals are temporal. The progressive filter increase (32→64→128→256) allows the network to capture low-level rhythmic features in early layers and complex morphological signatures of ischemia (like ST-segment elevation) in deeper layers.

**Why BCEWithLogitsLoss with Positional Weighting?**
MI cases are the minority in the dataset. We calculate a dynamic `pos_weight` during training to penalize false negatives more heavily, forcing the model to prioritize sensitivity over simple accuracy.

**Why MLflow Integration?**
Instead of manually tracking hyperparameters, MLflow logs the architecture, learning rate, and batch size automatically. This ensures full reproducibility and allows us to promote the "Best Model" to a production registry with a single click.

---

## How to Reproduce

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare Data**:
   Ensure PTB-XL records are in `data/raw/` and run the ETL script:
   ```bash
   python src/build_database.py
   ```
3. **Run Pipeline**:
   ```bash
   python src/train_mi_detector.py
   ```

---

## Academic References

- **Wagner, P. et al. (2020).** PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*.
- **Kiranyaz, S. et al. (2019).** Real-time Patient-specific ECG Classification via 1D Convolutional Neural Networks. *IEEE Transactions on Biomedical Engineering*.
- **MLflow Documentation:** https://mlflow.org/docs/latest/index.html
