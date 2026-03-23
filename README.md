<p align="center">
  <img src="docs/banner_12leads.png" width="100%" alt="ECG MLOps Banner" />
</p>

# 🩺 ECG Myocardial Infarction Detector (MLOps Pipeline)
> **Relational Data Engineering + Distributed Deep Learning + Enterprise Experiment Tracking**

This repository implements a high-performance **1D-CNN pipeline** designed for the automated detection of **Myocardial Infarction (MI)** from 12-lead ECG signals. Leveraging the **PTB-XL** dataset (21,837 signals), the project features a custom **SQL-based Data Engineering layer**, providing robust clinical diagnostic filtering and scalable signal processing.

---

## 🔬 Clinical Diagnosis: STEMI Architecture

In clinical practice, identifying a **Myocardial Infarction (MI)** requires the detection of subtle ST-segment elevation (STEMI). Our model is designed to assist in **automated screening**, prioritizing high **Sensitivity** to ensure clinically critical cases are not missed.

<p align="center">
  <img src="docs/mi_vs_normal_trace.png" width="85%" alt="ECG Comparison" />
</p>
<p align="center">
  <em>Figure 1: Comparison between a Normal Sinus Rhythm and a real Myocardial Infarction signal from the PTB-XL database.</em>
</p>

---

## 🏗️ Technical Pipeline & MLOps

```text
SIGNAL INGESTION       DATA ENGINEERING        DL ARCHITECTURE        MLOPS REGISTRY
PhysioNet / PTB-XL ──▶ SQLite Metadata  ──▶   4-Block 1D-CNN  ──▶   MLflow Backend
(Raw .dat Files)     (Relational ETL)      (BCE + PosWeight)      (Best .pth Model)
```

1.  **SQL-Driven ETL:** Unlike traditional CSV-based loading, our `build_database.py` maps complex clinical SCP-metadata into a queryable SQLite structure for advanced statistical filtering.
2.  **Neural Hierarchy:** 4-stage 1D-Convolutional blocks (Filters: 32 → 256) with Batch Normalization and Dropout for robust feature extraction from temporal waveforms.
3.  **Experiment Lifecycle:** Integrated with **MLflow** for real-time tracking of Loss/AUROC curves and versioning of the finalized weights.

---

## 📈 Performance & Results

The final model was evaluated on the **Hold-out Test Set (Fold 10)**, consisting of 2,158 unseen clinical exams.

### 📋 Metrics Summary
| Metric | Value | Clinical Interpretation |
|---|---|---|
| **AUROC** | **0.9242** | Excellent diagnostic reliability. |
| **Sensitivity (Recall)** | **86.91%** | **478/550 MI cases caught** (Primary screening goal). |
| **Specificity** | **80.35%** | Robust rejection of non-MI controls. |
| **F1-Score** | **71.13%** | High performance on imbalanced clinical data. |

<p align="center">
  <img src="docs/roc_curve.png" width="48%" />
  <img src="docs/confusion_matrix.png" width="48%" />
</p>
<p align="center">
  <em>Figure 2: (Left) ROC Curve showcasing an AUC of 0.9242. (Right) Confusion Matrix with clinical TP/FP/TN/FB metrics (YOLO-Style template).</em>
</p>

---

## 🛠️ Reproduction Guide

### 1. Requirements
```bash
git clone <repo_url>
pip install -r requirements.txt
```

### 2. Data Preparation
1.  Download **PTB-XL** from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/).
2.  Populate `data/raw/` with `.dat` and `.hea` files.
3.  Initialize the SQL Diagnostic Engine:
    ```bash
    python src/build_database.py
    ```

### 3. Pipeline Execution
```bash
# Start Training and MLOps Tracking
python src/train_mi_detector.py

# Launch Dashboard
mlflow ui
```

---

## 🔑 Key Engineering Differentials

*   **Weighted Loss Bias:** Uses `pos_weight` in BCE to account for clinical class imbalance (MI prevalence).
*   **Production Robustness:** The pipeline handles asynchronous SCP-metadata mapping and fragmented signal loading automatically.
*   **SQL Metadata Mapping:** Conversion of clinical SCP-codes into a relational schema for advanced medical researcher querying.

---

## 📜 Credits
- **Wagner, P. et al. (2020).** PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*.
- **Training Framework:** PyTorch Core with MLflow Integration.
