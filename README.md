# 🫀 MLOps & Deep Learning: ECG Myocardial Infarction Detector

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

An End-to-End MLOps pipeline designed to detect **Myocardial Infarctions (MI)** analyzing 12-lead raw Electrocardiogram (ECG) waveforms from the massive PTB-XL database (`~17,000` samples). The project covers Data Engineering (SQLite relation schemas), Deep Learning (State-of-the-Art 1D CNN heuristics), and Deployment (MLflow Experiment tracking).

## 🚀 Architecture & State-of-the-Art (SOTA) Heuristics
Unlike simple tabular models, sequential temporal biological signals require robust tactics. This repository was tuned with the following industry-standard approaches:

1. **Dimensional Modeling (Star-Schema)**: The raw `.csv` reports from PhysioNet were normalized into relational SQL Tables separating Patients, SCP Reference Codes, and ECG Fact diagnostics.
2. **Battling Imbalanced Data (`pos_weight`)**: Myocardial Infarctions are physically a minority class in the general population. We mathematically enforce a penalty factor on False Negatives using `nn.BCEWithLogitsLoss`, compelling the network to prioritize detecting rare diseases.
3. **ROC_AUC Mastery**: Naive Accuracy relies on majority-guessing. This pipeline benchmarks strictly against **Area Under the ROC Curve (AUROC)**, the golden standard in the Medical AI sector.
4. **Regularization & Stability**: Feature stabilization using `BatchNorm1d` and early overfitting prevention via `Dropout(0.3)`.
5. **Dynamic LR Allocation**: Integrated `ReduceLROnPlateau` scheduler to refine learning paths progressively upon stagnated validation epochs.

## 📂 Repository Structure

```text
├── data/                       # Ignored by git (.gitignore payload blindage)
│   ├── raw/                    # Raw 100Hz Signal arrays (.dat, .hea format)
│   └── processed/              # SQL artifacts (ptbxl.db)
├── sql/
│   └── schema.sql              # Relational database layout mapping PTB-XL
├── src/
│   ├── build_database.py       # SQL Engine ingestion script
│   └── train_mi_detector.py    # PyTorch + MLflow Orchestration Script
├── .gitignore                  
├── requirements.txt
└── README.md
```

## 🛠️ Reproduction Guide (Local or VLAB GPU)

Deploying the diagnostic model requires the base PTB-XL signal payload and pipeline dependencies:

**1. Clone the environment:**
```bash
git clone https://github.com/Valdirff/ecg-diagnostic-physionet.git
cd ecg-diagnostic-physionet
pip install -r requirements.txt
```

**2. Hydrate the Data Lake (PhysioNet payload):**
Because 1.7GB of physical waveforms natively break GitHub storage boundaries and CI/CD pipelines, you must populate the `data/raw` folder by grabbing the ZIP from the MIT servers directly:
```bash
# If using a high-speed Linux cluster/VLAB (Downloads in seconds):
wget -c "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
unzip -q ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip -d data/raw/
mv data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/* data/raw/
```
*(If running on a simple Windows Local Machine, just use your standard Web Browser to download the Zip file and paste it inside your `data/raw` folder).*

**3. Architect SQL Database:**
Extracts relational structures bridging diagnoses mapping to actual ECG array paths.
```bash
python src/build_database.py
```

**4. Train and Orchestrate 1D-CNN (PyTorch):**
The script will auto-detect GPU (Nvidia Cuda / H100) or fallback defensively to CPU. Watch the Model adapt Loss limits per epoch.
```bash
python src/train_mi_detector.py
```

## 📊 MLOps Dashboarding (MLflow)
Once training finishes silently parsing thousands of heartbeats, observe the mathematical visualization by spinning up the tracking UI locally in your console:
```bash
mlflow ui
```
You can now access your Graphical Interface at `http://127.0.0.1:5000` inside your Web Browser to oversee metrics, architectures logged, and serialized PyTorch models safely registry-ready.
