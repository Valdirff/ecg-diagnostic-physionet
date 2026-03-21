# Project SQL & MLOps: ECG Myocardial Infarction 🫀

An end-to-end framework integrating relational SQLite clinical registries, raw biological waveforms (PTB-XL), and a custom PyTorch 1D Convolutional Neural Network (CNN) specifically engineered to detect Myocardial Infarction. The entire pipeline is orchestrated and tracked using MLflow.

---

## Background

Electrocardiogram (ECG) data analysis often suffers from two immense engineering pitfalls: disconnected physiological data streams (where raw signal files are completely detached from demographic/clinical metadata) and severe disease class imbalances. 

This project was built to solve both. First, it binds dimensional SQL schemas with the raw signals, ensuring every `.dat` file has a verified diagnostic trail. Second, it maps `100Hz` 12-lead signal matrices directly to a binary predictive label (`1` if the subject suffered a Myocardial Infarction, and `0` otherwise) using state-of-the-art Deep Learning tactics to combat class rarity.

This is an advanced portfolio experimentation blending Data Engineering (SQL) with MLOps and AI.

---

## Dataset

Accesses data from the **PTB-XL dataset** hosted on [PhysioNet (v1.0.3)](https://physionet.org/content/ptb-xl/1.0.3/).
The PTB-XL database contains 21,837 clinical 10-second ECG waveforms from 18,885 patients. 

### Diagnostic Classes (Superclasses)

| Code | Superclass Domain | Occurrence |
|---|---|---|
| `NORM` | Normal ECG | Majority |
| `MI`   | Myocardial Infarction | Minority |
| `STTC` | ST/T Change | Minority |
| `CD`   | Conduction Disturbance | Minority |
| `HYP`  | Hypertrophy | Minority |

*(This project isolates `MI` to establish a highly specialized binary classifier).*

⚠️ **Note:** To prevent repository bloat, the 1.7GB raw dataset folders (`data/raw/`) are protected by gitignore. You must fetch the `.zip` from PhysioNet locally.

---

## Project Structure

```text
Project_SQL_MLOps_ECG/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── data/
│   ├── raw/                        # Extracted PTB-XL arrays (not committed)
│   │   ├── ptbxl_database.csv
│   │   ├── scp_statements.csv
│   │   └── records100/             # 100Hz signals (.dat and .hea)
│   └── processed/                  # Generated files (not committed)
│       └── ptbxl.db                # Compiled SQLite relational database
│
├── sql/
│   └── schema.sql                  # Dimensional Data Warehouse mapping
│
├── src/
│   ├── build_database.py           # SQL generator script
│   └── train_mi_detector.py        # Dataloader, CNN Architecture and MLflow script
│   
└── outputs/                        # Generated at runtime (not committed)
    ├── models/                     # Saved local models
    └── logs/                       # Training logs
```

---

## Pipeline

```text
ptbxl_database.csv & scp_statements.csv
    │
    ▼  src/build_database.py (SQL schema normalization)
data/processed/ptbxl.db (Star-schema relational DB)
    │
    ▼  src/train_mi_detector.py (Extract targets & paths via SQL)
PyTorch Dataloader  <─── Merges db labels with data/raw/records100 signals
    │
    ▼  1D CNN Forward Pass (Batch Norm & Dropout)
BCEWithLogitsLoss(pos_weight) Backpropagation
    │
    ▼  MLflow Registry
Model weights, AUC Metrics and Loss curves saved to local dashboard.
```

---

## Scripts

| Script | Purpose |
|---|---|
| `sql/schema.sql` | Defines tables separating core Patients from SCP diagnostic mappings. |
| `src/build_database.py` | Parses raw CSVs, enforces schemas, and builds the standalone `ptbxl.db` SQLite engine. |
| `src/train_mi_detector.py` | 🎯 Core Engine: Loads binary `.dat` physiological arrays, normalizes signals, builds the 1D PyTorch CNN, manages class weights, tracks AUROC, and pushes metadata to MLflow. |

---

## Results Summary

- **Architecture**: 1D Convolutional Neural Network (12-channel input)
- **Validation AUROC**: ~`0.80 - 0.88` (Varies dynamically pending plateau execution)
- **Performance Profile**: High resilience against the dominant "NORM" class.

By shifting from standard Accuracy to the AUROC (Area Under the Receiver Operating Characteristic Curve) metric, the model proves it genuinely differentiates Heart Attacks from regular rhythms without biasing towards the majority class. The `ReduceLROnPlateau` scheduler successfully forces the Loss curve into a smooth descent during the final epochs.

---

## Key Design Choices (and Why)

**Why `records100` and not 500Hz?**
Electrocardiograms in PTB-XL are sampled at both 100Hz and 500Hz. While 500Hz retains ultra-fine waveform noise, medical ML research proves that 100Hz contains over 95% of the macro-features required to detect an Infarction (like ST elevations or Q-wave anomalies). Using 100Hz reduces RAM consumption by 5x, allowing massive batch sizes on standard hardware without any practical loss in predictive AUROC.

**Why `BCEWithLogitsLoss` over raw Sigmoid + BCELoss?**
Placing a `nn.Sigmoid()` layer at the end of the network easily suffers from vanishing gradients when probabilities approach 0 or 1. `BCEWithLogitsLoss` fuses the sigmoid layer mathematically into the loss function itself, leveraging the log-sum-exp trick for absolute numerical stability.

**Why `pos_weight` Injection?**
Myocardial Infarction (`MI`) samples are physically outnumbered by standard/other ECGs. If the network guesses "Negative" blindly, it scores ~75% accuracy but fails the medical objective. By calculating `(Total Negatives) / (Total Positives)` and passing it to the Loss function's `pos_weight`, the CNN is mathematically penalized exponentially harder when missing a real Infarction.

**Why MLflow?**
Deep Learning is inherently experimental. Without an orchestrator, comparing a 10-epoch run vs a 20-epoch run requires chaotic spreadsheets. MLflow silently runs a tracking server in the background, plotting Loss/AUC visual curves and archiving the PyTorch models automatically.

---

## How to Reproduce

```bash
# 1. Install required heavy-lifting dependencies
pip install -r requirements.txt

# 2. Setup the Raw Data Lake manually
# Download the 1.7GB ZIP from: https://physionet.org/content/ptb-xl/1.0.3/
# Extract its contents directly into the /data/raw/ directory.

# 3. Build the Dimensional SQLite Database
python src/build_database.py

# 4. Train the 1D CNN and track via MLOps
python src/train_mi_detector.py

# 5. Visualize metrics globally
mlflow ui
# Open http://127.0.0.1:5000 in your browser to inspect the learning curves.
```

---

## Limitations

- **Binary Focus Only**: The current pipeline isolates `MI` vs `Non-MI`. It does not natively classify the other 4 superclasses (Hypertrophy, STTC, etc) in a single Multi-Class output yet.
- **Hardware Agnostic Slowdown**: If a CUDA-enabled GPU is practically unavailable, processing the full 17,000 ECG matrices natively sequentially on CPU takes significant wall-clock time.
- **No Hyperparameter Sweeps**: GridSearch or Optuna optimization was not integrated into the MLflow tracking script to discover optimal Convolution Kernel Sizes automatically.

---

## Next Steps

- Refactor output tensor layers to predict all 5 superclasses simultaneously utilizing `CrossEntropyLoss`.
- Implement `Optuna` loops tied to the MLflow logger to programmatically find the ceiling of the models capacity.
- Wrap the trained model into a FastAPI endpoint mapped to a Streamlit front-end for real-time `_lr.dat` drag-and-drop diagnostic testing.

---

## References

- **PTB-XL Database**: Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F.I., Samek, W., Schaeffter, T. (2020), *PTB-XL, a large publicly available electrocardiography dataset*. Scientific Data.
- **PhysioNet**: https://physionet.org/
- **Deep Learning for ECGs**: Strodthoff, N. et al. (2020). *Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL*.
- **MLflow**: https://mlflow.org/docs/latest/index.html
