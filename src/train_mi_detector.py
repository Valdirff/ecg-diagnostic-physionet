"""
Project: ECG Myocardial Infarction Detector (MLOps)
Description: 1D-CNN for MI Detection from 12-lead ECG signals with SQL & MLflow.
Author: Valdir (Research Engine Style)
"""

import os
import copy
import sqlite3
import torch
import numpy as np
import pandas as pd
import wfdb
import mlflow
import mlflow.pytorch
from pathlib import Path
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, 
                             recall_score, confusion_matrix)

# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
EPOCHS       = 50
BATCH_SIZE   = 128
LR           = 1e-3
ES_PATIENCE  = 7
THRESHOLD    = 0.5
DEV_MODE     = False  # Set to True for fast debugging with fewer samples
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# ==============================================================================
# 1. SQL DATA ENGINEERING LAYER
# ==============================================================================
def extract_dataset_from_db(db_path: str) -> pd.DataFrame:
    """
    Extracts high-quality ECG metadata from SQLite.
    Maps diagnostic SCP-codes to a binary MI target (1=Infarction, 0=Normal).
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}. Please run build_database.py first.")
        
    conn = sqlite3.connect(db_path)
    query = """
    SELECT r.filename_hr as path,
           MAX(CASE WHEN d.diagnostic_class = 'MI' THEN 1 ELSE 0 END) as target_mi,
           r.strat_fold
    FROM ecg_records r
    JOIN ecg_scp_diagnoses map ON r.ecg_id = map.ecg_id
    JOIN scp_codes d ON map.scp_code = d.scp_code
    WHERE d.is_diagnostic = 1
    GROUP BY r.ecg_id;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ==============================================================================
# 2. SIGNAL PROCESSING & DATASET
# ==============================================================================
class PTBXL_ECGDataset(Dataset):
    """
    Handles loading of 12-lead .dat files with per-lead normalization.
    Implements error tolerance for cloud-synced files (OneDrive).
    """
    def __init__(self, df: pd.DataFrame, data_dir: Path):
        self.paths    = df['path'].values
        self.labels   = df['target_mi'].values
        self.data_dir = data_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        while True:
            record_path = (self.data_dir / self.paths[idx]).as_posix()
            try:
                # Load signal (typically 12 channels x 1000 samples)
                signal, _ = wfdb.rdsamp(record_path)
                break
            except (OSError, Exception):
                # Fault Tolerance: Skip truncated/corrupted files
                idx = (idx + 1) % len(self.labels)

        # Standard Z-Score normalization per lead
        signal = (signal - np.mean(signal, axis=0)) / (np.std(signal, axis=0) + 1e-8)
        
        # Pytorch expected: (Channels, Sequence_Length)
        signal_tensor = torch.tensor(signal.T, dtype=torch.float32)
        label_tensor  = torch.tensor([self.labels[idx]], dtype=torch.float32)
        
        return signal_tensor, label_tensor

# ==============================================================================
# 3. DEEP LEARNING ARCHITECTURE (1D-CNN)
# ==============================================================================
class ECG_CNN(nn.Module):
    """
    Progressive 1D-CNN Architecture inspired by medical signal hierarchies.
    Filters: 32 -> 64 -> 128 -> 256.
    """
    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            # Block 1 - Early temporal features
            nn.Conv1d(12, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2 - Waveform morphology
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 3 - Complex arrhythmic signatures
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 4 - Deep feature abstraction
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        features = self.conv_blocks(x).squeeze(-1)
        return self.classifier(features)

# ==============================================================================
# 4. TRAINING & EVALUATION LOGIC
# ==============================================================================
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out   = model(X)
            total_loss += criterion(out, y).item()
            probs = torch.sigmoid(out).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(y.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    auc      = roc_auc_score(all_targets, all_preds)
    return avg_loss, auc, np.array(all_preds), np.array(all_targets)

def main():
    # Paths setup
    base_dir = Path(__file__).parent.parent
    db_path  = base_dir / "data" / "processed" / "ptbxl.db"
    raw_dir  = base_dir / "data" / "raw"
    model_save_path = base_dir / "outputs" / "models" / "best_mi_detector.pth"
    os.makedirs(model_save_path.parent, exist_ok=True)

    print("--- [ ECG MLOps Pipeline: MI Detection ] ---")
    df = extract_dataset_from_db(db_path)

    # Stratified split: Train (1-8), Val (9), Test (10)
    train_df = df[df['strat_fold'] <= 8].reset_index(drop=True)
    val_df   = df[df['strat_fold'] == 9].reset_index(drop=True)
    test_df  = df[df['strat_fold'] == 10].reset_index(drop=True)

    print(f"Dataset Split -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Loaders
    train_loader = DataLoader(PTBXL_ECGDataset(train_df, raw_dir), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(PTBXL_ECGDataset(val_df,   raw_dir), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(PTBXL_ECGDataset(test_df,  raw_dir), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation Device: {device.type.upper()}\n")

    model = ECG_CNN().to(device)
    
    # Checkpoint recovery
    if model_save_path.exists():
        print(f"Restoring weights from {model_save_path}...")
        model.load_state_dict(torch.load(model_save_path, map_location=device))

    # Loss with dynamic class weighting
    num_pos = train_df['target_mi'].sum()
    pos_weight = torch.tensor([(len(train_df)-num_pos)/num_pos]).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # MLflow Tracking
    mlflow.set_experiment("ECG_MI_Detector_V1")
    with mlflow.start_run():
        mlflow.log_params({"architecture": "1D-CNN-4Block", "batch_size": BATCH_SIZE, "lr": LR})
        
        best_auc = 0.0
        patience = 0

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_val_loss, val_auc, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val AUROC: {val_auc:.4f}")

            # Best Model Persistence
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), model_save_path)
                patience = 0
            else:
                patience += 1
                if patience >= ES_PATIENCE:
                    print("Early Stopping Triggered.")
                    break

        # Final Evaluation
        model.load_state_dict(torch.load(model_save_path))
        print("\n--- [ Held-out Test Set Results ] ---")
        _, test_auc, probs, targets = evaluate(model, test_loader, criterion, device)
        
        preds = (probs >= THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
        
        print(f"Test AUROC: {test_auc:.4f}")
        print(f"Sensitivity: {tp/(tp+fn):.4f} | Specificity: {tn/(tn+fp):.4f}")
        
if __name__ == "__main__":
    main()
