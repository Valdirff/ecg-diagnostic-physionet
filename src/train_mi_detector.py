import os
import copy
import sqlite3
import pandas as pd
import numpy as np
import wfdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch
from pathlib import Path
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,recall_score, confusion_matrix)

# Silence harmless MLflow Git warnings on Windows
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# ==========================================
# CONFIGURATION
# ==========================================
EPOCHS       = 50      # Train longer — early stopping will cut when optimal
BATCH_SIZE   = 128
LR           = 0.001
ES_PATIENCE  = 7       # Early Stopping: stop after 7 epochs without improvement
THRESHOLD    = 0.5     # Decision threshold for binary metrics

# ==========================================
# 1. SQL INTEGRATION (Data Engineering)
# ==========================================
def extract_dataset_from_db(db_path: str):
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

# ==========================================
# 2. ECG WAVE LOADING (Signal Processing)
# ==========================================
class ECGDataset(Dataset):
    def __init__(self, df, data_dir):
        self.paths  = df['path'].values
        self.labels = df['target_mi'].values
        self.data_dir = Path(data_dir)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        while True:
            record_path = (self.data_dir / self.paths[idx]).as_posix()
            try:
                signal, _ = wfdb.rdsamp(record_path)
                break
            except OSError:
                # MLOps Fault Tolerance: skip truncated / un-synced OneDrive files
                idx = (idx + 1) % len(self.labels)

        # Safe per-lead normalization
        signal = (signal - np.mean(signal, axis=0)) / (np.std(signal, axis=0) + 1e-8)
        signal_tensor = torch.tensor(signal.T, dtype=torch.float32)   # (12, T)
        label_tensor  = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return signal_tensor, label_tensor

# ==========================================
# 3. DEEP LEARNING ARCHITECTURE
# ==========================================
class ECG_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(12, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 4 — deeper feature extraction
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
            # No sigmoid — BCEWithLogitsLoss handles it numerically
        )

    def forward(self, x):
        features = self.conv_blocks(x).squeeze(-1)
        return self.classifier(features)

# ==========================================
# 4. EVALUATION HELPER
# ==========================================
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

# ==========================================
# 5. MLOps ORCHESTRATION
# ==========================================
def main():
    base_dir = Path(__file__).parent.parent
    db_path  = base_dir / "data" / "processed" / "ptbxl.db"
    raw_dir  = base_dir / "data" / "raw"
    model_save_path = base_dir / "outputs" / "models" / "best_mi_detector.pth"
    os.makedirs(model_save_path.parent, exist_ok=True)

    print("Extracting labels from SQL Database...")
    df = extract_dataset_from_db(db_path)

    # Stratified 3-way split: train (folds 1-8) | val (fold 9) | test (fold 10)
    train_df = df[df['strat_fold'] <= 8].reset_index(drop=True)
    val_df   = df[df['strat_fold'] == 9].reset_index(drop=True)
    test_df  = df[df['strat_fold'] == 10].reset_index(drop=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} exams")

    train_loader = DataLoader(ECGDataset(train_df, raw_dir), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(ECGDataset(val_df,   raw_dir), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(ECGDataset(test_df,  raw_dir), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Training on device: {device}\n")

    model = ECG_CNN().to(device)

    # ── Check for existing checkpoint ──────────────────────────────────
    if model_save_path.exists():
        print(f"Loading existing best weights from {model_save_path}...")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print("Model restored. Starting/Continuing training...")

    # Class imbalance weighting (MI is minority)
    num_pos    = train_df['target_mi'].sum()
    num_neg    = len(train_df) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # Cosine Annealing: smooth LR decay over full training budget
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    mlflow.set_experiment("ECG_Myocardial_Infarction_Detector")

    with mlflow.start_run():
        mlflow.log_params({
            "architecture": "CNN1D_4Block_BN_Dropout",
            "epochs_max":   EPOCHS,
            "early_stopping_patience": ES_PATIENCE,
            "learning_rate": LR,
            "scheduler": "CosineAnnealingLR",
            "pos_weight": round(pos_weight.item(), 3),
            "batch_size": BATCH_SIZE,
            "train_size": len(train_df),
            "val_size":   len(val_df),
            "test_size":  len(test_df),
        })

        # ── Early Stopping state ──────────────────────────────────────────
        best_auc        = 0.0
        best_weights    = None
        patience_counter = 0
        best_epoch      = 0

        for epoch in range(EPOCHS):
            # Training pass
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation pass
            avg_val_loss, val_auc, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            print(
                f"Epoch {epoch+1:02d}/{EPOCHS} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val AUROC: {val_auc:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss":   avg_val_loss,
                "val_auc":    val_auc,
                "lr":         current_lr,
            }, step=epoch)

            # ── Checkpoint best model ─────────────────────────────────────
            if val_auc > best_auc:
                best_auc      = val_auc
                best_weights  = copy.deepcopy(model.state_dict())
                torch.save(best_weights, model_save_path) # SALVAMENTO FÍSICO
                best_epoch    = epoch + 1
                patience_counter = 0
                print(f"  ✔ New best AUROC: {best_auc:.4f} — model saved to disk.")
            else:
                patience_counter += 1
                print(f"  ✗ No improvement ({patience_counter}/{ES_PATIENCE})")
                if patience_counter >= ES_PATIENCE:
                    print(f"\n⏹  Early Stopping triggered at epoch {epoch+1}.")
                    break

        # ── Restore best weights ──────────────────────────────────────────
        model.load_state_dict(best_weights)
        print(f"\n✅ Best model restored from epoch {best_epoch} (Val AUROC: {best_auc:.4f})")

        # ── Final evaluation on held-out TEST set ─────────────────────────
        print("\n── Test Set Evaluation ──────────────────────────────────────")
        _, test_auc, test_probs, test_targets = evaluate(model, test_loader, criterion, device)
        test_preds = (test_probs >= THRESHOLD).astype(int)

        precision = precision_score(test_targets, test_preds, zero_division=0)
        recall    = recall_score(test_targets, test_preds, zero_division=0)   # = Sensitivity
        f1        = f1_score(test_targets, test_preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(test_targets, test_preds).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy    = (tp + tn) / len(test_targets)

        print(f"  AUROC:       {test_auc:.4f}")
        print(f"  Accuracy:    {accuracy:.4f}")
        print(f"  Precision:   {precision:.4f}")
        print(f"  Sensitivity: {recall:.4f}  (Recall)")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  F1-Score:    {f1:.4f}")
        print(f"  Confusion Matrix → TP:{tp} | FP:{fp} | TN:{tn} | FN:{fn}")

        mlflow.log_metrics({
            "test_auc":         test_auc,
            "test_accuracy":    accuracy,
            "test_precision":   precision,
            "test_sensitivity": recall,
            "test_specificity": specificity,
            "test_f1":          f1,
            "best_val_auc":     best_auc,
            "best_epoch":       best_epoch,
        })

        # ── Push best model to MLflow Registry ───────────────────────────
        print("\nSaving Best Model to MLflow Registry...")
        mlflow.pytorch.log_model(model, artifact_path="ecg_best_model", pickle_module=copy)
        print("✅ Best model successfully pushed to MLOps Registry.")

if __name__ == "__main__":
    main()
