"""
🩺 ECG MI DETECTOR - TRAINING ENGINE V2.0 (Elite Edition)
- Reproducible Architecture (Deterministic Seeds)
- MLOps Integrated (MLflow + SQL ETL)
- Binary Myocardial Infarction Classification (Screening-biased)
"""
import os
import random
import sqlite3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wfdb
import mlflow
import mlflow.pytorch
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score
from pathlib import Path
import logging

# --- REPRODUCIBILITY SEEDING ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ Random seeds locked at {seed} for reproducibility.")

set_seed(42)

# --- CONFIGURATION ---
DATA_ROOT = Path("data/raw")
DB_PATH = Path("data/processed/ptbxl.db")
MODEL_SAVE_PATH = Path("outputs/models/best_mi_detector.pth")
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
RESUME_TRAINING = False # Flag controlada pelo usuario

# --- REUSABLE 1D CNN ARCHITECTURE ---
class ECGClassifier(nn.Module):
    """
    Standard 4-Block 1D CNN Architecture for Time-Series Workflow.
    Filters: 32 -> 64 -> 128 -> 256.
    Includes Batch Normalization and Dropout for robust regularization.
    """
    def __init__(self, num_leads=12):
        super(ECGClassifier, self).__init__()
        self.conv_blocks = nn.Sequential(
            self._make_block(num_leads, 32),
            self._make_block(32, 64),
            self._make_block(64, 128),
            self._make_block(128, 256)
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

# --- DATA GENERATION LAYER ---
class PTBXLDataset(Dataset):
    def __init__(self, file_paths, labels, root_dir):
        self.file_paths = file_paths
        self.labels = labels
        self.root_dir = root_dir

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = str(self.root_dir / self.file_paths[idx])
        try:
            # Using hr (high resolution) files as default
            signal, _ = wfdb.rdsamp(path)
            # Standardize and Transpose (Leads, Timesteps)
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            signal = torch.FloatTensor(signal).transpose(0, 1)
            target = torch.FloatTensor([self.labels[idx]])
            return signal, target
        except Exception as e:
            logging.error(f"⚠️ Error loading file {path}: {e}")
            # Returns dummy zero signal to prevent pipeline crash (Better to log and skip in production)
            return torch.zeros((12, 1000)), torch.FloatTensor([0.0])

def get_clinical_data(db_path, fold):
    """
    Queries SQLite metadata using SQL diagnostic class mapping.
    Folds: 1-8 (Train), 9 (Val), 10 (Test).
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    query = """
    SELECT r.filename_hr, (d.diagnostic_class = 'MI') as label
    FROM ecg_records r
    JOIN ecg_scp_diagnoses map ON r.ecg_id = map.ecg_id
    JOIN scp_codes d ON map.scp_code = d.scp_code
    WHERE r.strat_fold IN ({})
    """
    
    # Train
    cur.execute(query.format(", ".join(map(str, [1,2,3,4,5,6,7,8]))))
    train_data = cur.fetchall()
    # Val
    cur.execute(query.format("9"))
    val_data = cur.fetchall()
    # Test
    cur.execute(query.format("10"))
    test_data = cur.fetchall()
    
    conn.close()
    return train_data, val_data, test_data

# --- PIPELINE ENGINE ---
def main():
    print("🚀 Initializing ECG MI Detection Training Pipeline...")
    train_raw, val_raw, test_raw = get_clinical_data(DB_PATH, fold=10)
    
    train_loader = DataLoader(PTBXLDataset([r[0] for r in train_raw], [r[1] for r in train_raw], DATA_ROOT), 
                             batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PTBXLDataset([r[0] for r in val_raw], [r[1] for r in val_raw], DATA_ROOT), 
                           batch_size=BATCH_SIZE)
    test_loader = DataLoader(PTBXLDataset([r[0] for r in test_raw], [r[1] for r in test_raw], DATA_ROOT), 
                            batch_size=BATCH_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECGClassifier().to(device)
    
    # Optional Resume
    if RESUME_TRAINING and MODEL_SAVE_PATH.exists():
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print(f"🔄 Resuming from checkpoint: {MODEL_SAVE_PATH}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device)) # clinical imbalance handling
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    with mlflow.start_run(run_name="ECG_MI_Detector_V2"):
        mlflow.log_params({"lr": LEARNING_RATE, "batch_size": BATCH_SIZE, "arch": "4-block-1DCNN"})
        
        best_auc = 0.0
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for signals, targets in train_loader:
                signals, targets = signals.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(signals)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Validation Step
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for signals, targets in val_loader:
                    signals = signals.to(device)
                    outputs = torch.sigmoid(model(signals))
                    all_preds.extend(outputs.cpu().numpy())
                    all_labels.extend(targets.numpy())
            
            val_auc = roc_auc_score(all_labels, all_preds)
            mlflow.log_metric("val_auc", val_auc, step=epoch)
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")
            
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                mlflow.pytorch.log_model(model, "model")
                print("✨ New BEST Model Saved!")

        # --- FINAL EXTERNAL EVALUATION (FOLD 10) ---
        print("\n--- TEST SET EVALUATION (FOLD 10) ---")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for signals, targets in test_loader:
                signals = signals.to(device)
                outputs = torch.sigmoid(model(signals))
                test_preds.extend(outputs.cpu().numpy())
                test_labels.extend(targets.numpy())
        
        y_pred_binary = (np.array(test_preds) > 0.5).astype(int)
        cm = confusion_matrix(test_labels, y_pred_binary)
        
        # Explicit metrics for README alignment
        final_auc = roc_auc_score(test_labels, test_preds)
        final_acc = accuracy_score(test_labels, y_pred_binary)
        final_f1 = f1_score(test_labels, y_pred_binary)
        sens = cm[1,1] / (cm[1,1] + cm[1,0]) # Sensitivity
        spec = cm[0,0] / (cm[0,0] + cm[0,1]) # Specificity
        
        print(f"Final AUROC:      {final_auc:.4f}")
        print(f"Final Accuracy:   {final_acc:.4f}")
        print(f"Sensitivity:      {sens:.4f}")
        print(f"Specificity:      {spec:.4f}")
        print(f"F1-Score:         {final_f1:.4f}")
        print(f"Confusion Matrix: TN:{cm[0,0]} FP:{cm[0,1]} | FN:{cm[1,0]} TP:{cm[1,1]}")

        # Metrics for MLflow
        mlflow.log_metrics({"test_auc": final_auc, "test_f1": final_f1, "sens": sens, "spec": spec})

if __name__ == "__main__":
    main()
