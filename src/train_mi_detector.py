import os
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
from sklearn.metrics import roc_auc_score

# Silence harmless MLflow Git warnings on Windows
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

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
        self.paths = df['path'].values
        self.labels = df['target_mi'].values
        self.data_dir = Path(data_dir)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        record_path = (self.data_dir / self.paths[idx]).as_posix()
        signal, metadata = wfdb.rdsamp(record_path)
        
        # PRO TACTIC 1: Safe Normalization
        signal = (signal - np.mean(signal, axis=0)) / (np.std(signal, axis=0) + 1e-8)
        
        signal_tensor = torch.tensor(signal.T, dtype=torch.float32)
        label_tensor = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return signal_tensor, label_tensor

# ==========================================
# 3. ARTIFICIAL INTELLIGENCE BUILD (Deep Learning)
# ==========================================
class ECG_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), # PRO TACTIC 2: Batch Normalization stabilizes waves
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) 
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(0.3), # PRO TACTIC 3: Dropout avoids Overfitting early
            nn.ReLU(),
            nn.Linear(64, 1)
            # PRO TACTIC 4: No sigmoid here! We use BCEWithLogitsLoss for numerical stability
        )
        
    def forward(self, x):
        features = self.conv_blocks(x).squeeze(-1)
        return self.classifier(features)

# ==========================================
# 4. MLOps ORCHESTRATION (MLflow Training)
# ==========================================
def main():
    base_dir = Path(__file__).parent.parent
    db_path = base_dir / "data" / "processed" / "ptbxl.db"
    raw_dir = base_dir / "data" / "raw"
    
    print("Extracting labels from SQL Database...")
    df = extract_dataset_from_db(db_path)
    
    # PRO TACTIC: SSD Physical-Sync Tolerance Filter
    # Ignora silenciosamente arquivos cortados ou não terminados de sincronizar da nuvem
    print("Verifying physical data availability on SSD (500Hz)...")
    df['exists'] = df['path'].apply(lambda x: (raw_dir / x).with_suffix('.dat').exists() and (raw_dir / x).with_suffix('.hea').exists())
    valid_df = df[df['exists']]
    
    missing = len(df) - len(valid_df)
    if missing > 0:
        print(f"⚠️ IGNORANDO {missing} exames fantasmas (Não baixados pelo Windows/OneDrive).")
    df = valid_df.drop(columns=['exists'])
    
    train_df = df[df['strat_fold'] <= 8].reset_index(drop=True)
    val_df = df[df['strat_fold'] == 9].reset_index(drop=True)
    
    # PRO TACTIC 5: Remove the df.head() lock and train on the COMPLETE dataset!
    print(f"Training on Full Size: {len(train_df)} Exams")
    
    train_loader = DataLoader(ECGDataset(train_df, raw_dir), batch_size=128, shuffle=True)
    val_loader = DataLoader(ECGDataset(val_df, raw_dir), batch_size=128, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECG_CNN().to(device)
    
    # PRO TACTIC 6: Class Imbalance Weighting. MI is minority, we force network to care 3x more.
    num_pos = train_df['target_mi'].sum()
    num_neg = len(train_df) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # PRO TACTIC 7: Learning Rate adaptation when plateauing
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    mlflow.set_experiment("ECG_Myocardial_Infarction_Detector")
    
    with mlflow.start_run():
        epochs = 10 # Train longer dynamically
        mlflow.log_param("architecture", "CNN1D_BatchNorm_Dropout")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("pos_weight", pos_weight.item())
        
        print(f"\nStarting Training on device: {device}")
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            # Validation
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    val_out = model(X_val)
                    val_loss += criterion(val_out, y_val).item()
                    
                    # Apply sigmoid for probabilities in AUC calculation
                    probs = torch.sigmoid(val_out)
                    all_preds.extend(probs.cpu().numpy())
                    all_targets.extend(y_val.cpu().numpy())
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # PRO TACTIC 8: ROC_AUC is the golden standard for medical AI performance, not Raw Accuracy.
            val_auc = roc_auc_score(all_targets, all_preds)
            scheduler.step(val_auc) # Step based on AUC
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUROC: {val_auc:.4f}")
            
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_auc", val_auc, step=epoch)
            
        print("\nSaving Model and Metadata to MLflow Registry...")
        mlflow.pytorch.log_model(model, "ecg_model_artifacts")
        print("Success! Advanced Architecture pushed to MLOps Registry.")

if __name__ == "__main__":
    main()
