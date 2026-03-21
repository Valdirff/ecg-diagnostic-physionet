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

# ==========================================
# 1. INTEGRAÇÃO SQL (Engenharia de Dados)
# ==========================================
def extract_dataset_from_db(db_path: str):
    """
    Usa SQL Avancado para buscar os caminhos dos arquivos originais
    e calcular dinamicamente se o paciente teve um Infarto (MI -> Myocardial Infarction).
    Evitando fazer merges pesados e imperativos no Pandas.
    """
    conn = sqlite3.connect(db_path)
    query = """
    SELECT r.filename_lr as path,
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
# 2. CARREGAMENTO DOS ONDAS ECG (Processamento de Sinais)
# ==========================================
class ECGDataset(Dataset):
    def __init__(self, df, data_dir):
        self.paths = df['path'].values
        self.labels = df['target_mi'].values
        self.data_dir = Path(data_dir)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        # wfdb carrega o sinal binario (Shape original: [1000 amostras, 12 canais])
        # PyTorch Conv1d espera [12 canais, 1000 amostras], por isso aplicamos .T (Transposta)
        record_path = str(self.data_dir / self.paths[idx])
        signal, metadata = wfdb.rdsamp(record_path)
        
        # Normalizando o sinal (Z-score básico)
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        
        signal_tensor = torch.tensor(signal.T, dtype=torch.float32)
        label_tensor = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return signal_tensor, label_tensor

# ==========================================
# 3. CONSTRUÇÃO DA INTELIGÊNCIA ARTIFICIAL (Deep Learning)
# ==========================================
class ECG_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Arquitetura capaz de ler padroes em 12 fios Eletricos de uma so vez
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            # O AdaptivePool forca que não importe o tamanho final, ele retorne 1 celula por canal
            nn.AdaptiveAvgPool1d(1) 
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.conv_blocks(x).squeeze(-1)
        return self.classifier(features)

# ==========================================
# 4. ORQUESTRAÇÃO MLOps (Treinamento com MLflow)
# ==========================================
def main():
    base_dir = Path(__file__).parent.parent
    db_path = base_dir / "data" / "processed" / "ptbxl.db"
    raw_dir = base_dir / "data" / "raw"
    
    print("Extraindo labels do Banco de Dados SQL...")
    df = extract_dataset_from_db(db_path)
    
    # Stratified Split recomendado pelo PhysioNet
    train_df = df[df['strat_fold'] <= 8].reset_index(drop=True)
    val_df = df[df['strat_fold'] == 9].reset_index(drop=True)
    
    # Para teste do codigo nao travar a internet, vamos pegar so as primeiras 2000 amsotras
    train_df = train_df.head(2000)
    val_df = val_df.head(500)
    
    train_loader = DataLoader(ECGDataset(train_df, raw_dir), batch_size=64, shuffle=True)
    val_loader = DataLoader(ECGDataset(val_df, raw_dir), batch_size=64, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECG_CNN().to(device)
    
    criterion = nn.BCELoss() # Binary Cross Entropy para Infarto (Sim/Nao)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # -----------------------------------------------------
    # Inicia o Tracking do MLOps
    # -----------------------------------------------------
    mlflow.set_experiment("ECG_Myocardial_Infarction_Detector")
    
    with mlflow.start_run():
        epochs = 3
        mlflow.log_param("arquitetura", "CNN1D_AdaptiveResnet")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", 0.001)
        
        print(f"\nIniciando Treinamento no dispositivo: {device}")
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
                
            # Validacao
            model.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    val_preds = model(X_val)
                    val_loss += criterion(val_preds, y_val).item()
                    
                    # Calcula Acuracia Bruta
                    predictions_rounded = val_preds.round()
                    correct += (predictions_rounded == y_val).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / len(val_df)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
            
            # Log metricas no MLflow ao vivo
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            
        print("\nSalvando Modelo e Metadados no Registry do MLflow...")
        mlflow.pytorch.log_model(model, "ecg_model_artifacts")
        print("Sucesso! Pipeline End-to-End MLOps validada.")

if __name__ == "__main__":
    main()
