"""
📂 ECG METADATA ETL - SQLITE CONSTRUCTOR V2.0
- Relational mapping of clinical SCP-metadata.
- Robust exception handling with summary reporting.
- Diagnostic class taxonomy generation.
"""
import os
import sqlite3
import pandas as pd
from pathlib import Path
import ast
import logging

# --- CONFIGURATION ---
DATA_ROOT = Path("data/raw")
DB_OUT = Path("data/processed/ptbxl.db")
METADATA_FILE = DATA_ROOT / "ptbxl_database.csv"
SCP_STATEMENTS_FILE = DATA_ROOT / "scp_statements.csv"

def print_banner(msg):
    print(f"\n{'='*50}\n{msg}\n{'='*50}")

def build_relational_engine():
    print_banner("🚀 INICIALIZANDO ETL: PTB-XL TO SQLITE")
    
    if not METADATA_FILE.exists() or not SCP_STATEMENTS_FILE.exists():
        print(f"❌ ERRO: Arquivos de metadados não encontrados em {DATA_ROOT}!")
        return

    # Load Source Tables
    df = pd.read_csv(METADATA_FILE, index_col='ecg_id')
    print(f"✅ Arquivo de ECGs carregado: {len(df)} registros.")
    
    scp_df = pd.read_csv(SCP_STATEMENTS_FILE, index_col=0)
    print(f"✅ Tabela de diagnósticos SCP carregada: {len(scp_df)} códigos diagnósticos.")

    # Create DB and Schema
    os.makedirs(DB_OUT.parent, exist_ok=True)
    conn = sqlite3.connect(DB_OUT)
    cur = conn.cursor()

    cur.executescript("""
        DROP TABLE IF EXISTS ecg_records;
        DROP TABLE IF EXISTS scp_codes;
        DROP TABLE IF EXISTS ecg_scp_diagnoses;
        
        CREATE TABLE ecg_records (
            ecg_id INTEGER PRIMARY KEY,
            patient_id INTEGER,
            age REAL,
            sex INTEGER,
            height REAL,
            weight REAL,
            filename_lr TEXT,
            filename_hr TEXT,
            strat_fold INTEGER
        );
        
        CREATE TABLE scp_codes (
            scp_code TEXT PRIMARY KEY,
            description TEXT,
            diagnostic_class TEXT
        );
        
        CREATE TABLE ecg_scp_diagnoses (
            ecg_id INTEGER,
            scp_code TEXT,
            probability REAL,
            FOREIGN KEY(ecg_id) REFERENCES ecg_records(ecg_id),
            FOREIGN KEY(scp_code) REFERENCES scp_codes(scp_code)
        );
    """)
    conn.commit()

    # --- 1. INSERT SCP CODES ---
    print("🛠️ Mapeando Classificação SCP...")
    scp_count = 0
    for code, row in scp_df.iterrows():
        cur.execute("INSERT INTO scp_codes VALUES (?, ?, ?)", 
                    (code, row['description'], row['diagnostic_class']))
        scp_count += 1
    
    # --- 2. INSERT ECG RECORDS AND DIAGNOSES ---
    print("🛠️ Processando Registros de ECG e Diagnósticos (Relacional)...")
    ecg_success = 0
    ecg_error = 0
    diag_count = 0

    for ecg_id, row in df.iterrows():
        try:
            # 2a. Insert Main Record
            cur.execute("""
                INSERT INTO ecg_records VALUES (?,?,?,?,?,?,?,?,?)
            """, (ecg_id, row['patient_id'], row['age'], row['sex'], row['height'], row['weight'], 
                  row['filename_lr'], row['filename_hr'], row['strat_fold']))
            
            # 2b. Parse and Insert Clinical Diagnoses (M-N relationship)
            scp_diags = ast.literal_eval(row['scp_codes'])
            for code, prob in scp_diags.items():
                cur.execute("INSERT INTO ecg_scp_diagnoses VALUES (?, ?, ?)", 
                            (ecg_id, code, prob))
                diag_count += 1
            
            ecg_success += 1
        except Exception as e:
            ecg_error += 1
            logging.error(f"⚠️ Erro ao inserir ECG {ecg_id}: {e}")

    conn.commit()
    conn.close()

    # --- SUMMARY REPORT ---
    print_banner("✅ ETL CONCLUÍDO COM SUCESSO")
    print(f"✔️ Registros de ECG inseridos:       {ecg_success}")
    print(f"❌ Registros falhos (Ignorados):     {ecg_error}")
    print(f"✔️ Diagnósticos Mapeados:           {diag_count}")
    print(f"✔️ Categorias SCP Indexadas:        {scp_count}")
    print(f"📂 Banco gerado em:                {DB_OUT}")
    print("="*50)

if __name__ == "__main__":
    build_relational_engine()
