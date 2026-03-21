import sqlite3
import pandas as pd
import ast
from pathlib import Path

def main():
    # Caminhos absolutos
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    db_path = processed_dir / "ptbxl.db"
    schema_path = base_dir / "sql" / "schema.sql"
    
    print("\n[1/4] Inicializando o Banco de Dados SQL...")
    # Cria a conexao e o banco vazio
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Aplica o esqueleto SQL
    with open(schema_path, "r", encoding="utf-8") as f:
        cur.executescript(f.read())
        
    print("V Tabelas criadas a partir do schema.sql!")

    # -----------------------------------------------------
    print("\n[2/4] Extraindo e Limpando Tabela de Diagnosticos (SCP)...")
    # A base original de códigos vem com a sigla no Index (Abaixo de Unnamed: 0)
    scp_df = pd.read_csv(raw_dir / "scp_statements.csv", index_col=0)
    scp_df.index.name = "scp_code"
    scp_df = scp_df.reset_index()
    # Múltiplos nomes bagunçados, vamos padronizar:
    scp_df = scp_df.rename(columns={
        "diagnostic": "is_diagnostic", 
        "form": "is_form", 
        "rhythm": "is_rhythm"
    })
    
    # Manter apenas as colunas projetadas no schema SQL
    cols_to_keep = ["scp_code", "description", "is_diagnostic", "is_form", "is_rhythm", "diagnostic_class", "diagnostic_subclass"]
    scp_df = scp_df[cols_to_keep]
    
    scp_df.to_sql("scp_codes", conn, if_exists="append", index=False)
    print(f"V Inseridos {len(scp_df)} dicionarios médicos SCP na tabela dimensional!")

    # -----------------------------------------------------
    print("\n[3/4] Extraindo e Limpando Dados dos Pacientes e ECGs...")
    df = pd.read_csv(raw_dir / "ptbxl_database.csv", index_col='ecg_id')
    df = df.reset_index() # ecg_id vira coluna normal
    
    # Isolar a tabela de Pacientes e remover duplicatas antigas 
    # (um paciente pode ter tirado 10 exames em datas diferentes)
    patients = df[['patient_id', 'age', 'sex', 'height', 'weight']].copy()
    patients = patients.sort_values(by="patient_id").drop_duplicates(subset="patient_id", keep="last")
    patients.to_sql("patients", conn, if_exists="append", index=False)
    print(f"V Inseridos {len(patients)} Pacientes Únicos na tabela base!")
    
    # Isolar a tabela principal de Exames Fato
    records = df[['ecg_id', 'patient_id', 'recording_date', 'report', 
                  'nurse', 'site', 'device', 'filename_lr', 'filename_hr', 'strat_fold']].copy()
    records.to_sql("ecg_records", conn, if_exists="append", index=False)
    print(f"V Inseridos {len(records)} Registros de Eletrocardiogramas na tabela fato!")
    
    # -----------------------------------------------------
    print("\n[4/4] Desenvolvendo Relacionamento N:M (Pacientes vs Vários Diagnósticos)...")
    # A base orginal empacota todos os diagnosticos dentro de um dicionario feio na coluna `scp_codes`
    # Ex: {'NORM': 100.0, 'LMI': 0, 'IMI': 35.0} -> Vamos desembrulhar isso!
    diagnoses_records = []
    
    for _, row in df.iterrows():
        ecg_id = row['ecg_id']
        try:
            # Transforma a String em um Dicionario Python Real
            dict_codes = ast.literal_eval(row['scp_codes'])
            for code, likelihood in dict_codes.items():
                diagnoses_records.append((ecg_id, code, float(likelihood)))
        except (ValueError, SyntaxError) as e:
            continue
            
    diagnoses_df = pd.DataFrame(diagnoses_records, columns=['ecg_id', 'scp_code', 'likelihood'])
    diagnoses_df.to_sql("ecg_scp_diagnoses", conn, if_exists="append", index=False)
    print(f"V Tabela relacional ECG_SCP_DIAGNOSES preenchida com {len(diagnoses_df)} associações M:N!")
    
    conn.commit()
    conn.close()
    
    print("\n=============================================")
    print("🚀 SUCESSO! Banco de Dados SQL em 'data/processed/ptbxl.db'")
    print("=============================================\n")

if __name__ == "__main__":
    main()
