import sqlite3
import pandas as pd
import ast
from pathlib import Path

def main():
    # Absolute paths
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    db_path = processed_dir / "ptbxl.db"
    schema_path = base_dir / "sql" / "schema.sql"
    
    print("\n[1/4] Initializing SQL Database...")
    # Create the connection and empty database
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Apply SQL schema
    with open(schema_path, "r", encoding="utf-8") as f:
        cur.executescript(f.read())
        
    print("V Tables created from schema.sql!")

    # -----------------------------------------------------
    print("\n[2/4] Extracting and Cleaning Diagnoses Table (SCP)...")
    # The original codes dataset has the acronym in the Index (Under Unnamed: 0)
    scp_df = pd.read_csv(raw_dir / "scp_statements.csv", index_col=0)
    scp_df.index.name = "scp_code"
    scp_df = scp_df.reset_index()
    
    # Standardize messy column names:
    scp_df = scp_df.rename(columns={
        "diagnostic": "is_diagnostic", 
        "form": "is_form", 
        "rhythm": "is_rhythm"
    })
    
    # Keep only columns projected in the SQL schema
    cols_to_keep = ["scp_code", "description", "is_diagnostic", "is_form", "is_rhythm", "diagnostic_class", "diagnostic_subclass"]
    scp_df = scp_df[cols_to_keep]
    
    scp_df.to_sql("scp_codes", conn, if_exists="append", index=False)
    print(f"V Inserted {len(scp_df)} medical SCP dictionaries into the dimension table!")

    # -----------------------------------------------------
    print("\n[3/4] Extracting and Cleaning Patient and ECG Data...")
    df = pd.read_csv(raw_dir / "ptbxl_database.csv", index_col='ecg_id')
    df = df.reset_index() # ecg_id becomes a normal column
    
    # Isolate the Patients table and remove old duplicates 
    # (one patient may have taken 10 exams on different dates)
    patients = df[['patient_id', 'age', 'sex', 'height', 'weight']].copy()
    patients = patients.sort_values(by="patient_id").drop_duplicates(subset="patient_id", keep="last")
    patients.to_sql("patients", conn, if_exists="append", index=False)
    print(f"V Inserted {len(patients)} Unique Patients into the base table!")
    
    # Isolate the main Fact Exams table
    records = df[['ecg_id', 'patient_id', 'recording_date', 'report', 
                  'nurse', 'site', 'device', 'filename_lr', 'filename_hr', 'strat_fold']].copy()
    records.to_sql("ecg_records", conn, if_exists="append", index=False)
    print(f"V Inserted {len(records)} Electrocardiogram Records into the fact table!")
    
    # -----------------------------------------------------
    print("\n[4/4] Developing N:M Relationship (Patients vs Multiple Diagnoses)...")
    # The original base packages all diagnoses inside an ugly dictionary in the `scp_codes` column
    # Ex: {'NORM': 100.0, 'LMI': 0, 'IMI': 35.0} -> Let's unwrap this!
    diagnoses_records = []
    
    for _, row in df.iterrows():
        ecg_id = row['ecg_id']
        try:
            # Transform the String into a Real Python Dictionary
            dict_codes = ast.literal_eval(row['scp_codes'])
            for code, likelihood in dict_codes.items():
                diagnoses_records.append((ecg_id, code, float(likelihood)))
        except (ValueError, SyntaxError) as e:
            continue
            
    diagnoses_df = pd.DataFrame(diagnoses_records, columns=['ecg_id', 'scp_code', 'likelihood'])
    diagnoses_df.to_sql("ecg_scp_diagnoses", conn, if_exists="append", index=False)
    print(f"V Relational table ECG_SCP_DIAGNOSES filled with {len(diagnoses_df)} M:N associations!")
    
    conn.commit()
    conn.close()
    
    print("\n=============================================")
    print("🚀 SUCCESS! SQL Database built at 'data/processed/ptbxl.db'")
    print("=============================================\n")

if __name__ == "__main__":
    main()
