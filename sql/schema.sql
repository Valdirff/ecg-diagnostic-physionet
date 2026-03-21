-- schema.sql
-- Robust Normalization (Star Schema) of the PTB-XL ECG Dataset

DROP TABLE IF EXISTS ecg_scp_diagnoses;
DROP TABLE IF EXISTS ecg_records;
DROP TABLE IF EXISTS scp_codes;
DROP TABLE IF EXISTS patients;

-- Dimension Table: Patients
CREATE TABLE patients (
    patient_id INTEGER PRIMARY KEY,
    age        INTEGER,
    sex        INTEGER,
    height     REAL,
    weight     REAL
);

-- Dimension Table: SCP Codes Dictionary (Medical Diagnoses)
CREATE TABLE scp_codes (
    scp_code             TEXT PRIMARY KEY,
    description          TEXT,
    is_diagnostic        REAL,
    is_form              REAL,
    is_rhythm            REAL,
    diagnostic_class     TEXT,
    diagnostic_subclass  TEXT
);

-- Fact Table: Electrocardiograms (Core Events)
CREATE TABLE ecg_records (
    ecg_id           INTEGER PRIMARY KEY,
    patient_id       INTEGER,
    recording_date   TEXT,
    report           TEXT,
    nurse            REAL,
    site             REAL,
    device           TEXT,
    filename_lr      TEXT,
    filename_hr      TEXT,
    strat_fold       INTEGER,
    FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
);

-- Many-to-Many Relational Table (N:M): Maps which ECG had which Diagnoses (and their likelihood)
CREATE TABLE ecg_scp_diagnoses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ecg_id      INTEGER,
    scp_code    TEXT,
    likelihood  REAL,
    FOREIGN KEY(ecg_id) REFERENCES ecg_records(ecg_id),
    FOREIGN KEY(scp_code) REFERENCES scp_codes(scp_code)
);
