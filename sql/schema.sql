-- schema.sql
-- Normalizacao Robusta (Star Schema) do Dataset PTB-XL de ECG

DROP TABLE IF EXISTS ecg_scp_diagnoses;
DROP TABLE IF EXISTS ecg_records;
DROP TABLE IF EXISTS scp_codes;
DROP TABLE IF EXISTS patients;

-- Tabela Dimensoes: Pacientes
CREATE TABLE patients (
    patient_id INTEGER PRIMARY KEY,
    age        INTEGER,
    sex        INTEGER,
    height     REAL,
    weight     REAL
);

-- Tabela Dimensoes: Dicionario de Codigos SCP (Diagnosticos Medicos)
CREATE TABLE scp_codes (
    scp_code             TEXT PRIMARY KEY,
    description          TEXT,
    is_diagnostic        REAL,
    is_form              REAL,
    is_rhythm            REAL,
    diagnostic_class     TEXT,
    diagnostic_subclass  TEXT
);

-- Tabela Fatos: Os Eletrocardiogramas (Eventos Centrais)
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

-- Tabela Multi-Relacional (N:M): Mapeia qual ECG teve quais Diagnosticos (e sua certeza/probabilidade)
CREATE TABLE ecg_scp_diagnoses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ecg_id      INTEGER,
    scp_code    TEXT,
    likelihood  REAL,
    FOREIGN KEY(ecg_id) REFERENCES ecg_records(ecg_id),
    FOREIGN KEY(scp_code) REFERENCES scp_codes(scp_code)
);
