# 🩺 Myocardial Infarction Detector (MLOps ECG Project)

Este projeto implementa uma pipeline completa de Machine Learning (MLOps) para a detecção de Infarto do Miocárdio (MI) a partir de sinais de ECG (Eletrocardiograma) de 12 derivações, utilizando o banco de dados PTB-XL.

## 🚀 Resultados Atuais (Test Set)
O modelo treinado atingiu métricas excelentes de performance (V. 1.0):

*   **AUROC:** `0.9242` — Alto poder de separação.
*   **Sensitivity (Recall):** `0.8691` — Detectou ~87% dos casos reais de infarto.
*   **Specificity:** `0.8035` — Poucos falsos positivos.
*   **Accuracy:** `0.8202`
*   **F1-Score:** `0.7113`

---

## 🏗️ Arquitetura do Projeto
1.  **Ingestão:** Os metadados clínicos do PTB-XL são extraídos de um banco de dados SQLite (`sql/schema.sql`).
2.  **Processamento:** Sinais ECG de 12 derivações raw (`wfdb`) são processados com normalização por canal.
3.  **Modelo:** Deep CNN 1D (4 Blocos) com Batch Normalization e Dropout (0.4) para evitar overfitting.
4.  **Treinamento:** Split estratificado (Train 1-8, Val 9, Test 10), Early Stopping e Scheduler Cosine Annealing.
5.  **MLOps:** Integração total com **MLflow** para versionamento de modelos e tracking de métricas.

## 🛠️ Como Executar

### Pré-requisitos
* Python 3.10+
* GPU NVIDIA (GTX 1650+ recomendado para treino)

### Instalação
1. Clone o repositório
2. Crie o ambiente virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Ou .venv\Scripts\activate no Windows
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

### Execução
Para iniciar o treinamento e monitoramento:
```bash
python src/train_mi_detector.py
```

Para visualizar o MLflow:
```bash
mlflow ui
```
