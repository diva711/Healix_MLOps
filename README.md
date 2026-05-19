# Healix MLOps

An end-to-end MLOps pipeline for predicting patient satisfaction based on wait time. The project covers data versioning, experiment tracking, model registry, drift monitoring, and CI/CD automation.

## What this project does

A Random Forest classifier is trained on `data.csv` to predict whether a patient is satisfied given their wait time in minutes. On every push to main, the CI pipeline pulls the latest data, trains the model, logs the run to MLflow, and registers the best version under the alias `@champion`.

## Stack

- **DVC** -- data versioning, remote storage on DagsHub (S3-compatible)
- **MLflow** -- experiment tracking and model registry, hosted on DagsHub
- **Evidently** -- data drift detection between reference and production data
- **scikit-learn** -- Random Forest classifier
- **GitHub Actions** -- CI/CD pipeline
- **FastAPI** -- model serving

## Project structure

```
Healix_MLOps/
├── .dvc/
│   └── config              # DVC remote pointing to DagsHub S3
├── .github/
│   └── workflows/
│       └── mlops_pipeline.yml
├── data.csv                # tracked by DVC, not committed to git
├── data.csv.dvc            # DVC pointer file
├── train.py                # training, MLflow logging, model registration
├── serve.py                # FastAPI inference endpoint
├── monitor.py              # Evidently drift report generation
├── production_logs.csv     # logs from serve.py for drift monitoring
├── drift_report.html       # output of monitor.py
└── requirements.txt
```

## CI pipeline

On every push to `main`, the pipeline runs these steps:

1. Checkout code
2. Set up Python 3.10
3. Install dependencies (pinned versions)
4. Pull data from DagsHub via `dvc pull --force`
5. Train the model and log to MLflow (`python train.py`)

DagsHub credentials are stored as a GitHub secret (`DAGSHUB_TOKEN`) and injected at runtime.

## Setup

### Prerequisites

- Python 3.10
- A DagsHub account with this repo connected

### Local setup

```bash
git clone https://github.com/diva711/Healix_MLOps.git
cd Healix_MLOps
pip install -r requirements.txt
```

Configure DVC credentials locally:

```bash
dvc remote modify origin --local access_key_id YOUR_DAGSHUB_TOKEN
dvc remote modify origin --local secret_access_key YOUR_DAGSHUB_TOKEN
dvc pull
```

### Run training

```bash
python train.py
```

Experiments are logged to: https://dagshub.com/divachandra583/Healix_MLOps/experiments

### Run inference server

```bash
uvicorn serve:app --reload
```

### Run drift monitoring

```bash
python monitor.py
```

Opens `drift_report.html` with an Evidently data drift report comparing `data.csv` (reference) against `production_logs.csv` (current).

## GitHub Actions secrets required

| Secret | Description |
|---|---|
| `DAGSHUB_TOKEN` | DagsHub access token, used for both DVC and MLflow auth |

## DVC remote

```
url = s3://dvc
endpointurl = https://dagshub.com/divachandra583/Healix_MLOps.s3
```

## MLflow tracking

```
https://dagshub.com/divachandra583/Healix_MLOps.mlflow
```

The trained model is registered as `Healix_Sentiment_Model` and aliased as `@champion` after each successful run.