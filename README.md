# 🚀 MLOps, LLMOps & AIOps Hands-on Journey

A step-by-step implementation repository tracking my hands-on journey through the full lifecycle of traditional machine learning (MLOps), large language models (LLMOps), and AI for IT operations (AIOps).

I am building this sequentially, starting from core ML model training and scaling up to automated production pipelines.

## 🗺️ Roadmap & Architecture

### 🏗️ Phase 1: MLOps (Current Focus)

Building the foundational automation pipeline for traditional machine learning.

- **Model Training:** Taxi trip duration regression with experiment tracking, hyperparameter tuning, and feature engineering.
- **Experiment Tracking & Model Management:** MLflow for logging runs, comparing metrics, and managing the best model.
- **CI/CD Pipelines:** Automated testing for data, code, and model artifacts.
- **Deployment:** Containerized microservices using Docker and Kubernetes.
- **Monitoring:** Drift detection and automated model retraining triggers.

### 🤖 Phase 2: LLMOps (Upcoming)

Scaling operational practices to generative AI and LLMs.

- **Prompt Engineering:** Versioning and tracking optimal prompt templates.
- **Fine-Tuning:** Efficient fine-tuning loops (LoRA/QLoRA) with resource tracking.
- **Vector DBs & RAG:** Retrieval-augmented generation pipelines.
- **Evaluation & Guardrails:** Tracking hallucination metrics, latency, and token costs.

### 🧠 Phase 3: AIOps (Upcoming)

Applying AI to optimize system operations and infrastructure.

- **Log Anomaly Detection:** Real-time processing of system logs to flag failures.
- **Predictive Scaling:** Forecasting infrastructure load to scale resources.
- **Root Cause Analysis (RCA):** Automated telemetry tracking to isolate system bugs.

## 🛠️ Tech Stack

- **Language:** Python
- **Machine Learning:** Scikit-Learn, XGBoost, PyTorch
- **Hyperparameter Tuning:** Hyperopt
- **Experiment Tracking & Model Management:** MLflow
- **DevOps & Deployment:** Docker, GitHub Actions

## 📈 Current Progress Tracker

- [x] Set up repository structure and developer environment
- [x] Phase 1: Baseline model training (Lasso, Linear Regression)
- [x] Phase 1: MLflow experiment tracking set up and activated
- [x] Phase 1: XGBoost model with Hyperopt hyperparameter tuning
- [x] Phase 1: Best model selected by lowest RMSE and managed in MLflow
- [x] Phase 1: Training orchestrator script (parameterized by year and month)
- [ ] Phase 1: Packaging model with Docker
- [ ] Phase 2: Core LLM pipeline implementation
- [ ] Phase 3: Telemetry and log collection setup

## 🧪 Experiment Tracking & Model Selection

**Problem:** Regression task predicting taxi trip duration (in minutes) from pickup/dropoff locations and trip distance. Duration is derived from the pickup and dropoff timestamps, and trips outside 1 to 60 minutes are filtered out.

**Approach:**

1. Train baseline models (Linear Regression, Lasso) and log them to MLflow.
2. Train an XGBoost regressor and tune it with Hyperopt (`fmin` with the TPE algorithm).
3. Log every Hyperopt trial as an MLflow run with its parameters and validation RMSE.
4. Pick the run with the lowest validation RMSE and manage that model in MLflow.

**Results:**

| Model | Validation RMSE |
|---|---|
| Linear Regression | 7.360062854588416 |
| Lasso (alpha=0.01) | 7.715332561743763 |
| XGBoost + Hyperopt (best run) | 3.341255488245023 |
| RandomForest | 4.492519796951406 |
| ExtraTrees| 8.160463358348522 |
| GradientBoosting | 4.473238078277749 |
| LinearSVR | 1199.114101329614 |


### Finding the best model

```python
import mlflow

client = mlflow.tracking.MlflowClient()
best_run = client.search_runs(
    experiment_ids=["<experiment_id>"],
    order_by=["metrics.rmse ASC"],
    max_results=1,
)[0]
print(best_run.info.run_id, best_run.data.metrics["rmse"])
```

### Registering the best model

```python
mlflow.register_model(
    model_uri=f"runs:/{best_run.info.run_id}/models",
    name="taxi-duration-predictor",
)
```

### Viewing experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000 to compare runs, sort by RMSE, and inspect the registered model.

## ⚙️ Training Pipeline (Deployment Orchestrator)

A single parameterized script that trains the final XGBoost model end to end for a chosen month and logs everything to MLflow. It is the entry point for the deployment steps that follow (Docker packaging, CI/CD).

### What it does

1. **Reads data:** downloads the NYC yellow taxi parquet file for the training month, and for the following month (used as validation), directly from the TLC public bucket. December rolls over to January of the next year.
2. **Cleans data:** computes `duration` in minutes and keeps trips between 1 and 60 minutes.
3. **Builds features:** a combined `PU_DO` pickup/dropoff categorical plus numerical `trip_distance`. A `DictVectorizer` is fit on the training month only and reused on the validation month, so there is no leakage.
4. **Trains:** XGBoost with the best Hyperopt parameters, up to 1000 boosting rounds with early stopping after 50 rounds without improvement.
5. **Logs to MLflow:** validation RMSE, the fitted `DictVectorizer` (`preprocessor/preprocessor.b`), and the XGBoost model (`models_mlflow`).
6. **Hands off:** writes the MLflow run ID to `run_id.txt` so later pipeline steps know which model to pick up.

### Pipeline structure

| Function | Purpose |
|---|---|
| `read_dataframe(year, month)` | Download, compute duration, filter outliers, build `PU_DO` |
| `create_X(df, dv=None)` | Fit or reuse the `DictVectorizer` and return a sparse feature matrix |
| `train_model(...)` | Train XGBoost, log metric, preprocessor and model, return the run ID |
| `run(year, month)` | Orchestrates the steps above for a train/validation month pair |

### Model parameters

| Parameter | Value |
|---|---|
| `learning_rate` | 0.6880 |
| `max_depth` | 6 |
| `min_child_weight` | 6.9373 |
| `reg_alpha` | 0.0127 |
| `reg_lambda` | 0.0297 |
| `objective` | `reg:squarederror` |
| `seed` | 42 |

### Usage

Start the MLflow server first (it must be reachable at `http://localhost:5000`), then run:

```bash
python src/mlops/orchestrator.py --year 2021 --month 1
```

This trains on January 2021 and validates on February 2021. The script prints the run ID and writes it to `run_id.txt`.

| Argument | Description |
|---|---|
| `--year` | Year of the training data (required) |
| `--month` | Month of the training data, 1 to 12 (required) |

**Requirements:** internet access (the parquet files are downloaded on every run), plus `pandas`, `pyarrow`, `scikit-learn`, `xgboost` and `mlflow`. The validation month must already be published by NYC TLC, so very recent months may fail.

### Using the trained model

```python
import pickle
import mlflow
import xgboost as xgb

run_id = open("run_id.txt").read().strip()

# Load the model and the matching preprocessor from the same run
model = mlflow.xgboost.load_model(f"runs:/{run_id}/models_mlflow")
dv_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/preprocessor/preprocessor.b")
with open(dv_path, "rb") as f:
    dv = pickle.load(f)

X = dv.transform([{"PU_DO": "43_151", "trip_distance": 1.2}])
print(model.predict(xgb.DMatrix(X)))   # predicted duration in minutes
```

Register the model from the run:

```python
mlflow.register_model(
    model_uri=f"runs:/{run_id}/models_mlflow",
    name="taxi-duration-predictor",
)
```

## 🏃 How to Run the Current Iteration

### Prerequisites

Python 3.10+ and `venv` installed.

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Deepak-dey/mlops-handson
   cd mlops-handson
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the MLflow tracking server (in a separate terminal):
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```

5. Run the training script (baseline models, then XGBoost with Hyperopt):
   ```bash
   python src/mlops/train.py
   ```

6. Run the deployment orchestrator to train the final model for a given month:
   ```bash
   python src/mlops/orchestrator.py --year 2021 --month 1
   ```

7. Open the MLflow UI and sort runs by `rmse` to see the best model.