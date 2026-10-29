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
- **Serving:** Flask, Gunicorn
- **Dependency Management:** Pipenv
- **DevOps & Deployment:** Docker, GitHub Actions

## 📈 Current Progress Tracker

- [x] Set up repository structure and developer environment
- [x] Phase 1: Baseline model training (Lasso, Linear Regression)
- [x] Phase 1: MLflow experiment tracking set up and activated
- [x] Phase 1: XGBoost model with Hyperopt hyperparameter tuning
- [x] Phase 1: Best model selected by lowest RMSE and managed in MLflow
- [x] Phase 1: Training orchestrator script (parameterized by year and month)
- [x] Phase 1: Flask prediction web service
- [x] Phase 1: Packaging model with Docker
- [ ] Phase 1: CI/CD with GitHub Actions
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

## 🌐 Prediction Web Service (Flask + Docker)

The model is served as a small REST API and packaged as a Docker image, so it runs the same way on a laptop, in CI, or in the cloud. The service lives in the `web-service/` folder and does not depend on MLflow at runtime.

### Folder layout

```
web-service/
├── predict.py        # Flask app with the /predict endpoint
├── lin_reg.bin       # pickled (DictVectorizer, model) tuple
├── Pipfile           # service dependencies
├── Pipfile.lock      # exact pinned versions
└── Dockerfile
```

### How the service works

1. On startup it loads `lin_reg.bin`, which holds the fitted `DictVectorizer` and the trained model as a tuple.
2. `POST /predict` receives a JSON ride, builds the `PU_DO` feature (`PULocationID_DOLocationID`) and passes `trip_distance` through.
3. The preprocessor turns the features into a sparse matrix, the model predicts, and the API returns the predicted duration in minutes.

### API

**Endpoint:** `POST /predict` on port `9696`

Request:

```json
{
  "PULocationID": 43,
  "DOLocationID": 151,
  "trip_distance": 1.2
}
```

Response:

```json
{
  "duration": 9.87
}
```

`duration` is the predicted trip time in minutes. The value above is only an illustration of the format.

### Dependencies (Pipenv)

The service needs only Flask, scikit-learn and gunicorn. MLflow is not required.

```bash
cd web-service
pipenv --python 3.12
pipenv install scikit-learn==<training version> flask gunicorn
```

Pin scikit-learn to the exact version used to create `lin_reg.bin`. Check it with `pip show scikit-learn` in the training environment. A pickle loaded under a different version can warn or break.

This creates `Pipfile` and `Pipfile.lock`. Commit both. To refresh only the lock file, run `pipenv lock`.

### Dockerfile

```dockerfile
FROM python:3.12-slim

RUN pip install --no-cache-dir -U pip pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy

COPY ["predict.py", "lin_reg.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
```

Notes:

- The Python version in `FROM` must match `python_version` in the Pipfile, otherwise `pipenv install --deploy` fails.
- `--deploy` also fails if `Pipfile.lock` is out of date, so run `pipenv lock` after any dependency change.
- The container uses **gunicorn** as the production server. Flask's built-in server (`app.run(debug=True)`) is only for local development.

A `.dockerignore` keeps the build context small:

```
venv/
.venv/
mlruns/
mlflow.db
data/
*.ipynb
.git
```

### Build and run

```bash
cd web-service
docker build -t duration-prediction:v1 .
docker run -it --rm -p 9696:9696 duration-prediction:v1
```

On an Apple Silicon Mac, build with `--platform linux/amd64` if the image will run on an x86 server:

```bash
docker build --platform linux/amd64 -t duration-prediction:v1 .
```

### Test the service

With the container running, from another terminal:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"PULocationID": 43, "DOLocationID": 151, "trip_distance": 1.2}'
```

Or from Python:

```python
import requests

ride = {"PULocationID": 43, "DOLocationID": 151, "trip_distance": 1.2}
print(requests.post("http://localhost:9696/predict", json=ride).json())
```

### Running without Docker (local development)

```bash
cd web-service
pipenv install --ignore-pipfile      # install exactly what Pipfile.lock says
pipenv run python predict.py         # Flask dev server on port 9696
```

### Troubleshooting

| Problem | Fix |
|---|---|
| `docker: command not found` | Start Docker Desktop (`open -a Docker`) and make sure `/Applications/Docker.app/Contents/Resources/bin` is on your `PATH` |
| `pipenv install --deploy` fails in the build | Check that the Python version matches the Pipfile and that `Pipfile.lock` is current (`pipenv lock`) |
| Sklearn version warning or unpickling error | Pin scikit-learn to the version used to train the model |
| Port 9696 already in use | Map another host port, for example `-p 9797:9696` |

## 🏃 How to Run the Current Iteration

### Prerequisites

Python 3.10+, `venv`, and Docker (for the web service) installed.

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

7. Build and run the prediction service:
   ```bash
   cd web-service
   docker build -t duration-prediction:v1 .
   docker run -it --rm -p 9696:9696 duration-prediction:v1
   ```

8. Open the MLflow UI and sort runs by `rmse` to see the best model, and send a test request to `http://localhost:9696/predict`.