# 🚀 MLOps, LLMOps & AIOps Hands-on Journey

A step-by-step implementation repository tracking my hands-on journey through the full lifecycle of traditional machine learning (MLOps), large language models (LLMOps), and AI for IT operations (AIOps).

I am building this sequentially, starting from core ML model training and scaling up to automated production pipelines.

## 🗺️ Roadmap & Architecture

### 🏗️ Phase 1: MLOps (Current Focus)

Building the foundational automation pipeline for traditional machine learning.

- **Model Training:** Taxi trip duration regression with experiment tracking, hyperparameter tuning, and feature engineering.
- **Experiment Tracking & Model Management:** MLflow for logging runs, comparing metrics, and managing the best model.
- **Workflow Orchestration:** Prefect flows for scheduled batch scoring.
- **CI/CD Pipelines:** Automated testing for data, code, and model artifacts.
- **Deployment:** Online serving with Flask and Docker, offline batch scoring with Prefect, and Kubernetes later.
- **Monitoring:** Evidently reports for prediction drift and data quality, with daily reports tracked in a local Evidently dashboard. Automated retraining triggers come next.

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
- **Data Handling:** Pandas, PyArrow
- **Hyperparameter Tuning:** Hyperopt
- **Experiment Tracking & Model Management:** MLflow
- **Workflow Orchestration:** Prefect
- **Monitoring:** Evidently (pinned to 0.6.7)
- **Monitoring Dashboards:** PostgreSQL, Adminer and Grafana via Docker Compose
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
- [x] Phase 1: Data preparation aligned with the serving features (`PU_DO` + `trip_distance`)
- [x] Phase 1: Flask prediction web service
- [x] Phase 1: API client test script
- [x] Phase 1: Packaging model with Docker
- [x] Phase 1: Random forest pipeline logged and registered in the MLflow Model Registry
- [x] Phase 1: Web service that loads the model from MLflow by run ID
- [x] Phase 1: Local batch scoring flow with Prefect (monthly, model loaded from MLflow)
- [x] Phase 1: Evidently drift report (prediction drift, dataset drift, missing values) on a baseline model
- [x] Phase 1: Evidently dashboard with daily data-quality reports in a local workspace
- [x] Phase 1: Docker Compose stack for monitoring (PostgreSQL, Adminer, Grafana)
- [x] Phase 1: Prefect flow that writes daily Evidently metrics (prediction drift, drifted columns, missing values) to PostgreSQL
- [ ] Phase 1: Build Grafana dashboard panels on the metrics table
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

## 🧹 Data Preparation for the Served Model

The web service loads a pickled `(DictVectorizer, model)` tuple. For predictions to be correct, **training and serving must use exactly the same features**: a `PU_DO` categorical (`"<pickup>_<dropoff>"`) and the numerical `trip_distance`.

### Loading and cleaning the data

The data is the NYC **yellow taxi** dataset, so every datetime column is `tpep_*` (green taxi data uses `lpep_*`). Mixing the two prefixes creates unused attributes instead of converting the real columns.

```python
import pandas as pd

def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    df["duration"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60

    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)].copy()

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

    return df
```

Key points:

- Vectorized `.dt.total_seconds()` replaces the slow row-wise `apply`.
- `.copy()` after filtering avoids `SettingWithCopyWarning`.
- `PU_DO` is created here, once. Do not build it again in a later cell.

### Building features

```python
from sklearn.feature_extraction import DictVectorizer

train_df = read_dataframe("data/yellow_tripdata_2021-01.parquet")
val_df = read_dataframe("data/yellow_tripdata_2021-02.parquet")

categorical = ["PU_DO"]
numerical = ["trip_distance"]
target = "duration"

dv = DictVectorizer()

train_dicts = train_df[categorical + numerical].to_dict(orient="records")
X_train = dv.fit_transform(train_dicts)      # fit on training data only
Y_train = train_df[target].values

val_dicts = val_df[categorical + numerical].to_dict(orient="records")
X_val = dv.transform(val_dicts)              # transform only, never fit again
Y_val = val_df[target].values
```

### Training and saving the model

```python
import pickle
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, Y_train)

with open("lin_reg.bin", "wb") as f_out:
    pickle.dump((dv, lr), f_out)
```

Copy `lin_reg.bin` into `deployment/web-service/`. Check that it is not empty (`ls -l lin_reg.bin`). Use the same scikit-learn version for training and serving.

## 🌐 Prediction Web Service (Flask + Docker)

The model is served as a small REST API and packaged as a Docker image, so it runs the same way on a laptop, in CI, or in the cloud. The service lives in `deployment/web-service/` and does not depend on MLflow at runtime.

### Folder layout

```
deployment/web-service/
├── predict.py        # Flask app with the /predict endpoint
├── test.py           # client script that calls the API
├── lin_reg.bin       # pickled (DictVectorizer, model) tuple
├── Pipfile           # service dependencies
├── Pipfile.lock      # exact pinned versions
└── Dockerfile
```

### How the service works

1. On startup it loads `lin_reg.bin`, which holds the fitted `DictVectorizer` and the trained model as a tuple.
2. `POST /predict` receives a JSON ride and builds the `PU_DO` feature (`PULocationID_DOLocationID`), passing `trip_distance` through.
3. The preprocessor turns the features into a sparse matrix, the model predicts, and the API returns the predicted duration in minutes.

### `predict.py`

```python
import pickle
from flask import Flask, request, jsonify

with open("lin_reg.bin", "rb") as f_in:
    (dv, model) = pickle.load(f_in)


def prepare_features(ride):
    features = {}
    features["PU_DO"] = "%s_%s" % (ride["PULocationID"], ride["DOLocationID"])
    features["trip_distance"] = ride["trip_distance"]
    return features


def predict(features):
    X = dv.transform([features])
    y_pred = model.predict(X)
    return float(y_pred[0])


app = Flask("duration-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)
    return jsonify({"duration": pred})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
```

Two details that matter:

- `prepare_features` runs **once**, in the endpoint. `predict` receives the already-prepared features. Calling `prepare_features` inside `predict` as well raises `KeyError: 'PULocationID'`, because the second call receives the prepared dict.
- The prediction is converted with `float(...)` so the response is always JSON serializable.

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

`duration` is the predicted trip time in minutes. The value above only illustrates the format.

### Test script (`test.py`)

```python
import requests

ride = {
    "PULocationID": 43,
    "DOLocationID": 151,
    "trip_distance": 1.2,
}

uri = "http://localhost:9696/predict"

response = requests.post(uri, json=ride)

print(response.status_code)
print(response.json())
```

Run it while the server is up:

```bash
python test.py
```

Use realistic rides when sanity-checking. A very long distance such as 40 miles is far outside typical trips, and a linear model can return an odd value for it.

### Dependencies (Pipenv)

The service needs only Flask, scikit-learn and gunicorn. MLflow is not required.

```bash
cd deployment/web-service
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
- The container uses **gunicorn** as the production server. Flask's built-in server (`app.run(debug=True)`) is for local development only.

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
cd deployment/web-service
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

Or run `python test.py`.

### Running without Docker (local development)

```bash
cd deployment/web-service
pipenv install --ignore-pipfile      # install exactly what Pipfile.lock says
pipenv run python predict.py         # Flask dev server on port 9696
```

If you edit `predict.py`, stop the server with Ctrl+C and start it again so it runs the new code.

### Troubleshooting

| Problem | Cause and fix |
|---|---|
| `EOFError: Ran out of input` when starting `predict.py` | `lin_reg.bin` is empty (0 bytes). Check with `ls -l lin_reg.bin` and recreate it with `pickle.dump((dv, lr), f_out)` |
| `KeyError: 'PULocationID'` (HTTP 500) | `prepare_features` is being applied twice. Call it only in the endpoint, and make `predict(features)` take the prepared features |
| `KeyError: 'PU_DO'` during training | The DataFrame did not come from the new `read_dataframe`. Re-run the data loading cell |
| Datetime columns not converted, pandas attribute warning | Mixed `lpep_` and `tpep_` prefixes. Use `tpep_` for yellow taxi data |
| Sklearn version warning or unpickling error | Pin scikit-learn to the version used to train the model |
| `docker: command not found` | Start Docker Desktop (`open -a Docker`) and make sure `/Applications/Docker.app/Contents/Resources/bin` is on your `PATH` |
| `pipenv install --deploy` fails in the build | Check that the Python version matches the Pipfile and that `Pipfile.lock` is current (`pipenv lock`) |
| Port 9696 already in use | Map another host port, for example `-p 9797:9696` |

## 🔗 Serving a Model Straight from MLflow

The first service loaded a pickle file (`lin_reg.bin`). This version loads the model **directly from the MLflow tracking server by run ID**, so what is served is exactly what was logged and can be traced back to its run. Everything runs locally (no AWS or other cloud): MLflow uses a local SQLite database and local artifact storage.

### Logging the model

A scikit-learn pipeline bundles the `DictVectorizer` and the model, so the service does not need a separate preprocessor file. The pipeline is logged and registered in one step:

```python
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline

with mlflow.start_run():
    params = dict(max_depth=20, n_estimators=100, min_samples_leaf=10, random_state=0)
    mlflow.log_params(params)

    pipeline = make_pipeline(
        DictVectorizer(),
        RandomForestRegressor(**params, n_jobs=-1),
    )
    pipeline.fit(dict_train, y_train)

    y_pred = pipeline.predict(dict_val)
    rmse = root_mean_squared_error(y_val, y_pred)
    mlflow.log_metric("rmse", rmse)

    mlflow.sklearn.log_model(
        pipeline,
        name="model-rf",
        registered_model_name="nyc-taxi-duration-rf",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )
```

Notes:

- Recent MLflow versions save sklearn models in the safer `skops` format and refuse types they have not been told to trust. A random forest contains `sklearn.tree._tree.Tree`, so it is listed explicitly in `skops_trusted_types`. This is safe because the model was trained locally. Only trust the specific types you have reviewed.
- `name=` replaces the deprecated `artifact_path=`.
- For speed while experimenting, use fewer trees (for example `n_estimators=30`) or a subsample of the training data.

### Starting MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

- Start it from the folder that holds `mlflow.db`, or use an absolute path (`sqlite:////Users/<you>/.../mlflow.db`, four slashes).
- Use **one MLflow version everywhere** (notebook, orchestrator, UI, service). A database upgraded by a newer MLflow cannot be opened by an older one and fails with `Can't locate revision identified by '...'`.
- If port 5000 is taken (AirPlay uses it on macOS), add `--port 5001` and update the tracking URI in the scripts.

### The service (`deployment/web-service-mlflow/predict.py`)

```python
import os
import mlflow
from flask import Flask, request, jsonify

RUN_ID = os.getenv("RUN_ID")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

logged_model = f"runs:/{RUN_ID}/model-rf"
model = mlflow.sklearn.load_model(logged_model)   # DictVectorizer + RandomForest pipeline


def prepare_features(ride):
    features = {}
    features["PU_DO"] = "%s_%s" % (ride["PULocationID"], ride["DOLocationID"])
    features["trip_distance"] = ride["trip_distance"]
    return features


def predict(features):
    preds = model.predict([features])    # list of dicts, like in training
    return float(preds[0])


app = Flask("duration-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)
    return jsonify({"duration": pred})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
```

How it works:

1. The MLflow tracking URI and the run ID come from environment variables, so the same code serves any run.
2. `runs:/<RUN_ID>/model-rf` points at the artifact folder logged above. The folder name must match what was logged.
3. The loaded pipeline takes a **list of dicts**, exactly like training, and does the vectorizing and prediction in one call.

### Running it

Copy the run ID from the MLflow UI, keep the MLflow server running, then:

```bash
cd deployment/web-service-mlflow
RUN_ID=<run id> python predict.py
```

Test it from another terminal with `python test.py` or the `curl` command in the previous section.

### Running it in Docker

Inside a container, `127.0.0.1` is the container itself, not your computer. There are two options:

1. **Talk to the local MLflow server.** Point the container at the host and pass the run ID:
   ```bash
   docker run -it --rm -p 9696:9696 \
     -e RUN_ID=<run id> \
     -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
     duration-prediction:v1
   ```
   This only works if the artifacts are served through the MLflow server. Check with `mlflow.get_run(RUN_ID).info.artifact_uri`. If it is a plain local folder path, the container cannot read it unless that folder is mounted with `-v`.
2. **Bake the model into the image (recommended without a cloud).** Download the model once before building, copy it into the image, and load it without MLflow at runtime:
   ```python
   mlflow.artifacts.download_artifacts(run_id=RUN_ID, artifact_path="model-rf", dst_path="model")
   ```
   The image is then self-contained and does not need the MLflow server or a database.

### Troubleshooting (MLflow-backed service)

| Problem | Cause and fix |
|---|---|
| `TypeError: set_tracking_uri() takes 1 positional argument but 2 were given` | `set_tracking_uri` takes only the URI. Use `mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))` |
| `RESOURCE_DOES_NOT_EXIST: Run with id=None not found` | `RUN_ID` is not set. Start with `RUN_ID=<run id> python predict.py` |
| `UntrustedTypesFoundException: sklearn.tree._tree.Tree` | Pass `skops_trusted_types=["sklearn.tree._tree.Tree"]` to `mlflow.sklearn.log_model` |
| `Can't locate revision identified by '...'` when starting the UI | The database was upgraded by a newer MLflow. Start the UI with that same newer version, or upgrade MLflow in the active environment |
| UI opens but runs are missing | The UI is reading a different `mlflow.db`. Start it from the right folder or use an absolute path |
| Model path not found in the run | The artifact folder name in `logged_model` must match the logged `name` (for example `model-rf`); check the run's Artifacts tab |
| Port 5000 already in use | Use `--port 5001` and update `MLFLOW_TRACKING_URI` |

## 📦 Batch Scoring with Prefect (Local)

The web services above answer one ride at a time. This pipeline scores a **whole month of trips** in one go, which is how predictions are produced for reporting and monitoring. It is a Prefect flow that runs entirely on your machine, with no AWS account: input files are cached in `./data`, the model comes from local MLflow, and results are written to `./output`.

### What it does

For a given run date, the flow:

1. Works out the **previous month** (run date `2021-03-01` scores `2021-02` data).
2. Downloads the public NYC TLC parquet file and caches it in `./data`. Re-runs reuse the cached file.
3. Loads the model from local MLflow by run ID.
4. Predicts the trip duration for every ride and writes a parquet file under `./output`.

### Folder layout

```
deployment/batch/
├── score_batch_local.py   # Prefect flow and helper functions
├── data/                  # cached TLC input files (created on first run)
└── output/                # prediction parquet files (created on first run)
```

### Pipeline structure

| Function | Purpose |
|---|---|
| `fetch_input(url, local_path)` | Prefect task. Downloads the TLC file once, with 2 retries |
| `read_dataframe(filename)` | Reads parquet, computes `duration`, filters 1 to 60 minutes, adds a UUID `ride_id` |
| `prepare_dictionaries(df)` | Builds `PU_DO` and returns `PU_DO` + `trip_distance` as a list of dicts |
| `load_model(run_id)` | Loads the model from MLflow by run ID |
| `apply_model(...)` | Prefect task. Reads, predicts and saves results |
| `get_paths(run_date, taxi_type, run_id)` | Previous-month logic, source URL, cache path and output path |
| `ride_duration_prediction(...)` | The Prefect flow that ties it together |

### Aligning the script with this repo's model

The script was first written around the green taxi dataset and a generic `model` artifact name. This repo trains on **yellow taxi** data and logs the random forest pipeline as `model-rf`, so make three changes:

| Where | Change |
|---|---|
| `read_dataframe`, `save_results` | Use `tpep_pickup_datetime` and `tpep_dropoff_datetime` instead of `lpep_*` |
| `load_model` | Use `mlflow.sklearn.load_model(f"runs:/{run_id}/model-rf")`, the same loader as the web service |
| Command line | Pass `yellow` as the taxi type |

If the model is the XGBoost one from the orchestrator, that run has a separate `DictVectorizer` (`preprocessor/preprocessor.b`) and `models_mlflow`, so load both as shown in "Using the trained model" instead.

### Usage

```bash
python deployment/batch/score_batch_local.py <taxi_type> <year> <month> <mlflow_run_id>
```

Example (scores February 2021, because the flow always processes the month before the run date):

```bash
python deployment/batch/score_batch_local.py yellow 2021 3 <mlflow_run_id>
```

Output is written to:

```
output/taxi_type=yellow/year=2021/month=02/<run_id>.parquet
```

The MLflow location is read from the `MLFLOW_TRACKING_URI` environment variable and defaults to `sqlite:///mlflow.db` in the folder you run from.

### Output columns

| Column | Description |
|---|---|
| `ride_id` | Generated UUID per ride |
| `lpep_pickup_datetime` | Pickup time (`tpep_pickup_datetime` for yellow taxi) |
| `PULocationID`, `DOLocationID` | Pickup and dropoff zones |
| `actual_duration` | Actual trip duration in minutes |
| `predicted_duration` | Model prediction in minutes |
| `diff` | `actual_duration - predicted_duration` |
| `model_version` | MLflow run ID used for the predictions |

### Prefect UI (optional)

The flow runs without any setup. To see runs in a dashboard, start the server and run the script from another terminal:

```bash
prefect server start
```

Then open http://127.0.0.1:4200.

### Scheduling

To score each month automatically, call the flow without a run date. It then uses the flow's expected start time:

```python
from score_batch_local import ride_duration_prediction

ride_duration_prediction.serve(
    name="monthly-scoring",
    cron="0 6 1 * *",
    parameters={"taxi_type": "yellow", "run_id": "<your_run_id>"},
)
```

Free alternatives to a local process: Prefect Cloud's free tier, or a GitHub Actions cron job.

### Moving to the cloud later

Only three places change:

- `get_paths`: swap local paths for `s3://` URIs. Cloudflare R2 and Backblaze B2 are S3-compatible.
- `load_model`: point at your remote MLflow artifact store.
- `save_results`: write to the remote path. `pandas` handles `s3://` once `s3fs` is installed.

### Troubleshooting (batch scoring)

| Problem | Cause and fix |
|---|---|
| `RESOURCE_DOES_NOT_EXIST` or run not found | Wrong `run_id`, or `MLFLOW_TRACKING_URI` does not match the backend used in training |
| Model artifact not found | The artifact folder name in `load_model` must match what was logged (`model-rf`) |
| 403 or 404 when downloading | That month may not be published yet. Try an earlier month |
| `KeyError: 'lpep_pickup_datetime'` | Yellow taxi data uses `tpep_*` columns. See "Aligning the script" above |
| Unpickling or version errors | Install the same `scikit-learn` version used for training |
| Stale or partial input file | Delete the file in `./data` to force a fresh download |

## 📊 Model Monitoring with Evidently

Once a model is serving predictions, the question is whether the data it sees, and the predictions it makes, still look like what it was trained on. This section answers that with [Evidently](https://github.com/evidentlyai/evidently). A one-off **drift report** compares a reference dataset (training data) with a current dataset (validation data), and a local **workspace and dashboard** tracks daily data-quality reports over time.

> **This track is a separate baseline, not the yellow-taxi random forest above.** It uses **green taxi** data (January 2022, `lpep_*` columns) and a `LinearRegression` model with six input columns. The monitoring steps are the same for any model: swap in the yellow-taxi data, the random forest and its features, and update the column mapping.

### Pin Evidently to 0.6.7

Evidently 0.7 replaced its API. On 0.7.x, `from evidently import ColumnMapping` fails with `ImportError: cannot import name 'ColumnMapping'`, and the code below only runs on the older API. Pin the version and restart the notebook kernel:

```bash
pip install "evidently==0.6.7"
```

Install into a clean environment (for example `conda create -n mlops python=3.12`) rather than Anaconda base. Do not pin `fsspec` or `cryptography` by hand; that caused version conflicts (see the troubleshooting table).

### Folder layout

```
.
├── data/         # downloaded TLC files and val_data.parquet
├── models/       # lin_reg.bin
└── workspace/    # Evidently workspace (created on first run)
```

Create `data/` and `models/` before running, because the download and the model save do not create them.

### Step 1: Download the data

```python
import os, requests
from tqdm import tqdm

os.makedirs('./data', exist_ok=True)
os.makedirs('./models', exist_ok=True)

files = [('green_tripdata_2022-02.parquet', './data'), ('green_tripdata_2022-01.parquet', './data')]

for file, path in files:
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{file}'
    resp = requests.get(url, stream=True)
    save_path = f"{path}/{file}"
    with open(save_path, 'wb') as f:
        for data in tqdm(resp.iter_content(),
                         desc=f"{file}",
                         postfix=f"save to {save_path}",
                         total=int(resp.headers["Content-Length"])):
            f.write(data)
```

Only January is used below. February is the natural "next month" to monitor later.

### Step 2: Prepare the data and train a baseline model

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

jan_data = pd.read_parquet('./data/green_tripdata_2022-01.parquet')   # 62,495 rows, 20 columns

# target: trip duration in minutes
jan_data['duration_min'] = (jan_data.lpep_dropoff_datetime - jan_data.lpep_pickup_datetime).dt.total_seconds() / 60

# filter outliers
jan_data = jan_data[(jan_data.duration_min >= 0) & (jan_data.duration_min <= 60)]
jan_data = jan_data[(jan_data.passenger_count > 0) & (jan_data.passenger_count <= 8)]   # 55,211 rows

target = 'duration_min'
num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
cat_features = ["PULocationID", "DOLocationID"]

# first 30,000 rows = reference (training), the rest = current (validation)
train_data = jan_data[:30000].copy()
val_data = jan_data[30000:].copy()

model = LinearRegression()
model.fit(train_data[num_features + cat_features], train_data[target])
```

The `.copy()` calls avoid pandas `SettingWithCopyWarning` when the prediction columns are added next.

### Step 3: Predict, evaluate and save

Use **the same column name, `predictions`, in both datasets**. Evidently needs the prediction column to exist in the reference and the current data.

```python
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

train_data['predictions'] = model.predict(train_data[num_features + cat_features])
val_data['predictions'] = model.predict(val_data[num_features + cat_features])

print(f"Train MAE: {mean_absolute_error(train_data[target], train_data['predictions'])}")
print(f"Val MAE:   {mean_absolute_error(val_data[target], val_data['predictions'])}")

with open('models/lin_reg.bin', 'wb') as f_out:
    dump(model, f_out)

val_data.to_parquet('data/val_data.parquet', index=False)
```

| Metric | Train | Validation |
|---|---|---|
| MAE (minutes) | 3.80 | 4.14 |

MAPE came out around 8e13 to 9e13 in the notebook, which is meaningless. It is most likely caused by trips with a duration of 0 minutes, which the `>= 0` filter keeps and MAPE divides by. Rely on MAE, or filter with `>= 1` to use MAPE.

### Step 4: Drift report

The report compares the training data (reference) with the validation data (current).

```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric

column_mapping = ColumnMapping(
    target=None,
    prediction='predictions',
    numerical_features=num_features,
    categorical_features=cat_features
)

report = Report(metrics=[
    ColumnDriftMetric(column_name='predictions'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric()
])

report.run(reference_data=train_data, current_data=val_data, column_mapping=column_mapping)
report            # renders the interactive report in the notebook
```

Define `column_mapping` and `report` before calling `report.run`, and run the cells in this order.

Pull the numbers out for further processing (alerts, tests, logging):

```python
result = report.as_dict()

result['metrics'][0]['result']['drift_score']                           # prediction drift
result['metrics'][1]['result']['number_of_drifted_columns']             # drifted input columns
result['metrics'][2]['result']['current']['share_of_missing_values']    # missing values in the current data
```

| Check | Result | Reading |
|---|---|---|
| Prediction drift (`predictions`) | 0.0305 (normed Wasserstein distance, threshold 0.1) | No drift detected |
| Drifted columns | 0 | No input drift detected |
| Share of missing values (current data) | 0.0455 | Equals exactly 1/22, which looks like one column that is entirely empty. Check with `val_data.isna().mean()` |

### Step 5: Dashboard with daily data-quality reports

An Evidently **workspace** is a folder that stores reports. A **project** inside it holds the reports and the dashboard panels.

```python
import datetime
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.ui.workspace import Workspace
from evidently.ui.dashboards import DashboardPanelCounter, DashboardPanelPlot, CounterAgg, PanelValue, PlotType, ReportFilter
from evidently.renderers.html_widgets import WidgetSize

ws = Workspace("workspace")

project = ws.create_project("NYC Taxi Data Quality Project")
project.description = "Monitoring data quality for NYC taxi trip data"
project.save()
```

Run `create_project` **once**. Running it again creates a second project with the same name.

Add one report per day. This is a data-quality report on a single day of `val_data`, with no reference dataset, stamped with that day's date so the dashboard can plot it over time:

```python
regular_report = Report(
    metrics=[DataQualityPreset()],
    timestamp=datetime.datetime(2022, 1, 28)
)

regular_report.run(
    reference_data=None,
    current_data=val_data[val_data.lpep_pickup_datetime.between('2022-01-28', '2022-01-29', inclusive='left')],
    column_mapping=column_mapping
)

ws.add_report(project.id, regular_report)
```

To add several days without copying the cell, loop over them. Pick days that exist in `val_data`; check with `val_data.lpep_pickup_datetime.min()` and `.max()`. The raw January file also contains a few pickups from 2009, so always filter by date range as above.

```python
for day in pd.date_range('2022-01-28', '2022-01-31'):
    daily = val_data[val_data.lpep_pickup_datetime.between(day, day + pd.Timedelta(days=1), inclusive='left')]
    if daily.empty:
        continue
    daily_report = Report(metrics=[DataQualityPreset()], timestamp=day.to_pydatetime())
    daily_report.run(reference_data=None, current_data=daily, column_mapping=column_mapping)
    ws.add_report(project.id, daily_report)
```

Configure the dashboard panels once (a title counter, inference count per day, and missing values per day):

```python
project.dashboard.add_panel(
    DashboardPanelCounter(
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        agg=CounterAgg.NONE,
        title="NYC taxi data dashboard"
    )
)

project.dashboard.add_panel(
    DashboardPanelPlot(
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        title="Inference Count",
        values=[PanelValue(metric_id="DatasetSummaryMetric", field_path="current.number_of_rows", legend="count")],
        plot_type=PlotType.BAR,
        size=WidgetSize.HALF,
    ),
)

project.dashboard.add_panel(
    DashboardPanelPlot(
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        title="Number of Missing Values",
        values=[PanelValue(metric_id="DatasetSummaryMetric", field_path="current.number_of_missing_values", legend="count")],
        plot_type=PlotType.LINE,
        size=WidgetSize.HALF,
    ),
)

project.save()
```

### Step 6: Open the dashboard

From the folder that contains `workspace/`:

```bash
evidently ui --workspace ./workspace
```

Then open http://localhost:8000 (Evidently's default port), choose the project, and view the panels and the stored reports.

### Troubleshooting (monitoring)

| Problem | Cause and fix |
|---|---|
| `ImportError: cannot import name 'ColumnMapping' from 'evidently'` | Evidently 0.7+ has a new API. Run `pip install "evidently==0.6.7"` and restart the kernel. |
| `ValueError: Prediction column (predictions) is partially present in data` | The reference and current data use different prediction column names. Use `predictions` in both. |
| `KeyError: 'prediction'` in the evaluation cell | The evaluation reads `train_data['prediction']` (singular) while the prediction cell writes `predictions`. Use `predictions` everywhere. |
| `pip` conflicts between `fsspec`, `cryptography`, `s3fs` or `pyopenssl` | Do not pin `fsspec` or `cryptography` by hand. Use a clean environment, run `pip install -U s3fs pyopenssl`, then `pip check`. |
| `FutureWarning: 'H' is deprecated` when running `DataQualityPreset` | A pandas warning raised inside Evidently. It is harmless. |
| Two projects with the same name in the UI | `create_project` was run twice. Run it once, or delete the `workspace/` folder to start over. |
| MAPE is around 1e13 | Zero-minute trips are kept by the `>= 0` filter. Use MAE, or filter with `>= 1`. |
| Dashboard panels are empty | No reports were added yet, or the reports have no `DatasetSummaryMetric`. Add reports built with `DataQualityPreset()`. |

### Next steps

- Run the same monitoring against the yellow-taxi random forest and its batch predictions.
- Generate the daily report from the Prefect batch flow instead of by hand.
- Trigger retraining when the drift score crosses its threshold.

## 🐳 Monitoring Stack with Docker Compose (Postgres, Adminer, Grafana)

The Evidently workspace is good for exploring reports. For dashboards that live next to other services, this stack stores monitoring metrics in **PostgreSQL** and plots them in **Grafana**, with **Adminer** as a small web UI for the database. Everything runs locally in Docker.

```
 notebook / batch job ──► PostgreSQL (db:5432) ◄── Grafana (localhost:3000)
                              ▲
                        Adminer (localhost:8080)
```

### Folder layout

```
monitoring/
├── docker-compose.yml
├── config/
│   ├── grafana_datasources.yaml     # connects Grafana to Postgres
│   └── grafana_dashboards.yaml      # tells Grafana where dashboard JSON files live
└── dashboards/                      # dashboard JSON files (can start empty)
```

Create the mounted files **before** the first `up`. If they are missing, Docker creates folders in their place and Grafana fails to start.

### `docker-compose.yml`

```yaml
volumes:
  grafana_data: {}

networks:
  front-tier:
  back-tier:

services:
  db:
    image: postgres
    restart: always
    environment:
      POSTGRES_PASSWORD: example
    ports:
      - "5432:5432"
    networks:
      - back-tier

  adminer:
    image: adminer
    restart: always
    ports:
      - "8080:8080"
    networks:
      - back-tier
      - front-tier

  grafana:
    image: grafana/grafana-enterprise
    user: "472"
    ports:
      - "3000:3000"
    volumes:
      - ./config/grafana_datasources.yaml:/etc/grafana/provisioning/datasources/datasource.yaml:ro
      - ./config/grafana_dashboards.yaml:/etc/grafana/provisioning/dashboards/dashboards.yaml:ro
      - ./dashboards:/opt/grafana/dashboards
    networks:
      - back-tier
      - front-tier
    restart: always
```

The `version:` line is left out because Compose v2 ignores it and prints a warning. `example` is a throwaway local password; change it for anything shared. Note that `db` has no volume, so its data is lost when the container is removed. To keep it, add `- postgres_data:/var/lib/postgresql/data` under `db` and declare `postgres_data: {}` under `volumes`.

### `config/grafana_datasources.yaml`

```yaml
apiVersion: 1
datasources:
  - name: PostgreSQL
    type: postgres
    access: proxy
    url: db:5432
    database: test
    user: postgres
    secureJsonData:
      password: 'example'
    jsonData:
      sslmode: 'disable'
```

The host is `db`, the service name on the Docker network, not `localhost`.

### `config/grafana_dashboards.yaml`

```yaml
apiVersion: 1
providers:
  - name: 'Dashboards'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /opt/grafana/dashboards
      foldersFromFilesStructure: true
```

### Start it

```bash
cd monitoring
mkdir -p config dashboards        # then create the two YAML files above
docker compose up -d
docker compose ps                 # db, adminer and grafana should be running
docker compose logs grafana --tail 30
```

The first run pulls the images, so it takes a few minutes. Nothing is built, because all three services use ready-made images.

### Create the `test` database

The Grafana datasource points at a database named `test`, which does not exist yet.

1. Open http://localhost:8080 (Adminer).
2. Log in with System **PostgreSQL**, Server `db`, Username `postgres`, Password `example`, Database empty.
3. Click **Create database**, name it `test`, and save.

### Check Grafana

1. Open http://localhost:3000 and log in with `admin` / `admin` (you can skip the password change).
2. Go to Connections, then Data sources, then PostgreSQL, and click **Save & test**. It should report that the connection works.

### Stop it

```bash
docker compose down        # stop and remove containers (Postgres data is lost without a volume)
docker compose down -v     # also remove named volumes
```

### Troubleshooting (Docker Compose)

| Problem | Cause and fix |
|---|---|
| `docker: command not found` on macOS | Docker Desktop is installed but its CLI is not on your `PATH`. Add it: `export PATH="$PATH:/Applications/Docker.app/Contents/Resources/bin"`, and put the same line in `~/.bash_profile` (or `~/.zshrc`) to keep it |
| `Cannot connect to the Docker daemon` | Docker Desktop is not running. Start it with `open -a Docker` and wait for the whale icon to settle |
| Grafana fails to start or the mounts show as folders | The files under `config/` did not exist when you ran `up`. Create them, run `docker compose down`, delete any folders Docker created in their place, then `up -d` again |
| Grafana permission errors on `dashboards/` (Linux) | Run `chmod -R a+rX config dashboards` |
| Grafana "Save & test" fails | The `test` database does not exist yet, or the password differs from `POSTGRES_PASSWORD` |
| Port 3000, 5432 or 8080 already in use | Change the left side of the port mapping, for example `"3001:3000"` |
| Warning that `version` is obsolete | Delete the `version:` line |

### Next step: feed it with metrics

The stack is empty until something writes to it. The next section shows a Prefect flow that inserts the Evidently results (drift score, number of drifted columns, share of missing values) into a Postgres table, using `localhost:5432`, user `postgres`, password `example`, database `test`. Grafana panels then query that table.

## 🔄 Writing Evidently Metrics to PostgreSQL (Prefect Backfill)

This connects the two previous sections. A Prefect flow replays February 2022 green taxi data one day at a time, scores each day with the baseline model, runs an Evidently report against the reference data, and inserts the results into PostgreSQL in the Docker Compose stack. Grafana can then plot the table.

```
 data/reference.parquet ─┐
 models/lin_reg.bin ─────┼─► Prefect flow ─► Evidently report (per day) ─► PostgreSQL: drift_metrics
 green_tripdata_2022-02 ─┘
```

### Prerequisites

- The Compose stack is running (`docker compose ps` shows `db` as `Up`). The script creates the `test` database itself, so the manual step in Adminer is optional.
- Python packages: `pip install "evidently==0.6.7" prefect "psycopg[binary]" joblib pandas pyarrow`
- `data/green_tripdata_2022-02.parquet` and `models/lin_reg.bin` from the Evidently section.
- `data/reference.parquet`: the training data **including the `predictions` column**. Save it from the notebook once:

```python
train_data.to_parquet('data/reference.parquet', index=False)
```

### What the flow stores

One row per day in the table `drift_metrics` (database `test`):

| Column | Meaning |
|---|---|
| `timestamp` | The day being monitored |
| `prediction_drift` | Drift score of the `predictions` column (normed Wasserstein, threshold 0.1) |
| `num_drifted_columns` | Number of drifted input columns |
| `share_missing_values` | Share of missing values in that day's data |

### `evidently_metrics_calculation.py`

```python
import datetime
import time
import logging

import joblib
import numpy as np
import pandas as pd
import psycopg
from prefect import task, flow
from prefect.cache_policies import NO_CACHE

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10   # seconds between days, to mimic a live feed

create_table_statement = """
drop table if exists drift_metrics;
create table drift_metrics(
    timestamp timestamp,
    prediction_drift float,
    num_drifted_columns integer,
    share_missing_values float
)
"""

num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
cat_features = ["PULocationID", "DOLocationID"]
features = num_features + cat_features

reference_data = pd.read_parquet("data/reference.parquet")
model = joblib.load("models/lin_reg.bin")
raw_data = pd.read_parquet("data/green_tripdata_2022-02.parquet")
begin = datetime.datetime(2022, 2, 1, 0, 0)

# the same prediction column name ("predictions") must exist in reference and current data
column_mapping = ColumnMapping(
    prediction="predictions",
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None,
)


def clean(df):
    # inf or extreme values break the linear model; treat them as missing, then fill with 0
    return df[features].replace([np.inf, -np.inf], np.nan).fillna(0)


@task
def prep_db():
    with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
    with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
        conn.execute(create_table_statement)


# NO_CACHE: Prefect cannot hash a database cursor, so skip input caching for this task
@task(cache_policy=NO_CACHE)
def calculate_metrics_postgresql(curr, i):
    start = begin + datetime.timedelta(i)
    end = begin + datetime.timedelta(i + 1)
    current_data = raw_data[
        (raw_data.lpep_pickup_datetime >= start) & (raw_data.lpep_pickup_datetime < end)
    ].copy()

    if current_data.empty:
        logging.warning("no rows for %s, skipping", start.date())
        return

    current_data["predictions"] = model.predict(clean(current_data))

    # a fresh report per day; reusing one Report object across runs is not safe
    report = Report(metrics=[
        ColumnDriftMetric(column_name="predictions"),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
    ])
    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    result = report.as_dict()

    prediction_drift = result["metrics"][0]["result"]["drift_score"]
    num_drifted_columns = result["metrics"][1]["result"]["number_of_drifted_columns"]
    share_missing_values = result["metrics"][2]["result"]["current"]["share_of_missing_values"]

    curr.execute(
        "insert into drift_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values) "
        "values (%s, %s, %s, %s)",
        (start, prediction_drift, num_drifted_columns, share_missing_values),
    )


@flow
def batch_monitoring_backfill():
    prep_db()
    with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
        for i in range(0, 27):
            with conn.cursor() as curr:
                calculate_metrics_postgresql(curr, i)
            time.sleep(SEND_TIMEOUT)
            logging.info("data sent for day %s", i)


if __name__ == "__main__":
    batch_monitoring_backfill()
```

Key points:

- **`prep_db` drops and recreates the table on every run.** That is fine for a backfill demo. Remove the `drop table` line before keeping real history.
- **`autocommit=True`** on every connection, so each insert is visible immediately and Grafana can read rows while the flow is still running.
- **`range(0, 27)`** covers 1 to 27 February. February has 28 days, so use `range(0, 28)` for the full month.
- A full run takes about 4.5 minutes because of the 10 second pause per day. Set `SEND_TIMEOUT = 0` for a fast run.
- Prefect starts a temporary local server for the run (it logs `Starting temporary server`). No setup is needed.

### Run it

```bash
cd monitoring        # the folder with data/ and models/
python evidently_metrics_calculation.py
```

Check the rows:

```bash
docker compose exec db psql -U postgres -d test -c "select * from drift_metrics order by timestamp;"
```

Or open Adminer (http://localhost:8080), choose database `test`, and open the `drift_metrics` table.

### Query it in Grafana

In Grafana, go to Explore, pick the PostgreSQL datasource, switch to Code mode and run:

```sql
SELECT timestamp AS time, prediction_drift, num_drifted_columns, share_missing_values
FROM drift_metrics
ORDER BY 1
```

Set the time range to **February 2022**, otherwise Grafana shows "No data" because the timestamps are in the past. A drift threshold line at 0.1 on the `prediction_drift` panel makes drifting days easy to spot.

### Troubleshooting (metrics to Postgres)

| Problem | Cause and fix |
|---|---|
| `connection refused` on port 5432 | The `db` container is not running. Run `docker compose up -d` in the folder with the compose file, then `docker compose exec db pg_isready -U postgres` |
| `ValueError: Prediction column (...) is partially present in data` | Reference and current data use different prediction column names. `reference.parquet` and the code must both use `predictions` |
| `HashError ... psycopg.Cursor` logged by Prefect | Prefect cannot hash the cursor argument. The task is still completed, but add `cache_policy=NO_CACHE` to silence it |
| `RuntimeWarning: divide by zero / overflow ... in matmul` | Some rows contain inf or extreme values. The `clean()` helper above replaces inf with NaN and fills with 0. If it persists, inspect `raw_data[num_features].describe()` |
| `InconsistentVersionWarning` for `LinearRegression` | The model was saved with a different scikit-learn version than the one loaded. Pin the version used for training |
| Grafana shows "No data" | Widen the time range to February 2022 and check that the table has rows |
| No rows for some days | `raw_data` has no trips on that date. The flow logs a warning and skips the day |

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
   cd deployment/web-service
   docker build -t duration-prediction:v1 .
   docker run -it --rm -p 9696:9696 duration-prediction:v1
   ```

8. Serve a model straight from MLflow (set `RUN_ID` to a run from the UI):
   ```bash
   cd deployment/web-service-mlflow
   RUN_ID=<run id> python predict.py
   ```

9. Run batch scoring for a month with Prefect (after aligning the script, see the batch section):
   ```bash
   python deployment/batch/score_batch_local.py yellow 2021 3 <run id>
   ```

10. Test the API with `python test.py`, and open the MLflow UI to sort runs by `rmse` and see the best model.

11. Run the monitoring notebook (it needs Evidently 0.6.7) and open the dashboard:
    ```bash
    pip install "evidently==0.6.7"
    evidently ui --workspace ./workspace
    ```

12. Start the Postgres and Grafana monitoring stack (see the Docker Compose section for the config files):
    ```bash
    cd monitoring
    docker compose up -d
    ```
    Adminer is on http://localhost:8080 and Grafana on http://localhost:3000.

13. Write the daily Evidently metrics to Postgres (needs the Compose stack from step 12, plus `data/reference.parquet` and `models/lin_reg.bin`):
    ```bash
    python evidently_metrics_calculation.py
    ```