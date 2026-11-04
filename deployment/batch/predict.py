#!/usr/bin/env python
"""
Local version of the batch ride-duration scoring flow.

- Input data:  downloaded once from the public NYC TLC URL, cached in ./data
- Model:       loaded from your local MLflow (sqlite + ./mlruns) by run_id
- Output:      parquet written under ./output (same partition layout as S3 version)
- Orchestration: Prefect (runs locally, no account needed)

Usage:
    python score_batch_local.py green 2021 3 <mlflow_run_id>

Optional env vars:
    MLFLOW_TRACKING_URI   default: sqlite:///mlflow.db
"""
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from prefect import flow, get_run_logger, task
from prefect.context import get_run_context

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

TLC_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))


def generate_uuids(n: int):
    return [str(uuid.uuid4()) for _ in range(n)]


def read_dataframe(filename: Path) -> pd.DataFrame:
    df = pd.read_parquet(filename)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df["ride_id"] = generate_uuids(len(df))
    return df


def prepare_dictionaries(df: pd.DataFrame):
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

    return df[["PU_DO", "trip_distance"]].to_dict(orient="records")


def load_model(run_id: str):
    # Works when the model was logged under artifact path "model" in local MLflow.
    return mlflow.pyfunc.load_model(f"runs:/{run_id}/model")


def save_results(df, y_pred, run_id, output_file: Path):
    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["lpep_pickup_datetime"] = df["lpep_pickup_datetime"]
    df_result["PULocationID"] = df["PULocationID"]
    df_result["DOLocationID"] = df["DOLocationID"]
    df_result["actual_duration"] = df["duration"]
    df_result["predicted_duration"] = y_pred
    df_result["diff"] = df_result["actual_duration"] - df_result["predicted_duration"]
    df_result["model_version"] = run_id

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_parquet(output_file, index=False)


@task(retries=2, retry_delay_seconds=10)
def fetch_input(url: str, local_path: Path) -> Path:
    logger = get_run_logger()
    if local_path.exists():
        logger.info(f"using cached file {local_path}")
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"downloading {url}...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = local_path.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp.rename(local_path)
    return local_path


@task
def apply_model(input_file: Path, run_id: str, output_file: Path) -> Path:
    logger = get_run_logger()

    logger.info(f"reading the data from {input_file}...")
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)

    logger.info(f"loading the model with RUN_ID={run_id}...")
    model = load_model(run_id)

    logger.info("applying the model...")
    y_pred = model.predict(dicts)

    logger.info(f"saving the result to {output_file}...")
    save_results(df, y_pred, run_id, output_file)
    return output_file


def get_paths(run_date: datetime, taxi_type: str, run_id: str):
    prev_month = run_date - relativedelta(months=1)
    year, month = prev_month.year, prev_month.month

    url = TLC_URL.format(taxi_type=taxi_type, year=year, month=month)
    input_file = DATA_DIR / f"{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = (
        OUTPUT_DIR
        / f"taxi_type={taxi_type}"
        / f"year={year:04d}"
        / f"month={month:02d}"
        / f"{run_id}.parquet"
    )
    return url, input_file, output_file


@flow
def ride_duration_prediction(taxi_type: str, run_id: str, run_date: datetime = None):
    if run_date is None:
        ctx = get_run_context()
        run_date = ctx.flow_run.expected_start_time

    url, input_file, output_file = get_paths(run_date, taxi_type, run_id)

    local_input = fetch_input(url, input_file)
    apply_model(input_file=local_input, run_id=run_id, output_file=output_file)


def run():
    taxi_type = sys.argv[1]  # 'green'
    year = int(sys.argv[2])  # 2021
    month = int(sys.argv[3])  # 3
    run_id = sys.argv[4]  # MLflow run id

    ride_duration_prediction(
        taxi_type=taxi_type,
        run_id=run_id,
        run_date=datetime(year=year, month=month, day=1),
    )


if __name__ == "__main__":
    run()