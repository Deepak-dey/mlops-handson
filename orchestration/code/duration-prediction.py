import pickle
from pathlib import Path
import pandas as pd
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path("models")
models_folder.mkdir(exist_ok=True)


def read_dataframe(year, month):
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    df = pd.read_parquet(url)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    
    return df


def create_X(df,dv = None):
    categorical = ['PU_DO']
    numerical = ["trip_distance"]
    dicts = df[categorical + numerical].to_dict(orient = 'records')
    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    
    return X,dv



def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run() as run:
        mlflow.log_param("train_data_path", "data/yellow_tripdata_2021-01.parquet")
        mlflow.log_param("valid_data_path", "data/yellow_tripdata_2021-02.parquet")
        train = xgb.DMatrix(X_train, label=y_train)
        val = xgb.DMatrix(X_val, label=y_val)

        params = {
            "learning_rate": 0.6879794592819034,
            "max_depth": 6,
            "min_child_weight": 6.937305288360974,
            "reg_alpha": 0.012669613705483586,
            "reg_lambda": 0.029694554429786836,
            "objective": "reg:squarederror",
            "seed": 42,
        }

        boost = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(val, "validation")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        y_pred = boost.predict(val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        
        with open("models/preprocessor.b","wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.xgboost.log_model(boost, artifact_path="models_mlflow")
        
        return run.info.run_id
    
    
def run(year,month):
    df_train = read_dataframe(year, month)
    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(next_year, next_month)

    y_train = df_train.duration.values
    y_val = df_val.duration.values

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"Model saved in run {run_id}")
    return run_id
    
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()

    run_id = run(year = args.year, month = args.month)

    with open("run_id.txt", "w") as f:
        f.write(run_id)
