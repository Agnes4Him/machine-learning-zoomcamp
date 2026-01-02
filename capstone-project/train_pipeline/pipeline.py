from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_URL = "http://127.0.0.1:5000"
DATASET_URL = "../data/capstone/smart_home_energy_consumption_large.csv"

mlflow.set_tracking_uri(MLFLOW_URL)

NUMERICAL = [
    "outdoor_temperature_°c",
    "household_size",
    "hour",
    "day_of_week",
    "day",
    "month",
    "is_weekend",
]

CATEGORICAL = ["appliance_type", "season"]
TARGET = "energy_consumption_kwh"

# Load and process data
@task(retries=3, retry_delay_seconds=5)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@task
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r"[()]", "", regex=True)
          .str.replace(r"\s+", "_", regex=True)
    )
    return df

# Feature engineering

@task
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["time"] = pd.to_datetime(df["time"], format="%H:%M")

    df["timestamp"] = pd.to_datetime(
        df["date"].dt.strftime("%Y-%m-%d") + " " +
        df["time"].dt.strftime("%H:%M")
    )

    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df.drop(columns=["date", "time", "timestamp"])

# Split data and prepare target

@task
def split_data(df: pd.DataFrame):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
        df_full_train.reset_index(drop=True),
    )


@task
def prepare_targets(df: pd.DataFrame):
    y = df[TARGET].values
    df = df.drop(columns=[TARGET])
    return df, y

# Vectorization

@task
def vectorize_data(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame
):
    dv = DictVectorizer()

    train_dict = df_train[CATEGORICAL + NUMERICAL].to_dict(orient="records")
    val_dict = df_val[CATEGORICAL + NUMERICAL].to_dict(orient="records")

    X_train = dv.fit_transform(train_dict)
    X_val = dv.transform(val_dict)

    return dv, X_train, X_val

@task
def vectorize_data_full(
    df_full_train: pd.DataFrame,
    df_test: pd.DataFrame
):
    dv = DictVectorizer()

    full_train_dict = df_full_train[CATEGORICAL + NUMERICAL].to_dict(orient="records")
    test_dict = df_test[CATEGORICAL + NUMERICAL].to_dict(orient="records")

    X_full_train = dv.fit_transform(full_train_dict)
    X_test = dv.transform(test_dict)

    return dv, X_full_train, X_test


# Train and log model

@task
def train_and_log_model(
    experiment_name: str,
    dv: DictVectorizer,
    X_train,
    y_train,
    X_val,
    y_val,
):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        params = {"n_estimators": 10, "max_depth": 3}
        mlflow.log_params(params)

        rf = RandomForestRegressor(
            n_estimators=10,
            max_depth=3,
            random_state=1,
            n_jobs=-1,
        )

        rf.fit(X_train, y_train)
        preds = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, preds)
        mlflow.log_metric("rmse", rmse)

        pipeline = make_pipeline(dv, rf)
        mlflow.sklearn.log_model(pipeline, name="models")

        return mlflow.active_run().info.run_id
    

# Register the model

@task
def register_model(run_id: str, model_name: str):
    client = MlflowClient(MLFLOW_URL)
    mlflow.register_model(
        model_uri=f"runs:/{run_id}/models",
        name=model_name,
    )

# Flow definition

@flow(log_prints=True)
def training_flow():
    df = load_data(DATASET_URL)
    df = clean_columns(df)
    df = feature_engineering(df)

    df_train, df_val, df_test, df_full_train = split_data(df)

    df_train, y_train = prepare_targets(df_train)
    df_val, y_val = prepare_targets(df_val)
    df_test, y_test = prepare_targets(df_test)
    df_full_train, y_full_train = prepare_targets(df_full_train)

    dv, X_train, X_val = vectorize_data(
        df_train, df_val
    )

    dv, X_full_train, X_test = vectorize_data_full(
        df_full_train, df_test
    )

    # Experiment 1: Train / Val
    run_id = train_and_log_model(
        "RandomForest Experiment",
        dv,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    # Experiment 2: Full Train
    run_id_full = train_and_log_model(
        "RandomForest Full Train Experiment",
        dv,
        X_full_train,
        y_full_train,
        X_test,
        y_test,
    )

    register_model(
        run_id_full,
        "Energy Consumption RandomForestRegression - Full Train",
    )


if __name__ == "__main__":
    training_flow.serve(
        name="energy-consumption-training-pipeline",
        tags=["ml-pipeline", "energy-consumption"],
        #cron="* * * * *"

    )


