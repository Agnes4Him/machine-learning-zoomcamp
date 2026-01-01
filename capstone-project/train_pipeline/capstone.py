import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_URL = "http://127.0.0.1:5000"

mlflow.set_tracking_uri(MLFLOW_URL)


#dataset_url = "https://raw.githubusercontent.com/Agnes4Him/project-datasets/refs/heads/main/smart_home_energy_consumption_large.csv"
dataset_url = "../data/capstone/smart_home_energy_consumption_large.csv"


df = pd.read_csv(dataset_url)


df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(r"[()]", "", regex=True)
      .str.replace(r"\s+", "_", regex=True)
)

# Feature Engineering

df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
df["time"] = pd.to_datetime(df["time"], format="%H:%M")


df["timestamp"] = pd.to_datetime(
    df["date"].dt.strftime("%Y-%m-%d") + " " + df["time"].dt.strftime("%H:%M")
)


df.drop(columns=["date", "time"], inplace=True)


df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)



df.drop(columns=["timestamp"], inplace=True)


numerical = [
    "outdoor_temperature_°c", 
    "household_size", 
    "hour", 
    #"minute", 
    "day_of_week", 
    "day", "month", 
    "is_weekend"]

categorical = ["appliance_type", "season"]


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


y_train = df_train["energy_consumption_kwh"].values
y_val = df_val["energy_consumption_kwh"].values
y_test = df_test["energy_consumption_kwh"].values

del df_train["energy_consumption_kwh"]
del df_val["energy_consumption_kwh"]
del df_test["energy_consumption_kwh"]


# One Hot Encoding

dv = DictVectorizer()


train_dict = df_train[categorical + numerical].to_dict(orient="records")
val_dict = df_val[categorical + numerical].to_dict(orient="records")
test_dict = df_test[categorical + numerical].to_dict(orient="records")

X_train = dv.fit_transform(train_dict)
X_val = dv.transform(val_dict)
X_test = dv.transform(test_dict)


mlflow.set_experiment("RandomForest Experiment")

with mlflow.start_run():
    params = {"max_depth": 3, "n_estimators": 10}
    mlflow.log_params(params)

    rf = RandomForestRegressor(
            n_estimators=10,
            max_depth=3,
            random_state=1, 
            n_jobs=-1)

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)

    mlflow.log_metric("root mean square error", rmse)

    pipeline = make_pipeline(dv, rf)
    mlflow.sklearn.log_model(pipeline, name="models")



df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train["energy_consumption_kwh"].values


del df_full_train["energy_consumption_kwh"]


dicts_full_train = df_full_train.to_dict(orient='records')

#dv = DictVectorizer()
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)


mlflow.set_experiment("RandomForest Full Train Experiment")

with mlflow.start_run():
    params = {"max_depth": 3, "n_estimators": 10}
    mlflow.log_params(params)
    rf = RandomForestRegressor(
            n_estimators=10,
            max_depth=3,
            random_state=1, 
            n_jobs=-1)

    rf.fit(X_full_train, y_full_train)

    y_pred = rf.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)

    mlflow.log_metric("root mean square error", rmse)

    pipeline = make_pipeline(dv, rf)
    mlflow.sklearn.log_model(pipeline, name="models")


# Register the model
client = MlflowClient(MLFLOW_URL)

run_id = client.search_runs(experiment_ids='4')[0].info.run_id
mlflow.register_model(
    model_uri=f"runs:/{run_id}/models",
    name='Energy Consumption RandomForestRegression - Full Train'
)


