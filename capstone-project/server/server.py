import mlflow.sklearn
from mlflow.tracking import MlflowClient

from fastapi import FastAPI
import uvicorn

MLFLOW_URL = "http://127.0.0.1:5000"

client = MlflowClient(MLFLOW_URL)

app = FastAPI("energy-consumption-predictor")

# Download model from Registry

model_name = "Energy Consumption RandomForestRegression - Full Train"
model_version = 1 

# Get the source path of the model
model_info = client.get_registered_model(model_name)
version_info = client.get_model_version(name=model_name, version=model_version)
source_path = version_info.source


sklearn_pipeline = mlflow.sklearn.load_model(source_path)


dv = sklearn_pipeline.named_steps["dictvectorizer"]
rf = sklearn_pipeline.named_steps["randomforestregressor"]

# Define a function to test model
def predict_consumption2(index):
    household = df.iloc[index][categorical + numerical]
    household_dict = household.to_dict()
    y_household_actual = df.iloc[index].energy_consumption_kwh

    X_household = dv.transform(household_dict)

    y_household_pred = rf.predict(X_household)

    print(f"Actual energy consumption: {y_household_actual}")
    print(f"Predicted eneregy consumption: {y_household_pred[0].round()}")


predict_consumption2(500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)