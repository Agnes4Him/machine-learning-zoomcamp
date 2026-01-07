import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from typing import Literal
import logging

from dotenv import load_dotenv

from fastapi import FastAPI, status, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
from pydantic import BaseModel, Field, ConfigDict

load_dotenv()

#MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")
#MODEL_NAME = os.getenv("MODEL_NAME", "Energy Consumption RandomForestRegression - Full Train")

MLFLOW_URL = "http://localhost:5000"
MODEL_NAME = "Energy Consumption RandomForestRegression - Full Train"

mlflow.set_tracking_uri(MLFLOW_URL)

client = MlflowClient(MLFLOW_URL)

logging.basicConfig(
    filename="capstone.log",               
    level=logging.INFO,               
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Energy Consumption Prediction",
    description="Web service for serving predictions for the energy consumption of households",
    version="1.0.0"
)

# Fetch the model
try:
    model_name = f"{MODEL_NAME}@latest"
    model_uri = f"models:/{model_name}"

    sklearn_pipeline = mlflow.sklearn.load_model(model_uri)

    dv = sklearn_pipeline.named_steps["dictvectorizer"]
    rf = sklearn_pipeline.named_steps["randomforestregressor"]

    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error occurred while loading model: {e}")
    dv, rf = None, None

# Define classes for request and response bodies
class HouseholdFeatures(BaseModel):
    """
    Pydantic model representing the features of a household for energy consumption prediction.

    Attributes:
        appliance_type: Type of appliance (e.g., Oven, Refrigerator).
        season: Season of the year (e.g., Winter, Summer).
        outdoor_temperature: Outdoor temperature in degrees Celsius.
        household_size: Number of people in the household.
        hour: Hour of the day (0-23).
        day_of_week: Day of the week (0-6, where 0 is Monday).
        day: Day of the month (1-31).
        month: Month of the year (1-12).
        is_weekend: 1 if the day is a weekend, 0 otherwise.
    """

    model_config = ConfigDict(extra="forbid")

    appliance_type: str
    season: Literal["Spring", "Summer", "Autumn", "Winter"]
    outdoor_temperature: float = Field(..., ge=-50, le=60)
    household_size: int = Field(..., ge=1)
    hour: int = Field(..., ge=0, le=24)
    day_of_week: int = Field(..., ge=0, le=6)
    day: int = Field(..., ge=1, le=31)
    month: int = Field(..., ge=1, le=12)
    is_weekend: int = Field(..., ge=0, le=1)

class PredictResponse(BaseModel):
    """
    Response model representing predicted energy consumption.
    """

    energy_consumption: float = Field(
        ...,
        ge=0.0,
        description="Predicted energy consumption (kWh), rounded to 2 decimal places"
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handles validation errors for incoming requests.

    Args:
        request (Request): The HTTP request object.
        exc (RequestValidationError): The validation exception raised by Pydantic.

    Returns:
        JSONResponse: A JSON response with 422 status and error details.
    """

    logger.error(f"Validation error for request {request.url}: {exc}")
    return JSONResponse(
        status_code=422,
        content={"message": "Invalid input", "details": exc.errors()},
    )

def predict_single(household_dict: dict) -> float:
    """
    Perform a prediction for a single household input using the preloaded model.

    Args:
        household (dict): A dictionary of feature values representing a household.

    Returns:
        float: Predicted energy consumption.

    Raises:
        ValueError: If the DictVectorizer or Model is not loaded.
        Exception: If an error occurs during prediction.
    """

    try:
        if dv is None or rf is None:
            raise ValueError("DictVectorizer or Model not loaded properly.")

        X = dv.transform(household_dict)
        result = rf.predict(X)

        return round(float(result[0]), 2)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise


@app.post("/predict")
async def predict(household: HouseholdFeatures) -> PredictResponse:
    """
    Receives JSON input representing a household's features and returns energy consumption prediction.

    Args:
        household (HouseholdFeatures): Validated input features of the household.

    Returns:
        PredictResponse: Predicted energy consumption.

    Raises:
        HTTPException: Returns 500 if prediction fails.
    """

    try:
        pred = predict_single(household.model_dump())
        logger.info(f"Prediction successful: {pred}")
        return PredictResponse(
            energy_consumption=pred
        )
    except Exception as e:
        logger.error(f"Error in /predict endpoint: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed. Check server logs.")

# For health check
@app.get("/ping")
def ping():
    """
    Health check endpoint.

    Returns:
        JSON with a message and status code 200
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "PONG"}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)