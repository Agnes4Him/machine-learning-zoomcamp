import pickle
from typing import Literal
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
from pydantic import BaseModel, Field, ConfigDict

import xgboost as xgb

logging.basicConfig(
    filename="server.log",               
    level=logging.INFO,               
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Teens Smartphone Use Addiction Prediction",
    description="Web service for serving predictions for the addiction level of teens to smartphone use",
    version="1.0.0"
)

class Teen(BaseModel):
    """
    Pydantic model representing the features of a teenager for addiction prediction.

    Attributes:
        Gender: Gender of the teen, either 'Male' or 'Female'.
        Location: Location or city where the teen resides.
        School_Grade: Current school grade (e.g., '12th').
        Phone_Usage_Purpose: Primary purpose of phone usage (Social Media, Gaming, Education, Browsing, Other).
        Age: Teen's age (between 13 and 19).
        Daily_Usage_Hours: Average daily phone usage hours.
        Sleep_Hours: Average sleep hours per day.
        Academic_Performance: Academic performance score (0-100).
        Social_Interactions: Level of social interactions (0 or higher).
        Exercise_Hours: Average exercise hours per day.
        Anxiety_Level: Anxiety level on a scale from 0 to 10.
        Depression_Level: Depression level on a scale from 0 to 10.
        Self_Esteem: Self-esteem level on a scale from 0 to 10.
        Parental_Control: 0 if no parental control, 1 if parental control exists.
        Screen_Time_Before_Bed: Hours of screen time before bed.
        Phone_Checks_Per_Day: Number of phone checks per day.
        Apps_Used_Daily: Number of apps used daily.
        Time_on_Social_Media: Hours spent on social media.
        Time_on_Gaming: Hours spent on gaming.
        Time_on_Education: Hours spent on educational activities.
        Family_Communication: Level of family communication (0-10).
        Weekend_Usage_Hours: Average phone usage on weekends.
    """

    model_config = ConfigDict(extra="forbid")

    Gender: Literal["Male", "Female"]
    Location: str
    School_Grade: str
    Phone_Usage_Purpose: Literal["Social Media", "Gaming", "Education", "Browsing", "Other"]

    Age: int = Field(..., ge=13, le=19, description="Teen's age (between 13 and 19)")
    Daily_Usage_Hours: float = Field(..., ge=0, le=24)
    Sleep_Hours: float = Field(..., ge=0, le=24)
    Academic_Performance: int = Field(..., ge=0, le=100)
    Social_Interactions: int = Field(..., ge=0)
    Exercise_Hours: float = Field(..., ge=0, le=24)
    Anxiety_Level: int = Field(..., ge=0, le=10)
    Depression_Level: int = Field(..., ge=0, le=10)
    Self_Esteem: int = Field(..., ge=0, le=10)
    Parental_Control: Literal[0, 1]
    Screen_Time_Before_Bed: float = Field(..., ge=0, le=24)
    Phone_Checks_Per_Day: int = Field(..., ge=0)
    Apps_Used_Daily: int = Field(..., ge=0)
    Time_on_Social_Media: float = Field(..., ge=0, le=24)
    Time_on_Gaming: float = Field(..., ge=0, le=24)
    Time_on_Education: float = Field(..., ge=0, le=24)
    Family_Communication: int = Field(..., ge=0, le=10)
    Weekend_Usage_Hours: float = Field(..., ge=0, le=24)

class PredictResponse(BaseModel):
    """
    Response model representing the predicted addiction outcome.

    Attributes:
        addiction_level: Predicted addiction level on a scale from 0 to 10.
        addiction_category: Addiction risk category ('Low', 'Medium', or 'High').
    """

    addiction_level: float = Field(..., ge=0.0, le=10.0, description="Predicted addiction level (1 - 10 scale)")
    addiction_category: str = Field(..., description="Predicted addiction risk category (Low, Moderate, High)")

try:
    with open("./model.bin", "rb") as f_out:
        dv, model = pickle.load(f_out)
        logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error occurred while loading model: {e}")
    dv, model = None, None


def predict_single(teen):
    """
    Perform a prediction for a single teen input using the preloaded model.

    Args:
        teen (dict): A dictionary of feature values representing a teen.

    Returns:
        float: Predicted addiction level.

    Raises:
        ValueError: If the model or DictVectorizer is not loaded.
        Exception: If an error occurs during prediction.
    """

    try:
        if model is None or dv is None:
            raise ValueError("Model or DictVectorizer not loaded properly.")
        
        X_teen = dv.transform(teen)
        features = list(dv.get_feature_names_out())
        dteen = xgb.DMatrix(X_teen, feature_names=features)
        result = model.predict(dteen)
        return float(result)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

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


@app.post("/predict")
async def predict(teen: Teen) -> PredictResponse:
    """
    Receives JSON input representing a teen's features and returns addiction prediction.

    Args:
        teen (Teen): Validated input features of the teen.

    Returns:
        PredictResponse: Predicted addiction level and category.

    Raises:
        HTTPException: Returns 500 if prediction fails.
    """

    try:
        prob = predict_single(teen.model_dump())
        addiction_category = "Low" if prob < 4 else "Medium" if prob < 7 else "High"
        logger.info(f"Prediction successful: {prob:.2f} ({addiction_category})")
        return PredictResponse(
            addiction_level=prob,
            addiction_category=addiction_category
        )
    except Exception as e:
        logger.error(f"Error in /predict endpoint: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed. Check server logs.")

@app.get("/ping")
def ping():
    """
    Health check endpoint.

    Returns:
        str: 'PONG' if the server is running.
    """
    
    return "PONG"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)