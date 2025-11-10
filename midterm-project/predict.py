import pickle
from typing import Literal

from fastapi import FastAPI, Request
import uvicorn
from pydantic import BaseModel, Field, ConfigDict
import xgboost as xgb

app = FastAPI(
    title="Teens Smartphone Use Addiction Prediction",
    description="Web service for serving predictions for the addiction level of teens to smartphone use",
    version="1.0.0"
)

class Teen(BaseModel):
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
    addiction_level: float = Field(..., ge=0.0, le=10.0, description="Predicted addiction level (1 - 10 scale)")
    addiction_category: str = Field(..., description="Predicted addiction risk category (Low, Moderate, High)")


with open("./model.bin", "rb") as f_out:
    dv, model = pickle.load(f_out)

def predict_single(teen):
    try:
        X_teen = dv.transform(teen)
        features = list(dv.get_feature_names_out())
        dteen = xgb.DMatrix(X_teen, feature_names=features)

        result = model.predict(dteen)
        return float(result)
    except Exception as e:
        print(f"An errror occured: {e}")

@app.post("/predict")
async def predict(teen: Teen) -> PredictResponse:
    """
    Receives JSON input and returns prediction.
    """
    try:
        prob = predict_single(teen.model_dump())

        addiction_category = "Low" if prob < 4 else "Medium" if prob < 7 else "High"

        return PredictResponse(
            addiction_level=prob,
            addiction_category=addiction_category
        )
    except Exception as e:
        pass

@app.get("/ping")
def ping():
    return "PONG"

# For running directly with: uvicorn app:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)