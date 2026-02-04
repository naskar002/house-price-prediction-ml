# ==============================
# app.py
# House Price Prediction API
# ==============================

import sys
from pathlib import Path

# -------------------------------------------------
# FIX: Add src/ to Python path so pickle can find
# FeatureEngineeringPipeline during joblib.load
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

# -------------------------------------------------
# Standard imports
# -------------------------------------------------
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# -------------------------------------------------
# Load model bundle (pipeline + model)
# -------------------------------------------------
MODEL_PATH = BASE_DIR / "models" / "house_price_model_bundle.pkl"

bundle = joblib.load(MODEL_PATH)
feature_pipeline = bundle["feature_pipeline"]
model = bundle["model"]

# -------------------------------------------------
# Initialize FastAPI app
# -------------------------------------------------
app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices using a trained ML model",
    version="1.0"
)

# -------------------------------------------------
# Input schema (DATA CONTRACT)
# -------------------------------------------------
class HouseInput(BaseModel):
    Size: float = Field(..., description="Property size in square feet")
    Bedrooms: int = Field(..., description="Number of bedrooms")
    Bathrooms: int = Field(..., description="Number of bathrooms")
    Year_Built: int = Field(..., alias="year_built", description="Year the property was built")
    Condition: str = Field(..., description="Property condition (e.g., Excellent, Good, Fair)")
    Type: str = Field(..., description="Property type (e.g., House, Condo, Townhouse)")
    Location: str = Field(..., description="Location/City of the property")
    Date_Sold: str = Field(..., alias="date_sold", description="Date sold (YYYY-MM-DD format)")

    class Config:
        populate_by_name = True  # Allow both alias and field name

# -------------------------------------------------
# Health check endpoint
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------
@app.post("/predict")
def predict_price(data: HouseInput):
    """
    Predict house price for a new property
    """

    # Convert input JSON to DataFrame
    df = pd.DataFrame([{
        "Size": data.Size,
        "Bedrooms": data.Bedrooms,
        "Bathrooms": data.Bathrooms,
        "Year Built": data.Year_Built,
        "Condition": data.Condition,
        "Type": data.Type,
        "Location": data.Location,
        "Date Sold": data.Date_Sold
    }])

    # Apply feature engineering
    X_transformed = feature_pipeline.transform(df)

    # Make prediction
    prediction = model.predict(X_transformed.values)[0]

    return {
        "predicted_price": round(float(prediction), 2)
    }
