from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Healthcare Shift Fill Prediction API")

model_package = joblib.load("shift_fill_model.pkl")

model = model_package["model"]
encoders = model_package["encoders"]
feature_columns = model_package["feature_columns"]


class ShiftInput(BaseModel):
    profession: str
    experience: str
    training_hours: int
    shift_duration: int
    weekend_shift: str
    night_shift: str
    hourly_rate: int
    travel_distance: int
    city_development_index: float


@app.get("/")
def home():
    return {
        "message": "Healthcare Shift Fill Prediction API is running"
    }


@app.post("/predict")
def predict_shift(data: ShiftInput):

    input_data = pd.DataFrame([data.model_dump()])

    for column, encoder in encoders.items():
        if column in input_data.columns:
            input_data[column] = encoder.transform(input_data[column])

    input_data = input_data[feature_columns]

    fill_probability = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]
    prediction_label = "Filled" if prediction == 1 else "Not Filled"

    difficulty_score = round((1 - fill_probability) * 100)

    return {
    "prediction": prediction_label,
    "fill_probability": round(float(fill_probability), 2),
    "difficulty_score": difficulty_score
}