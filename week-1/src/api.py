from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import numpy as np

class InputModel(BaseModel):
    x: float

model = load("./week-1/saved_models/linear_regression_model.joblib")

app = FastAPI()

@app.post("/predict_linear")
def predict_output(input: InputModel):

    input_vector = np.array([[input.x]])
    prediction = model.predict(input_vector)
    return {"prediction": prediction[0]}

@app.get("/")
def read_root():
    return {"message": "Welcome! Use /predict_linear to make predictions."}



  #  curl -X POST http://127.0.0.1:8000/predict_linear -H "Content-Type: application/json" -d '{ "x": 5.0 }'