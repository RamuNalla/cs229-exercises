from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from joblib import load
import numpy as np

class InputModel_linear(BaseModel):
    x: float = Field(..., description="Input for prediction", gt=-1e6, lt=1e6)     # Validating Inout range with pydantic

class InputModel_logistic(BaseModel):
    x1: float = Field(..., description="Input for prediction", gt=-1e6, lt=1e6)     # Validating Inout range with pydantic
    x2: float = Field(..., description="Input for prediction", gt=-1e6, lt=1e6)

linear_regression_model = load("linear_regression_model.joblib")
logistic_regression_model = load("logistic_regression_model.joblib")

app = FastAPI()

@app.post("/predict_linear")
def predict_output(input: InputModel_linear):
    try:
        input_vector = np.array([[input.x]])
        prediction = linear_regression_model.predict(input_vector)
        return {"prediction": float(prediction[0])}     #  FastAPI returns data as JSON. JSON only understands native Python types. If you return a numpy.float64, FastAPI tries to serialize it and fails.
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_logistic")
def predict_output(input: InputModel_logistic):
    try:
        input_vector = np.array([[1, input.x1, input.x2]])
        prediction = logistic_regression_model.predict(input_vector)
        return {"prediction": float(prediction[0])}     #  FastAPI returns data as JSON. JSON only understands native Python types. If you return a numpy.float64, FastAPI tries to serialize it and fails.
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome! Use /predict_linear to make predictions."}


  #  For Powershell: curl -X POST http://127.0.0.1:8000/predict_linear -H "Content-Type: application/json" -d '{ "x": 5.0 }'
  # Or verify using this http://127.0.0.1:8000/docs


  # docker build -t fastapi-ml-app1 .            # Builds an image from dockerfile
  # docker run -d -p 8000:8000 fastapi-ml-app    # Run a container (live, running instance of that image), can also run directly from docker desktop.
  # 
  # 
  # 
  # 
  # docker ps          which containers are currently running
  # docker stop <container ID> will stop the container (you can do this from docker desktop as well) 