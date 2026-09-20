from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

model = joblib.load('mlr.pkl')
scaler = joblib.load('scaler.pkl')

class InputData(BaseModel):
    interest_rate: float
    unemployment_rate: float

@app.post("/predict")
def predict(data: InputData):
    X = np.array([[data.interest_rate, data.unemployment_rate]])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    return {"index_price": round(float(prediction), 2)}