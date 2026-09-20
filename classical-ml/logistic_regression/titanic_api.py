from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

class Passenger(BaseModel):
    pclass : int
    sex : int
    age : float
    sibsp : int
    parch : int
    fare : float
    embarked : int
    
@app.get("/")
def home():
    return {"message": "Welcome to the Titanic Survival Prediction API!"}

@app.post("/predict")
def predict(data: Passenger):
    features = pd.DataFrame([[
        data.pclass,
        data.sex,
        data.age,
        data.sibsp,
        data.parch,
        data.fare,
        data.embarked
    ]], columns=['pclass','sex','age','sibsp','parch','fare','embarked'])
    
    scaled_data = scaler.transform(features)
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]
    
    return {
        'survived' : int(prediction),
        'survival_probability' : round(float(probability),2)
    }