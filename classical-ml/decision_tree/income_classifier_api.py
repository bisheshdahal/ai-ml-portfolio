from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Income Prediction API")

# Load the trained model and encoders
model = joblib.load("best_decision_tree_model.pkl")
encoders = joblib.load("adult_label_encoders.pkl")

class Person(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str
    

    
@app.get("/")
def home():
    return {"message": "Welcome to the Income Prediction API. Use the /predict endpoint to get predictions."}

@app.post("/predict")
def predict(person: Person):
    try:
        data = pd.DataFrame([person.dict()])
        data.rename(columns={
        "education_num": "education-num",
        "marital_status": "marital-status",
        "capital_gain": "capital-gain",
        "capital_loss": "capital-loss",
        "hours_per_week": "hours-per-week",
        "native_country": "native-country"
        }, inplace=True)
        for col in data.columns:
            if col in encoders:
                value = str(data[col].iloc[0]).strip()
                if value not in encoders[col].classes_:
                    raise HTTPException(status_code=400, detail=f"Invalid value '{value}' for column '{col}'")
                data[col] = encoders[col].transform(data[col])
        prediction = model.predict(data)
        income = encoders['income'].inverse_transform(prediction)[0]
        return {"prediction": income}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
