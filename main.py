from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load('model.pkl')

class PredictionInput(BaseModel):
    data: list

@app.post("/predict/")
def predict(input_data: PredictionInput):
    data = np.array(input_data.data).reshape(1, -1)
    prediction = model.predict(data)[0]
    return {"prediction": int(prediction)}
