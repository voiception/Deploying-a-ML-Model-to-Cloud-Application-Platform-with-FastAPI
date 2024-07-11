import os
import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml.data import process_data
from ml.model import inference

app = FastAPI()

categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

model_filepath = os.path.join(os.path.dirname(__file__), 'model', 'model.pkl')
with open(model_filepath, "rb") as model_file:
    trained_model = pickle.load(model_file)

lb_filepath = os.path.join(os.path.dirname(__file__), 'model', 'label_binarizer.pkl')
with open(lb_filepath, "rb") as lb_file:
    label_binarizer = pickle.load(lb_file)

encoder_filepath = os.path.join(os.path.dirname(__file__), 'model', 'encoder.pkl')
with open(encoder_filepath, "rb") as encoder_file:
    data_encoder = pickle.load(encoder_file)

@app.get("/")
def welcome() -> dict:
    return {"message": "Hello! Welcome to ML Cloud API"}

class ModelInput(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        }

@app.post("/predict")
def predict(request: ModelInput) -> dict:
    try:
        input_data = request.dict(by_alias=True)
        input_dataframe = pd.DataFrame([input_data])
        X, _, _, _ = process_data(input_dataframe, categorical_features=categorical_features, training=False, encoder=data_encoder, lb=label_binarizer)

        prediction_result = inference(trained_model, X)
        return {"input": input_data, "prediction": prediction_result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    #uvicorn.run(app, host="0.0.0.0", port=10000)

