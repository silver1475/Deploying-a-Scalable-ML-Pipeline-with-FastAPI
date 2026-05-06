import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model

# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


project_path = os.getcwd()

encoder_path = os.path.join(project_path, "model", "encoder.pkl")
encoder = load_model(encoder_path)

lb_path = os.path.join(project_path, "model", "lb.pkl")
lb = load_model(lb_path)

model_path = os.path.join(project_path, "model", "model.pkl")
model = load_model(model_path)

app = FastAPI(
    title="Census Income Prediction API",
    description="An API to predict income category based on census data.",
    version="1.0.0"
)


@app.get("/")
async def get_root():
    return {
        "message": "Welcome to the Census Income Prediction API. Use /data/ for inference."
    }


@app.post("/data/")
async def post_inference(data: Data):
    data_dict = data.dict()

    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    df = pd.DataFrame.from_dict(data)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    data_processed, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    _inference = inference(model, data_processed)
    prediction_label = apply_label(_inference)


    return {"result": prediction_label}
