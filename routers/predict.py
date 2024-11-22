from fastapi import APIRouter
import joblib
import numpy as np
from pandas.conftest import any_int_ea_dtype
from starlette.requests import Request
from datetime import datetime
from utils.prediction_utils import input_array
from utils import CustomORJSONResponse



router = APIRouter(prefix="/predict")


model_icl = joblib.load("models/model_icl.pkl")
model_rwk = joblib.load("models/model_rwk.pkl")

@router.get("/", response_class=CustomORJSONResponse)
async def predict(request: Request):
    data = await request.json()
    input_features = np.array(data["features"]).reshape(1, -1)
    prediction = model_icl.predict(input_features)
    return {"prediction": prediction.tolist()}

@router.post("/start_from_dt", response_class=CustomORJSONResponse)
async def cron(request: Request):
    try:
        dt = await request.json()
        start_time = datetime(dt["year"], dt["month"], dt["day"], dt["hour"], dt["minute"])
        data = input_array(start_time)
        for slot in data:
            input_features = np.array([list(slot.values())]).reshape(1, -1)
            prediction = model_icl.predict(input_features)
            slot["prediction"] = prediction.tolist()
    except Exception as e:
        return {"message": str(e)}
    return data

predictRouter = router