from fastapi import APIRouter
import joblib
import numpy as np
from fastapi.params import Depends
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sqlalchemy.orm import Session
from db.core import get_db
from db.training_data import get_training_data
from utils import CustomORJSONResponse
from utils import formatter

router = APIRouter(prefix="/train")


@router.post("/", response_class=CustomORJSONResponse)
async def train(db: Session = Depends (get_db)):

    try:
        # get the data from the database
        df = await get_training_data(db)
        # format the data
        times = map(formatter, df.createdAt)
        times_list = list(times)
        df['createdAt'] = times_list
        rdf = df.dropna()
        columns_to_drop = ['iclPercentage', 'id', 'rwkPercentage', 'currentTime', 'wind', 'rain', 'temp']
        yi = df['iclPercentage'].dropna()
        yr = rdf['rwkPercentage']
        xi = df.select_dtypes(include=np.number).drop(columns=columns_to_drop, axis=1).dropna()
        xr = rdf.select_dtypes(include=np.number).drop(columns=columns_to_drop, axis=1)
        # Split data into training and testing sets (80/20 split)
        xi_train, xi_test, yi_train, yi_test = train_test_split(xi, yi, test_size=0.2, random_state=42)
        xr_train, xr_test, yr_train, yr_test = train_test_split(xr, yr, test_size=0.2, random_state=42)
        xr_train.columns = [f"f{i}" for i in range(xr_train.shape[1])]
        xr_test.columns = [f"f{i}" for i in range(xr_train.shape[1])]
        xi_train.columns = [f"f{i}" for i in range(xi_train.shape[1])]
        xi_test.columns = [f"f{i}" for i in range(xi_train.shape[1])]
        # Train the model
        model_icl = xgb.XGBRegressor(objective='reg:squarederror',
                                     n_estimators=100, max_depth=9, learning_rate=0.2)
        model_rwk = xgb.XGBRegressor(objective='reg:squarederror',
                                     n_estimators=100, max_depth=9, learning_rate=0.2)
        model_icl.fit(xi_train, yi_train)
        model_rwk.fit(xr_train, yr_train)
        # Make predictions
        yi_pred = model_icl.predict(xi_test)
        yr_pred = model_rwk.predict(xr_test)
        # Evaluate the model
        results = {
            "MSE icl": float(mean_squared_error(yi_test, yi_pred)),
            "R2 icl": float(r2_score(yi_test, yi_pred)),
            "MSE rwk": float(mean_squared_error(yr_test, yr_pred)),
            "R2 rwk": float(r2_score(yr_test, yr_pred))
        }
        # save the model
        joblib.dump(model_icl, "models/model_icl.pkl")
        joblib.dump(model_rwk, "models/model_rwk.pkl")
    except Exception as e:
        return {"message": str(e)}
    return results


trainRouter = router