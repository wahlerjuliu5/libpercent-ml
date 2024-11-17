import os
from fastapi import FastAPI, HTTPException
import libsql_experimental as libsql
from dotenv import load_dotenv
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error as MAE
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from dbmodels import Base, PastEntries



load_dotenv()






# Initialize the FastAPI app
app = FastAPI()

#DB connection
url = os.getenv("TURSO_DATABASE_URL")
auth_token = os.getenv("TURSO_AUTH_TOKEN")
dbUrl = f"sqlite+{url}/?authToken={auth_token}&secure=true"
engine = create_engine(dbUrl, connect_args={'check_same_thread': False}, echo=True)


@app.post("/train")
async def train():
    try:
        #get the data from the database
        session = Session(engine)
        df = pd.read_sql_query(select(PastEntries).where(PastEntries.id >= 1000), session.bind)
        session.close()
        #format the data
        times = map (formatter, df.createdAt)
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
        print(f"MAE icl:{MAE(yi_test, yi_pred)} ")
        print(f"Mean Squared Error icl: {mean_squared_error(yi_test, yi_pred)}")
        print(f'R2 icl: {r2_score(yi_test, yi_pred)}')
        print(f"MAE rwk:{MAE(yr_test, yr_pred)} ")
        print(f"Mean Squared Error rwk: {mean_squared_error(yr_test, yr_pred)}")
        print(f'R2 rwk: {r2_score(yr_test, yr_pred)}')

        #save the model
        joblib.dump(model_icl, "models/model_icl.pkl")
        joblib.dump(model_rwk, "models/model_rwk.pkl")

    except Exception as e:
        return {"message": str(e)}
    return df.describe()

@app.get("/predict")
async def predict():
    return {"message": "Predicting"}


def formatter(string):
    string = string.split('T')[1].split('.')[0]
    hours, minutes, seconds = map(int, string.split(':'))
    # Calculate total minutes since start of the day
    total_minutes = hours * 60 + minutes

    return total_minutes


