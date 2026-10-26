import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import date
import pandas as pd
import uvicorn
import joblib
from pathlib import Path

from src.config import ARTIFACTS_DIR, FEATURE_COLUMNS

app = FastAPI(title="ML ETF Prediction Service", version="1.0.0")

ARTIFACTS_DIR = Path("data/articles")
FEATURES_DIR = ['log_return', 'volatility_10', 'close_to_sma_10', 'close_to_sma_20', 'volume_momentum']

try:
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
    rf_path = ARTIFACTS_DIR / "rf_model_optimized.pkl"
    if not rf_path.exists():
        rf_path = ARTIFACTS_DIR / "rf_model.pkl"
    rf_model = joblib.load(rf_path)
    models_loaded = True
    print("Successfully loaded RF model from {ARTIFACTS_DIR}")
except Exception as e:
    print("Warning: Could not load RF model from {ARTIFACTS_DIR}. Error: {e}")
    models_loaded = False

class HistoricalRecord(BaseModel):
    date: date
    closePrice: float
    volume: int

class PredictionRequest(BaseModel):
    symbol: str
    history: List[HistoricalRecord]

class PredictionResponse(BaseModel):
    targetDate: date
    predictedClosePrice: float


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['log_return'] = np.log(df['closePrice'] / df['closePrice'].shift(1))
    df['volatility_10'] = df['log_return'].rolling(window=10).std()

    df['sma_10'] = df['closePrice'].rolling(window=10).mean()
    df['sma_20'] = df['closePrice'].rolling(window=20).mean()

    df['close_to_sma_10'] = df['closePrice'] / df['sma_10'] - 1
    df['close_to_sma_20'] = df['closePrice'] / df['sma_20'] - 1

    df['vol_sma_10'] = df['volume'].rolling(window=10).mean()
    df['volume_momentum'] = df['volume'] / df['vol_sma_10'] - 1

    return df

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.history or len(request.history) < 20:
        raise HTTPException(status_code=400, detail="Not enough history data")

    #Converting payload to DataFrame
    data = [record.model_dump() for record in request.history]
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['date']).reset_index(drop=True)

    df = calculate_features(df)

    last_raw = df.iloc[-1:]
    last_close_price = last_raw['closePrice'].values[0]

    #getting 5 features in order
    featrues_unscaled = last_raw[FEATURE_COLUMNS]

    #Scalling by saved scaller form dataset
    features_scaled = scaler.transform(featrues_unscaled)
    #Prediction by RF  model
    predicted_log_return = rf_model.predict(features_scaled)[0]

    #reversed log price on actuall predicted price
    #P_tomorrow = P_today * exp(log_Return)
    predicted_price = last_close_price * np.exp(predicted_log_return)

    #counting next bussines day
    last_date = df['date'].iloc[-1]
    next_business_day = last_date + pd.offsets.BDay(1)

    return PredictionResponse(targetDate=next_business_day.date(), predictedClosePrice=round(predicted_price, 2))

if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

