from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import date
import pandas as pd
import uvicorn

app = FastAPI(title="ML ETF Prediction Service", version="1.0.0")

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

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not request.history:
        raise HTTPException(status_code=400, detail="Empty history")

    data = [record.model_dump() for record in request.history]
    df = pd.DataFrame(data)

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['date']).reset_index(drop=True)

    df['sma_5'] = df['closePrice'].rolling(window=5).mean()

    last_price = df['closePrice'].iloc[-1]
    mocked_prediction = last_price * 1.01

    last_date = df['date'].iloc[-1]
    next_business_day = last_date + pd.offsets.BDay(1)

    return PredictionResponse(targetDate=next_business_day.date(), predictedClosePrice=round(mocked_prediction, 2))

if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

