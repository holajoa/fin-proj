from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from config import input_dim, hidden_dim, num_layers, output_dim
import torch
import joblib # save minmaxscaler
import pandas as pd
import numpy as np
from model import StockLSTM
from yahoo_fin.stock_info import get_data
from datetime import date, datetime, timedelta

app = FastAPI()

class StockData(BaseModel):
    ticker: str  # stock ticker want to predict

@app.get("/")
def hello_world():
    return "hello folks!"

def predict(stock_prices):
    scaler = get_scaler()
    scaled_prices = scaler.transform(stock_prices)

    input_tensor = torch.tensor(scaled_prices, dtype=torch.float32)
    model = StockLSTM(input_dim, hidden_dim, num_layers, output_dim)
    model.load_state_dict(torch.load("models/lstm/model.pth"))
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor.unsqueeze(0))
    prediction = scaler.inverse_transform(prediction.numpy()).tolist()
    return prediction

def get_scaler():
    with open('models/lstm/scaler.pkl', 'rb') as f:
        scaler = joblib.load(f)
    return scaler

def fetch_stock_prices(ticker):
    today_date = date.today()
    next_working_day = get_next_working_day(today_date)
    fourty_days_before = (today_date - timedelta(days=40)).strftime("%m/%d/%Y")
    stock_data = get_data(ticker, start_date=fourty_days_before, end_date=next_working_day)
    stock_data = stock_data[["close"]][-20:].values.tolist() # get prices for last 20 days
    return stock_data

def get_next_working_day(current_date):
    current_date = pd.to_datetime(current_date)
    date_range = pd.date_range(current_date, current_date + timedelta(days=7))
    working_days = date_range[date_range.dayofweek < 5]
    next_working_day = working_days[working_days > current_date].min()
    return next_working_day.strftime("%m/%d/%Y")

@app.post("/predict")
def predict_stock_price_next_day(data: StockData):
    ticker = data.ticker
    stock_prices = fetch_stock_prices(ticker)
    return predict(stock_prices)
