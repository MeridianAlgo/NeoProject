import yfinance as yf
import pandas as pd
import ta
import os
import numpy as np
import src.database as database

def fetch_history_robust(ticker="BTC-USD"):
    """
    Fetches historical data using a combination of Daily and Hourly data
    to ensure the absolute latest data is present without gaps.
    """
    print(f"Fetching robust history for {ticker}...")
    
    # 1. Fetch Max Daily Data
    # Use auto_adjust=True to handle splits/dividends automatically
    df_daily = yf.download(ticker, period="max", interval="1d", progress=False, auto_adjust=True)
    if isinstance(df_daily.columns, pd.MultiIndex):
        df_daily.columns = df_daily.columns.get_level_values(0)
    
    # 2. Fetch recent Hourly data (yfinance keeps ~730 days of hourly)
    # This fills gaps where daily candles haven't finalized yet.
    df_hourly = yf.download(ticker, period="7d", interval="1h", progress=False, auto_adjust=True)
    if isinstance(df_hourly.columns, pd.MultiIndex):
        df_hourly.columns = df_hourly.columns.get_level_values(0)
    
    # Resample hourly to daily
    if not df_hourly.empty:
        # We want OHLCV resampling
        # Note: We use the daily start time to match yfinance daily candles
        df_hourly_resampled = df_hourly.resample('1D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        # Strip timezone info to match daily index format if necessary
        df_hourly_resampled.index = df_hourly_resampled.index.date
        df_hourly_resampled.index = pd.to_datetime(df_hourly_resampled.index)
        
        # Merge Daily and Resampled Hourly
        # Priority to Hourly Resampled for the recent days to fill the "22nd" gap
        # We cut the daily data just before the hourly start
        hourly_start = df_hourly_resampled.index.min()
        df_daily_past = df_daily[df_daily.index < hourly_start]
        df_combined = pd.concat([df_daily_past, df_hourly_resampled])
    else:
        df_combined = df_daily

    # Standardize column names
    df_combined = df_combined.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })
    
    # Sort and drop partial duplicates if any
    df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
    df_combined.sort_index(inplace=True)
    df_combined.dropna(inplace=True)
    
    return df_combined

def add_indicators(df):
    """
    Adds technical indicators to the dataframe.
    """
    if df.empty:
        return df
        
    df = df.copy()
    
    try:
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
        
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_width'] = df['bb_high'] - df['bb_low']
        
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)

        # Log Returns
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        # Lags
        df['ret_lag_1'] = df['log_ret'].shift(1)
        df['ret_lag_2'] = df['log_ret'].shift(2)
        df['ret_lag_3'] = df['log_ret'].shift(3)
        df['ret_lag_7'] = df['log_ret'].shift(7)
        # Volatility
        df['vol_5'] = df['log_ret'].rolling(window=5).std()
        df['vol_20'] = df['log_ret'].rolling(window=20).std()

        df.bfill(inplace=True)
        df.ffill(inplace=True)
    except Exception as e:
        print(f"Error adding indicators: {e}")
        return pd.DataFrame()
    
    return df

def get_processed_data(ticker="BTC-USD"):
    """
    Main function to get, store, and process data.
    Uses SQLite database for persistence.
    """
    database.init_db()
    
    # Robust fetch merging Daily + Hourly
    df_fresh = fetch_history_robust(ticker)
    
    if not df_fresh.empty:
        database.save_to_db(df_fresh, ticker)
    
    df_all = database.load_from_db(ticker)
    
    if df_all.empty:
        print(f"⚠️ No data found in database for {ticker}")
        return None
        
    df_processed = add_indicators(df_all)
    
    print(f"✅ Sync complete for {ticker}: {len(df_processed)} days.")
    return df_processed

if __name__ == "__main__":
    get_processed_data()
