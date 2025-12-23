import sqlite3
import pandas as pd
import os

DB_PATH = "data/trading_bot.db"

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    """Initializes the database schema."""
    conn = get_connection()
    cursor = conn.cursor()
    # Explicitly define columns to match yfinance output + ticker
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            timestamp DATETIME,
            ticker TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (timestamp, ticker)
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(df, ticker="BTC-USD"):
    """Saves a dataframe to the database, avoiding duplicates."""
    if df.empty:
        return
    
    conn = get_connection()
    df_to_save = df.copy()
    
    # Ensure columns match our schema exactly
    if 'ticker' not in df_to_save.columns:
        df_to_save['ticker'] = ticker
    
    # Reset index so timestamp is a column
    df_to_save = df_to_save.reset_index()
    if 'Date' in df_to_save.columns:
        df_to_save = df_to_save.rename(columns={'Date': 'timestamp'})
    elif 'index' in df_to_save.columns:
        df_to_save = df_to_save.rename(columns={'index': 'timestamp'})
    
    # Select only the columns we want in the correct order
    cols = ['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume']
    # Ensure all exist
    for col in cols:
        if col not in df_to_save.columns:
             # Add missing as null or 0
            df_to_save[col] = None
            
    df_to_save = df_to_save[cols]
    
    try:
        # Use a temporary table to handle 'INSERT OR REPLACE'
        df_to_save.to_sql('market_data_temp', conn, if_exists='replace', index=False)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO market_data (timestamp, ticker, open, high, low, close, volume)
            SELECT timestamp, ticker, open, high, low, close, volume FROM market_data_temp
        ''')
        cursor.execute("DROP TABLE market_data_temp")
        conn.commit()
        print(f"Saved {len(df_to_save)} rows to database for {ticker}.")
    except Exception as e:
        print(f"Error saving to DB: {e}")
    finally:
        conn.close()

def load_from_db(ticker="BTC-USD"):
    """Loads all data for a ticker from the database."""
    conn = get_connection()
    query = f"SELECT * FROM market_data WHERE ticker='{ticker}' ORDER BY timestamp ASC"
    try:
        df = pd.read_sql_query(query, conn, index_col='timestamp', parse_dates=['timestamp'])
        # Drop the ticker column from the returned DF as it's redundant for processing
        if 'ticker' in df.columns:
            df = df.drop(columns=['ticker'])
    except Exception as e:
        print(f"Error loading from DB: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

if __name__ == "__main__":
    init_db()
