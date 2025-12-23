import database
import pandas as pd

def inspect():
    df = database.load_from_db("BTC-USD")
    if df.empty:
        print("Database is empty or ticker not found.")
        return
    
    print("--- Database Summary ---")
    print(f"Ticker: BTC-USD")
    print(f"Total Rows: {len(df)}")
    print(f"Start Date: {df.index.min()}")
    print(f"End Date: {df.index.max()}")
    print("\nLast 5 rows:")
    print(df.tail())

if __name__ == "__main__":
    inspect()
