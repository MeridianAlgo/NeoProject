"""
Training Progress Tracker
Displays comprehensive statistics about Neo's training history
"""
import sqlite3
import os
from datetime import datetime

def get_training_stats():
    """Get comprehensive training statistics"""
    
    print("\n" + "="*70)
    print("  NEO v4 TRAINING STATISTICS")
    print("="*70 + "\n")
    
    # Database stats
    if os.path.exists("data/trading_bot.db"):
        conn = sqlite3.connect("data/trading_bot.db")
        cursor = conn.cursor()
        
        # Get tickers
        cursor.execute("SELECT DISTINCT ticker FROM market_data")
        tickers = [row[0] for row in cursor.fetchall()]
        
        print(f"📊 Available Tickers: {', '.join(tickers)}")
        print()
        
        # Get data points per ticker
        for ticker in tickers:
            cursor.execute("SELECT COUNT(*) FROM market_data WHERE ticker = ?", (ticker,))
            count = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM market_data WHERE ticker = ?", (ticker,))
            date_range = cursor.fetchone()
            
            print(f"  {ticker:10s}: {count:,} data points ({date_range[0][:10]} to {date_range[1][:10]})")
        
        conn.close()
        
        # Database size
        db_size = os.path.getsize("data/trading_bot.db") / (1024 * 1024)
        print(f"\n  Database Size: {db_size:.2f} MB")
    else:
        print("⚠️  No database found")
    
    print()
    
    # Model stats
    if os.path.exists("models/neo_recurrent_v4.zip"):
        model_size = os.path.getsize("models/neo_recurrent_v4.zip") / (1024 * 1024)
        model_time = datetime.fromtimestamp(os.path.getmtime("models/neo_recurrent_v4.zip"))
        print(f"🤖 Model Size: {model_size:.2f} MB")
        print(f"📅 Last Trained: {model_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("⚠️  No trained model found")
    
    print()
    
    # Best model stats
    if os.path.exists("models/best_neo/best_model.zip"):
        best_size = os.path.getsize("models/best_neo/best_model.zip") / (1024 * 1024)
        best_time = datetime.fromtimestamp(os.path.getmtime("models/best_neo/best_model.zip"))
        print(f"🏆 Best Model Size: {best_size:.2f} MB")
        print(f"📅 Best Model Date: {best_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("⚠️  No best model checkpoint found")
    
    print("\n" + "="*70)
    print("  TRAINING SCHEDULE")
    print("="*70 + "\n")
    
    schedule = [
        ("00:00, 08:00, 16:00 UTC", "BTC-USD", "Bitcoin"),
        ("02:00, 10:00, 18:00 UTC", "ETH-USD", "Ethereum"),
        ("04:00, 12:00, 20:00 UTC", "SPY", "S&P 500 ETF"),
        ("06:00, 14:00, 22:00 UTC", "QQQ", "Nasdaq 100 ETF"),
    ]
    
    for time, ticker, name in schedule:
        print(f"  {time:25s} → {ticker:10s} ({name})")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    get_training_stats()
