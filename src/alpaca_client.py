import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

load_dotenv()

class AlpacaClient:
    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_SECRET_KEY")
        self.base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        
        if not self.api_key or not self.api_secret:
            print("Warning: Alpaca API keys missing in .env file.")
            self.api = None
        else:
            self.api = tradeapi.REST(self.api_key, self.api_secret, self.base_url, api_version='v2')

    def get_account_balance(self):
        if not self.api: return 0
        account = self.api.get_account()
        return float(account.equity)

    def place_order(self, symbol, qty, side, order_type='market', time_in_force='gtc'):
        """
        side: 'buy' or 'sell'
        """
        if not self.api: 
            print(f"Dry Run: {side} {qty} {symbol}")
            return
            
        try:
            self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force
            )
            print(f"Order Successful: {side} {qty} {symbol}")
        except Exception as e:
            print(f"Order Failed: {e}")

    def get_position(self, symbol):
        if not self.api: return 0
        try:
            position = self.api.get_position(symbol)
            return float(position.qty)
        except:
            return 0
