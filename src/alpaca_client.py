"""
Alpaca Trading Client with WebSocket Integration
Handles live data streaming and trade execution with robust error handling
"""
import os
import time
import logging
import asyncio
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
    from alpaca.data.live import CryptoDataStream, StockDataStream
    from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("⚠️  Alpaca packages not installed. Trading features disabled.")

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AlpacaClient')


class AlpacaClient:
    """
    Robust Alpaca trading client with WebSocket support for live data
    """
    
    def __init__(self):
        """Initialize Alpaca client with API credentials"""
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        
        self.trading_client: Optional[TradingClient] = None
        self.crypto_stream: Optional[CryptoDataStream] = None
        self.stock_stream: Optional[StockDataStream] = None
        self.crypto_data_client: Optional[CryptoHistoricalDataClient] = None
        self.stock_data_client: Optional[StockHistoricalDataClient] = None
        
        self.is_paper = "paper" in self.base_url.lower()
        self.last_prices: Dict[str, float] = {}
        self.latest_bars: Dict[str, List[Dict[str, Any]]] = {}
        self.stream_thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        
        if not ALPACA_AVAILABLE:
            logger.warning("Alpaca packages not available")
            return
        
        if not self.api_key or not self.secret_key:
            logger.warning("Alpaca API credentials not configured")
            return
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize all Alpaca clients with error handling"""
        try:
            # Trading client
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.is_paper
            )
            
            # Data clients
            self.crypto_data_client = CryptoHistoricalDataClient()
            self.stock_data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            
            # Test connection
            account = self.trading_client.get_account()
            logger.info(f"✅ Connected to Alpaca ({'PAPER' if self.is_paper else 'LIVE'} trading)")
            logger.info(f"💰 Account equity: ${float(account.equity):,.2f}")
            logger.info(f"💵 Buying power: ${float(account.buying_power):,.2f}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca clients: {str(e)}")
            self.trading_client = None
    
    def get_account_balance(self) -> float:
        """Get current account equity with error handling"""
        if not self.trading_client:
            logger.warning("Trading client not initialized")
            return 0.0
        
        try:
            account = self.trading_client.get_account()
            return float(account.equity)
        except Exception as e:
            logger.error(f"Error getting account balance: {str(e)}")
            return 0.0
    
    def get_position(self, symbol: str) -> float:
        """Get current position quantity for a symbol"""
        if not self.trading_client:
            return 0.0
        
        try:
            position = self.trading_client.get_open_position(symbol)
            return float(position.qty)
        except Exception as e:
            # Position doesn't exist (no error, just no position)
            if "position does not exist" in str(e).lower():
                return 0.0
            logger.error(f"Error getting position for {symbol}: {str(e)}")
            return 0.0
    
    async def _crypto_handler(self, data):
        """Handler for crypto bar data"""
        symbol = data.symbol
        price = float(data.close)
        self.last_prices[symbol] = price
        # Update bar cache
        if symbol not in self.latest_bars:
            self.latest_bars[symbol] = []
        self.latest_bars[symbol].append({
            "timestamp": data.timestamp,
            "open": data.open,
            "high": data.high,
            "low": data.low,
            "close": data.close,
            "volume": data.volume
        })
        # Keep last 100 bars
        if len(self.latest_bars[symbol]) > 100:
            self.latest_bars[symbol].pop(0)

    async def _stock_handler(self, data):
        """Handler for stock bar data"""
        symbol = data.symbol
        price = float(data.close)
        self.last_prices[symbol] = price
        # Update bar cache
        if symbol not in self.latest_bars:
            self.latest_bars[symbol] = []
        self.latest_bars[symbol].append({
            "timestamp": data.timestamp,
            "open": data.open,
            "high": data.high,
            "low": data.low,
            "close": data.close,
            "volume": data.volume
        })
        if len(self.latest_bars[symbol]) > 100:
            self.latest_bars[symbol].pop(0)

    def start_streaming(self, symbols: List[str], is_crypto: bool = True):
        """Start WebSocket streaming for given symbols in a background thread"""
        if not ALPACA_AVAILABLE:
            return

        def _run_stream():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            if is_crypto:
                self.crypto_stream = CryptoDataStream(self.api_key, self.secret_key)
                self.crypto_stream.subscribe_bars(self._crypto_handler, *symbols)
                logger.info(f"📡 Subscribed to Crypto WebSocket: {symbols}")
                self.crypto_stream.run()
            else:
                self.stock_stream = StockDataStream(self.api_key, self.secret_key)
                self.stock_stream.subscribe_bars(self._stock_handler, *symbols)
                logger.info(f"📡 Subscribed to Stock WebSocket: {symbols}")
                self.stock_stream.run()

        self.stream_thread = threading.Thread(target=_run_stream, daemon=True)
        self.stream_thread.start()
        logger.info("🧵 WebSocket stream thread started")

    def stop_streaming(self):
        """Stop all active streams"""
        if self.crypto_stream:
            self.crypto_stream.stop()
        if self.stock_stream:
            self.stock_stream.stop()
        logger.info("🛑 WebSocket streams stopped")

    def get_latest_price(self, symbol: str, is_crypto: bool = True) -> Optional[float]:
        """Get latest price for a symbol, prioritizing WebSocket stream"""
        # 1. Check WebSocket cache first
        if symbol in self.last_prices:
            return self.last_prices[symbol]
            
        # 2. Fall back to REST API
        try:
            if is_crypto:
                if not self.crypto_data_client:
                    return None
                
                request = CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Minute,
                    limit=1
                )
                bars = self.crypto_data_client.get_crypto_bars(request)
                
                if bars and symbol in bars:
                    latest = bars[symbol][-1]
                    price = float(latest.close)
                    self.last_prices[symbol] = price
                    return price
            else:
                if not self.stock_data_client:
                    return None
                
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Minute,
                    limit=1
                )
                bars = self.stock_data_client.get_stock_bars(request)
                
                if bars and symbol in bars:
                    latest = bars[symbol][-1]
                    price = float(latest.close)
                    self.last_prices[symbol] = price
                    return price
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {str(e)}")
            return None
    
    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        time_in_force: str = "gtc"
    ) -> Optional[Dict[str, Any]]:
        """
        Place an order with comprehensive error handling
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSD", "SPY")
            qty: Quantity to trade
            side: "buy" or "sell"
            order_type: "market" or "limit"
            limit_price: Required for limit orders
            time_in_force: "gtc", "day", "ioc", "fok"
        
        Returns:
            Order details if successful, None otherwise
        """
        if not self.trading_client:
            logger.error("Trading client not initialized")
            return None
        
        if qty <= 0:
            logger.error(f"Invalid quantity: {qty}")
            return None
        
        try:
            # Convert side to enum
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            
            # Convert time in force
            tif_map = {
                "gtc": TimeInForce.GTC,
                "day": TimeInForce.DAY,
                "ioc": TimeInForce.IOC,
                "fok": TimeInForce.FOK
            }
            tif = tif_map.get(time_in_force.lower(), TimeInForce.GTC)
            
            # Create order request
            if order_type.lower() == "market":
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif
                )
            elif order_type.lower() == "limit":
                if limit_price is None:
                    logger.error("Limit price required for limit orders")
                    return None
                
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif,
                    limit_price=limit_price
                )
            else:
                logger.error(f"Invalid order type: {order_type}")
                return None
            
            # Submit order
            order = self.trading_client.submit_order(order_request)
            
            logger.info(f"✅ Order placed: {side.upper()} {qty} {symbol} @ {order_type.upper()}")
            logger.info(f"   Order ID: {order.id}")
            logger.info(f"   Status: {order.status}")
            
            return {
                "id": str(order.id),
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "type": order_type,
                "status": str(order.status),
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None
            }
            
        except Exception as e:
            logger.error(f"❌ Error placing order: {str(e)}")
            return None
    
    def close_all_positions(self) -> bool:
        """Close all open positions"""
        if not self.trading_client:
            return False
        
        try:
            positions = self.trading_client.get_all_positions()
            
            if not positions:
                logger.info("No open positions to close")
                return True
            
            logger.info(f"Closing {len(positions)} position(s)...")
            
            for position in positions:
                symbol = position.symbol
                qty = abs(float(position.qty))
                side = "sell" if float(position.qty) > 0 else "buy"
                
                self.place_order(symbol, qty, side)
            
            logger.info("✅ All positions closed")
            return True
            
        except Exception as e:
            logger.error(f"Error closing positions: {str(e)}")
            return False
    
    def get_account_status(self) -> Dict[str, Any]:
        """Get comprehensive account status"""
        if not self.trading_client:
            return {"error": "Trading client not initialized"}
        
        try:
            account = self.trading_client.get_account()
            positions = self.trading_client.get_all_positions()
            
            return {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "account_blocked": account.account_blocked,
                "num_positions": len(positions),
                "positions": [
                    {
                        "symbol": p.symbol,
                        "qty": float(p.qty),
                        "avg_entry_price": float(p.avg_entry_price),
                        "current_price": float(p.current_price),
                        "market_value": float(p.market_value),
                        "unrealized_pl": float(p.unrealized_pl),
                        "unrealized_plpc": float(p.unrealized_plpc)
                    }
                    for p in positions
                ]
            }
        except Exception as e:
            logger.error(f"Error getting account status: {str(e)}")
            return {"error": str(e)}
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        if not self.trading_client:
            return False
        
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {str(e)}")
            return False
    
    def wait_for_market_open(self, check_interval: int = 60):
        """Wait until market opens"""
        while not self.is_market_open():
            try:
                clock = self.trading_client.get_clock()
                time_to_open = (clock.next_open - clock.timestamp).total_seconds()
                
                logger.info(f"Market closed. Opens in {time_to_open/3600:.1f} hours")
                time.sleep(min(check_interval, time_to_open))
            except Exception as e:
                logger.error(f"Error waiting for market: {str(e)}")
                time.sleep(check_interval)


if __name__ == "__main__":
    # Test the client
    client = AlpacaClient()
    
    if client.trading_client:
        print("\n" + "="*70)
        print("  ALPACA CLIENT TEST")
        print("="*70 + "\n")
        
        # Account status
        status = client.get_account_status()
        print(f"Account Equity: ${status.get('equity', 0):,.2f}")
        print(f"Buying Power: ${status.get('buying_power', 0):,.2f}")
        print(f"Open Positions: {status.get('num_positions', 0)}")
        
        # Market status
        is_open = client.is_market_open()
        print(f"\nMarket Status: {'🟢 OPEN' if is_open else '🔴 CLOSED'}")
        
        # Test price fetch
        print("\nTesting price fetch...")
        btc_price = client.get_latest_price("BTC/USD", is_crypto=True)
        if btc_price:
            print(f"BTC/USD: ${btc_price:,.2f}")
        
        print("\n" + "="*70 + "\n")
