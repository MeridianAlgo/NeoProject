# 🤖 Neo v4: Advanced Autonomous Trading AI

[![Training Status](https://img.shields.io/badge/Training-Every%2030%20Minutes-brightgreen)](https://github.com/MeridianAlgo/NeoProject/actions)
[![Model](https://img.shields.io/badge/Model-RecurrentPPO%20LSTM-blue)](https://wandb.ai/your-username/neo-v4)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)

Neo v4 is a **state-of-the-art autonomous trading AI** powered by **Recurrent PPO (LSTM)** for deep time-series pattern recognition across multiple asset classes. It features adaptive time horizons, multi-ticker training, and fully autonomous execution via the **Alpaca API**.

## 🎯 What Makes Neo Special

### **Advanced Architecture**
- **Recurrent LSTM Policy**: 256-unit LSTM with dual-layer architecture for deep market memory
- **Rotating Ticker Training**: Trains on BTC, ETH, SPY, and QQQ in rotation every 2 hours
- **Adaptive Time Horizons**: Dynamically adjusts holding periods based on market volatility
- **Drawdown-Aware**: Actively penalizes large drawdowns to preserve capital

### **Smart Features**
- **Volatility-Adaptive Penalties**: Higher volatility = more tolerance for position holding
- **Momentum Tracking**: 5-period momentum signals integrated into decision-making
- **Risk Management**: Tracks peak net worth and current drawdown in real-time
- **Gradient Clipping**: Prevents catastrophic training collapse (max_grad_norm=0.5)

### **Production-Ready**
- **Automated Training**: GitHub Actions trains on different tickers every 2 hours
- **Continuous Learning**: Each ticker gets 3 training sessions per day (12 total/day)
- **Live Monitoring**: Full W&B integration with ticker-specific runs
- **Safe Execution**: Built-in sleep hours protection (11 PM - 7 AM)
- **Auto-Tagging**: Creates GitHub releases after each training run

## 📊 Training Metrics

Our model is continuously monitored on [Weights & Biases](https://wandb.ai/your-username/neo-v4):

**Key Metrics Tracked:**
- `train/value_loss` - Value function accuracy
- `train/policy_gradient_loss` - Policy optimization
- `train/explained_variance` - Model prediction quality (target: >0.8)
- `train/learning_rate` - Adaptive LR decay (3e-4 → 5e-5)
- `train/updates` - Total training iterations

**Current Performance:**
- Training Steps: 50,000 per cycle
- Explained Variance: Stable >0.5
- Multi-Asset Learning: 4 tickers simultaneously

## 🛠️ Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the root directory:
```env
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
WANDB_API_KEY=your_wandb_key
```

### 3. Initialize Database
The database will auto-initialize on first run, pulling historical data for:
- **BTC-USD** (Bitcoin)
- **ETH-USD** (Ethereum)
- **SPY** (S&P 500 ETF)
- **QQQ** (Nasdaq 100 ETF)

## 📈 Usage

### Train Locally
Train the model on a specific ticker:
```bash
# Train on Bitcoin
python neo.py --train --steps 50000 --ticker BTC-USD

# Train on Ethereum
python neo.py --train --steps 50000 --ticker ETH-USD

# Train on S&P 500
python neo.py --train --steps 50000 --ticker SPY

# Train on Nasdaq 100
python neo.py --train --steps 50000 --ticker QQQ
```

### Autonomous Trading
Run the bot in continuous trading mode (uses the ticker specified at initialization):
```bash
python neo.py --trade --ticker BTC-USD
```

**Safety Features:**
- Skips trading during sleep hours (11 PM - 7 AM)
- Validates model existence before execution
- Warms up LSTM memory with historical data
- Logs all decisions with timestamps

## 🧠 Technical Architecture

### Model Configuration
```python
RecurrentPPO(
    policy="MlpLstmPolicy",
    learning_rate=linear_schedule(3e-4, min_value=5e-5),
    n_steps=512,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    max_grad_norm=0.5,  # Critical for stability
    policy_kwargs={
        "net_arch": [256, 256],
        "lstm_hidden_size": 256,
        "enable_critic_lstm": True
    }
)
```

### Observation Space (40+ Features)
**Market Data:**
- OHLCV (Open, High, Low, Close, Volume)
- Technical Indicators: SMA(20/50/200), RSI, MACD, Bollinger Bands, ATR, OBV, CCI
- Log Returns & Lags (1, 2, 3, 7 periods)
- Volatility (5 & 20 period)

**Position State:**
- Balance/Initial Balance Ratio
- Net Worth/Initial Balance Ratio
- Current Position (-1: Short, 0: Neutral, 1: Long)
- Adaptive Time Horizon (volatility-adjusted)
- Unrealized PnL %
- Current Volatility
- Momentum Signal
- Drawdown from Peak

### Reward Function
```python
# Percentage-based returns (stable for value function)
reward = (net_worth_change / last_net_worth) * 100

# Adaptive time penalty (volatility-aware)
if position_flat:
    penalty_factor = max(0.5, 1.0 - volatility * 5)
    reward -= 0.01 * steps_in_position * penalty_factor

# Drawdown penalty (capital preservation)
if drawdown < -5%:
    reward -= abs(drawdown) * 10
```

## 🔄 Automated Training Pipeline

GitHub Actions workflow runs **every 2 hours** (12 times daily):

1. **Data Sync**: Fetches latest market data for all tickers
2. **Training**: 50,000 steps with full monitoring
3. **Model Save**: Updates best model checkpoint
4. **Database Commit**: Pushes updated data to repository
5. **W&B Logging**: All metrics synced to cloud

View live training: [GitHub Actions](https://github.com/MeridianAlgo/NeoProject/actions)

## 📁 Project Structure

```
Neo/
├── neo.py                      # Main entry point
├── src/
│   ├── trading_env.py         # Custom Gym environment
│   ├── data_fetcher.py        # Multi-ticker data pipeline
│   ├── database.py            # SQLite management
│   └── alpaca_client.py       # Trading execution
├── models/
│   ├── neo_recurrent_v4.zip   # Trained model
│   ├── neo_stats_v4.pkl       # Normalization stats
│   └── best_neo/              # Best checkpoints
├── data/
│   └── trading_bot.db         # Historical data
├── .github/workflows/
│   └── daily_train.yml        # Automated training
└── requirements.txt
```

## 🎓 Key Improvements from Previous Versions

### Fixed Issues:
✅ **Explained Variance Collapse** - Added gradient clipping and learning rate floor  
✅ **Reward Scaling** - Switched to percentage-based returns  
✅ **Overfitting** - Reduced steps from 1.2M to 50K per cycle  
✅ **Value Function Divergence** - Proper normalization and adaptive penalties  

### New Features:
🆕 **Multi-Ticker Training** - Generalizes across 4 different assets  
🆕 **Adaptive Time Horizons** - Volatility-aware position management  
🆕 **Drawdown Tracking** - Real-time risk monitoring  
🆕 **Enhanced LSTM** - 256-unit memory with critic LSTM  

## 📊 Monitoring & Debugging

### Check Training Progress
```bash
# View W&B dashboard
wandb login
# Your runs will appear at: https://wandb.ai/your-username/neo-v4
```

### Inspect Database
```python
python src/inspect_db.py
```

### View Model Stats
```python
from stable_baselines3 import RecurrentPPO
model = RecurrentPPO.load("models/neo_recurrent_v4")
print(model.policy)
```

## ⚠️ Important Notes

1. **Paper Trading First**: Always test with Alpaca paper trading before live
2. **Monitor Explained Variance**: Should stay above 0.5, ideally >0.8
3. **Drawdown Limits**: Bot penalizes >5% drawdowns heavily
4. **Sleep Hours**: Trading disabled 11 PM - 7 AM for safety

## 🔮 Future Enhancements

- [ ] Multi-timeframe analysis (1h, 4h, 1d)
- [ ] Ensemble models (combine multiple checkpoints)
- [ ] Dynamic position sizing based on confidence
- [ ] Options trading integration
- [ ] Sentiment analysis from news/social media

## 📝 License

MIT License - Trade at your own risk. Past performance does not guarantee future results.

---

**Built with ❤️ by MeridianAlgo**  
*Powered by Stable-Baselines3, W&B, and Alpaca Markets*
