# Neo v4: The Recurrent Autonomous Trader

Neo v4 is a unified, high-performance trading AI that uses **Recurrent PPO (LSTM)** for deep time-series pattern recognition. It handles its own database, scales its own training, and executes trades autonomously via the **Alpaca API**.

## 🚀 Key Improvements
- **Unified Logic**: `neo.py` is the single entry point for training, backtesting, and live trading.
- **Deep Memory**: Uses Recurrent Neural Networks (LSTM) to remember market cycles.
- **Time-Horizon Awareness**: The AI is now explicitly aware of its "time-in-position" and "unrealized PnL," allowing it to optimize for efficient exits.
- **Alpaca Integration**: Supports Paper and Live trading out of the box.
- **Anti-Overtraining**: Implements rigorous validation cycles and checkpointing.

## 🛠️ Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install sb3-contrib alpaca-trade-api
   ```

2. **Configure API Keys**:
   Create a `.env` file in the root directory:
   ```env
   ALPACA_API_KEY=your_key
   ALPACA_SECRET_KEY=your_secret
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   WANDB_API_KEY=your_wandb_key
   ```

## 📈 Usage

### 1. Train Neo (Recurrent LSTM)
Train the model on the full 10-year BTC history:
```bash
python neo.py --train --steps 250000
```
*Monitored live via Weights & Biases.*

### 2. Autonomous Trading
Run the bot in a continuous loop. It will sync the latest data, warm up its LSTM memory, and execute orders via Alpaca:
```bash
python neo.py --trade
```

## 🧠 Architecture
- **Model**: `RecurrentPPO` (LSTM-based Policy).
- **Features**: 30+ dimensions including Market Indicators, Portfolio Status, and Time-Metrics.
- **Database**: SQLite (`data/trading_bot.db`).
- **Monitoring**: WandB.
