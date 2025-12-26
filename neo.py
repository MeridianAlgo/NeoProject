import os
import argparse
import time
import pandas as pd
import numpy as np
import wandb
import json
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from wandb.integration.sb3 import WandbCallback

import src.data_fetcher as data_fetcher
import src.database as database
from src.trading_env import TradingEnv
from src.alpaca_client import AlpacaClient

from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

def linear_schedule(initial_value, min_value=1e-5):
    """
    Linear learning rate schedule with a minimum floor.
    :param initial_value: (float) Initial learning rate.
    :param min_value: (float) Minimum learning rate to prevent complete decay.
    :return: (function)
    """
    def func(progress_remaining):
        """
        :param progress_remaining: (float) 1 (start) to 0 (end)
        :return: (float)
        """
        # Linear decay with floor: never goes below min_value
        return max(progress_remaining * initial_value, min_value)
    return func

class CustomLoggerCallback(BaseCallback):
    """
    Suppresses terminal spam and shows a clean progress bar with ETA.
    Pipes all training metrics directly to WandB.
    """
    def __init__(self, total_timesteps, verbose=0):
        super(CustomLoggerCallback, self).__init__(verbose)
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.last_step = 0
        self.is_ci = os.getenv("GITHUB_ACTIONS") == "true"

    def _on_training_start(self):
        # Configure tqdm for less spam in CI
        if self.is_ci:
            self.pbar = tqdm(
                total=self.total_timesteps, 
                desc="Training Neo", 
                unit="steps",
                mininterval=30.0,  # Only update every 30 seconds in CI
                maxinterval=60.0,
                ascii=True         # Use plain ascii for better CI rendering
            )
        else:
            self.pbar = tqdm(
                total=self.total_timesteps, 
                desc="Training Neo", 
                unit="steps",
                mininterval=0.5
            )

    def _on_step(self):
        # Update progress bar
        if self.pbar:
            steps = self.num_timesteps - self.last_step
            self.pbar.update(steps)
            self.last_step = self.num_timesteps
        
        # Log to WandB periodically
        if self.n_calls % 100 == 0:
            metrics = {}
            for key, value in self.model.logger.name_to_value.items():
                metrics[f"train/{key}"] = value
            
            # Add to WandB every 100 steps to avoid overhead
            wandb.log(metrics, step=self.num_timesteps)
            
        return True

    def _on_training_end(self):
        self.pbar.close()

class FinancialMetricsCallback(BaseCallback):
    """
    Performs periodic evaluation of financial metrics and logs them to WandB.
    """
    def __init__(self, eval_env, ticker, verbose=0):
        super(FinancialMetricsCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.ticker = ticker

    def _on_step(self):
        # We perform final evaluation at the end of training for simplicity
        # but could also do it periodically here.
        return True

    def _on_training_end(self):
        # Already handled by the main train function after learn()
        pass

class NeoBot:
    def __init__(self, ticker="BTC-USD"):
        """Initialize bot with a single ticker for focused training."""
        self.ticker = ticker
        self.model_path = "models/neo_recurrent_v4"
        self.stats_path = "models/neo_stats_v4.pkl"
        self._alpaca = None

    @property
    def alpaca(self):
        """Lazy initialization of AlpacaClient."""
        if self._alpaca is None:
            self._alpaca = AlpacaClient()
        return self._alpaca

    def sync_data(self):
        """Pulls from yfinance for the current ticker."""
        print(f"Syncing data for {self.ticker}...")
        df = data_fetcher.get_processed_data(self.ticker)
        
        if df is None or df.empty:
            raise ValueError(f"No data fetched for {self.ticker}!")
        
        print(f"Total data points for {self.ticker}: {len(df)}")
        return df

    def get_env(self, df, training=True):
        """Creates a normalized environment."""
        env_maker = lambda: TradingEnv(df)
        env = DummyVecEnv([env_maker])
        
        if training:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
        else:
            if os.path.exists(self.stats_path):
                env = VecNormalize.load(self.stats_path, env)
                env.training = False
                env.norm_reward = False
            else:
                print("Warning: Normalization stats not found for inference.")
                env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
        
        return env

    def train(self, total_timesteps=50000, use_wandb=True):
        """Trains the Recurrent PPO model."""
        df = self.sync_data()
        
        # Split 90/10 for Time Series
        split = int(len(df) * 0.9)
        train_df = df.iloc[:split]
        test_df = df.iloc[split:]
        
        env = self.get_env(train_df, training=True)
        eval_env = self.get_env(test_df, training=True)  # Keep training=True for proper normalization
        # Don't sync stats - let eval_env track its own statistics
        
        # Callbacks
        callbacks = []
        eval_callback = EvalCallback(
            eval_env, 
            best_model_save_path='./models/best_neo/',
            log_path='./logs/', 
            eval_freq=5000,
            n_eval_episodes=5,
            deterministic=True,
            verbose=0 # Stop eval spam too
        )
        callbacks.append(eval_callback)
        callbacks.append(CustomLoggerCallback(total_timesteps))
        
        if use_wandb:
            run = wandb.init(
                project="neo-v4",
                name=f"train_{self.ticker}_{time.strftime('%Y%m%d_%H%M%S')}",
                tags=[self.ticker, "RecurrentPPO", "LSTM"],
                config={
                    "ticker": self.ticker,
                    "algorithm": "RecurrentPPO",
                    "total_timesteps": total_timesteps,
                    "n_steps": 2048,
                    "batch_size": 256,
                    "learning_rate": "2e-4 -> 1e-6",
                    "lstm_hidden_size": 256,
                    "adaptive_time_horizons": True,
                    "gradient_clipping": 0.5,
                    "gae_lambda": 0.98,
                    "vf_coef": 1.0,
                    "data_points": len(df)
                },
                sync_tensorboard=True,
                save_code=True
            )
            # Standard WandbCallback for architectural logging
            callbacks.append(WandbCallback())

        # Recurrent PPO Model with Stability Improvements
        model = RecurrentPPO(
            "MlpLstmPolicy", 
            env, 
            verbose=0, # ZERO SPAM
            learning_rate=linear_schedule(2e-4, min_value=1e-6),  # More conservative LR
            n_steps=2048,  # Increased significantly for stable gradients (2048 samples per update)
            batch_size=256,  # Larger batches for better gradient estimation
            n_epochs=10,  # Standard PPO epochs
            gamma=0.995,  # Slightly higher discount for longer-term planning
            gae_lambda=0.98,  # Increased for better advantage estimation
            clip_range=0.2,  # PPO clipping
            clip_range_vf=0.2,  # Added value function clipping for stability
            ent_coef=0.015,  # Slightly more exploration
            vf_coef=1.0,  # Increased weight on value learning to boost explained_variance
            max_grad_norm=0.5,  # CRITICAL: Gradient clipping to prevent explosion
            policy_kwargs=dict(
                net_arch=[256, 256],  # Larger network for complex patterns
                lstm_hidden_size=256,  # Larger LSTM memory
                n_lstm_layers=1,
                enable_critic_lstm=True  # LSTM for value function too
            )
        )
        
        print(f"Starting Training: {total_timesteps} steps...")
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
        
        # Performs a final detailed financial evaluation on the test set
        self.final_evaluation(model, test_df, use_wandb=use_wandb, train_env=env)

        # Save
        model.save(self.model_path)
        env.save(self.stats_path)
        print(f"Model and Stats saved to {self.model_path}")
        
        if use_wandb: run.finish()

    def final_evaluation(self, model, test_df, use_wandb=True, train_env=None):
        """
        Runs the model once over the entire test set and logs detailed metrics.
        """
        print(f"\n📊 Performing final financial evaluation for {self.ticker}...")
        
        # Create a fresh evaluation environment (no normalization here for raw metrics)
        eval_env = TradingEnv(test_df)
        obs, _ = eval_env.reset()
        
        net_worths = []
        prices = []
        positions = []
        cumulative_rewards = []
        daily_returns = []
        
        episode_starts = np.ones((1,), dtype=bool)
        lstm_states = None
        
        done = False
        while not done:
            # Important: Normalize the observation using the same stats as training
            if train_env is not None:
                norm_obs = train_env.normalize_obs(np.array([obs]))
            else:
                norm_obs = obs
 
            action, lstm_states = model.predict(
                norm_obs, 
                state=lstm_states, 
                episode_start=episode_starts, 
                deterministic=True
            )
            
            # Use item() to safely extract the scalar action, whether it's 0-dim or 1-dim
            action_val = action.item() if isinstance(action, np.ndarray) else action
            obs, reward, terminated, truncated, info = eval_env.step(action_val)
            done = terminated or truncated
            episode_starts = [done]
            
            net_worths.append(info['net_worth'])
            prices.append(info['current_price'])
            positions.append(info['position'])
            cumulative_rewards.append(info.get('cumulative_reward', 0))
            
            if len(net_worths) > 1:
                daily_returns.append((net_worths[-1] - net_worths[-2]) / net_worths[-2])
            else:
                daily_returns.append(0)
        
        # Calculate Metrics
        returns = np.diff(net_worths) / net_worths[:-1]
        is_crypto = "BTC" in self.ticker or "ETH" in self.ticker
        trading_days = 365 if is_crypto else 252
        
        avg_ret = np.mean(returns)
        std_ret = np.std(returns)
        sharpe = (avg_ret / std_ret) * np.sqrt(trading_days) if std_ret > 0 else 0
        
        total_return = (net_worths[-1] - net_worths[0]) / net_worths[0]
        bh_return = (prices[-1] - prices[0]) / prices[0]
        
        # SMA Crossover Baseline (SMA 20 vs 50)
        # We simulate this on the test_df
        sma_net_worth = [1.0]
        sma_pos = 0 # 0: flat, 1: long
        for i in range(1, len(test_df)):
            row = test_df.iloc[i]
            prev_row = test_df.iloc[i-1]
            
            # Crossover logic
            if prev_row['sma_20'] <= prev_row['sma_50'] and row['sma_20'] > row['sma_50']:
                sma_pos = 1
            elif prev_row['sma_20'] >= prev_row['sma_50'] and row['sma_20'] < row['sma_50']:
                sma_pos = 0
            
            # Update net worth
            price_change = (row['close'] - prev_row['close']) / prev_row['close']
            current_worth = sma_net_worth[-1] * (1 + (price_change if sma_pos == 1 else 0))
            sma_net_worth.append(current_worth)
        
        sma_return = sma_net_worth[-1] - 1.0

        # Drawdown
        peak = net_worths[0]
        max_drawdown = 0
        for nw in net_worths:
            if nw > peak: peak = nw
            dd = (nw - peak) / peak
            if dd < max_drawdown: max_drawdown = dd
            
        win_rate = info.get('win_rate', 0)
        trades = info.get('total_trades', 0)

        print(f"🏁 Eval Results ({self.ticker}):")
        print(f"   Total Return:  {total_return*100:.2f}%")
        print(f"   B&H Return:    {bh_return*100:.2f}%")
        print(f"   SMA Return:    {sma_return*100:.2f}%")
        print(f"   Sharpe Ratio:  {sharpe:.2f}")
        print(f"   Max Drawdown:  {max_drawdown*100:.2f}%")
        print(f"   Win Rate:      {win_rate*100:.2f}% ({trades} trades)")
        
        if use_wandb:
            # 1. Log summary metrics
            wandb.run.summary["eval/total_return_pct"] = total_return * 100
            wandb.run.summary["eval/buy_and_hold_return_pct"] = bh_return * 100
            wandb.run.summary["eval/sma_crossover_return_pct"] = sma_return * 100
            wandb.run.summary["eval/sharpe_ratio"] = sharpe
            wandb.run.summary["eval/max_drawdown_pct"] = max_drawdown * 100
            wandb.run.summary["eval/win_rate_pct"] = win_rate * 100
            wandb.run.summary["eval/total_trades"] = trades
            wandb.run.summary["eval/final_net_worth"] = net_worths[-1]
            
            # 2. Log custom curves
            # Normalize for comparison plot
            neo_curve = [nw / net_worths[0] for nw in net_worths]
            bh_curve = [p / prices[0] for p in prices]
            sma_curve = sma_net_worth
            
            # Performance Comparison (Normalized)
            wandb.log({
                "eval/portfolio_comparison": wandb.plot.line_series(
                    xs=[i for i in range(len(neo_curve))],
                    ys=[neo_curve, bh_curve, sma_curve],
                    keys=["Neo Portfolio", f"Buy & Hold {self.ticker}", "SMA Crossover (20/50)"],
                    title=f"Neo vs Baselines ({self.ticker})",
                    xname="Day",
                    yname="Relative Performance"
                )
            })

            # Cumulative Rewards
            wandb.log({
                "eval/cumulative_reward_plot": wandb.plot.line_series(
                    xs=[i for i in range(len(cumulative_rewards))],
                    ys=[cumulative_rewards],
                    keys=["Cumulative Reward"],
                    title=f"Learning Signal Stability ({self.ticker})",
                    xname="Day",
                    yname="Sum of Rewards"
                )
            })
            
            # Absolute Net Worth ($)
            wandb.log({
                "eval/absolute_profit": wandb.plot.line_series(
                    xs=[i for i in range(len(net_worths))],
                    ys=[net_worths],
                    keys=["Net Worth ($)"],
                    title=f"Neo Absolute Net Worth ({self.ticker})",
                    xname="Day",
                    yname="Portfolio Value ($)"
                )
            })

            # Holdings/Position over time
            wandb.log({
                "eval/position_strategy": wandb.plot.line_series(
                    xs=[i for i in range(len(positions))],
                    ys=[positions],
                    keys=["Position (0=Flat, 1=Long, -1=Short)"],
                    title=f"Neo Trading Activity ({self.ticker})",
                    xname="Day",
                    yname="Position State"
                )
            })

            # Drawdown Curve
            dd_curve = []
            current_peak = net_worths[0]
            for nw in net_worths:
                if nw > current_peak: current_peak = nw
                dd_curve.append((nw - current_peak) / current_peak * 100)
            
            wandb.log({
                "eval/drawdown_plot": wandb.plot.line_series(
                    xs=[i for i in range(len(dd_curve))],
                    ys=[dd_curve],
                    keys=["Drawdown (%)"],
                    title=f"Neo Drawdown ({self.ticker})",
                    xname="Day",
                    yname="Drawdown (%)"
                )
            })
            
            # Log separate curves for clarity in WandB dashboard (Step-wise logging)
            for i in range(len(neo_curve)):
                wandb.log({
                    "eval/neo_value": neo_curve[i],
                    "eval/bh_value": bh_curve[i],
                    "eval/sma_value": sma_curve[min(i, len(sma_curve)-1)],
                    "eval/net_worth": net_worths[i],
                    "eval/drawdown": dd_curve[i],
                    "eval/position": positions[i],
                    "eval/cumulative_reward": cumulative_rewards[i],
                    "eval/daily_return": daily_returns[i],
                    "eval/trading_step": i
                }, commit=False)
            
            wandb.log({"eval/completed": 1})

        # Save results to local JSON for GitHub Actions/CI consumption
        results = {
            "ticker": self.ticker,
            "total_return_pct": float(total_return * 100),
            "buy_and_hold_return_pct": float(bh_return * 100),
            "sma_return_pct": float(sma_return * 100),
            "sharpe_ratio": float(sharpe),
            "max_drawdown_pct": float(max_drawdown * 100),
            "win_rate_pct": float(win_rate * 100),
            "total_trades": int(trades),
            "final_net_worth": float(net_worths[-1])
        }
        with open("eval_results.json", "w") as f:
            json.dump(results, f, indent=4)
        print(f"✅ Evaluation results saved to eval_results.json")

    def run_rigorous_tests(self, use_wandb=True):
        """Perform chronological Out-of-Sample (OOS) and Walk-forward tests."""
        print(f"\n{'='*70}")
        print(f"  RIGOROUS ROBUSTNESS TESTING: {self.ticker}")
        print(f"{'='*70}\n")
        
        df = self.sync_data()
        
        # Chronological Split: 2024 for training, 2025 for OOS testing
        # Or relative splits if data is shorter
        split_point = int(len(df) * 0.8)
        train_df = df.iloc[:split_point]
        test_df = df.iloc[split_point:]
        
        print(f"Training Period: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df)} days)")
        print(f"OOS Test Period: {test_df.index[0]} to {test_df.index[-1]} ({len(test_df)} days)")
        
        # Train locally for testing
        env = self.get_env(train_df, training=True)
        
        if use_wandb:
            wandb.init(
                project="neo-v4-rigorous",
                name=f"rigorous_{self.ticker}_{time.strftime('%Y%m%d_%H%M%S')}",
                config={"ticker": self.ticker, "mode": "rigorous_oos"}
            )

        model = RecurrentPPO(
            "MlpLstmPolicy", env, verbose=0,
            learning_rate=1e-4, n_steps=1024, batch_size=128
        )
        
        print(f"Training base model for OOS test (25k steps)...")
        model.learn(total_timesteps=25000)
        
        # Run evaluation on OOS data
        print(f"Running Out-of-Sample Evaluation...")
        self.final_evaluation(model, test_df, use_wandb=use_wandb, train_env=env)
        
        if use_wandb: wandb.finish()
        print(f"\n✅ Rigorous testing complete for {self.ticker}")

    def run_autonomous(self):
        """The loop that runs the bot independently with live Alpaca data."""
        print("\n" + "="*70)
        print("  NEO v4 AUTONOMOUS TRADING MODE")
        print("="*70 + "\n")
        
        model_path = "./models/best_neo/best_model"
        if not os.path.exists(model_path + ".zip"):
            print(f"❌ Model not found at {model_path}. Please train first.")
            return

        # Load model
        print("📦 Loading trained model...")
        model = RecurrentPPO.load(model_path)
        print(f"✅ Model loaded successfully")
        print(f"🎯 Trading ticker: {self.ticker}\n")
        
        # Check Alpaca connection
        if not self.alpaca.trading_client:
            print("❌ Alpaca not configured. Cannot trade.")
            print("   Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
            return
        
        # Get account status
        status = self.alpaca.get_account_status()
        print(f"💰 Account Equity: ${status.get('equity', 0):,.2f}")
        print(f"💵 Buying Power: ${status.get('buying_power', 0):,.2f}")
        print(f"📊 Open Positions: {status.get('num_positions', 0)}\n")
        
        # Determine if crypto or stock
        is_crypto = "BTC" in self.ticker or "ETH" in self.ticker
        symbol = self.ticker.replace("-", "/") if is_crypto else self.ticker
        
        print(f"Asset Type: {'Crypto' if is_crypto else 'Stock'}")
        print(f"Trading Symbol: {symbol}\n")
        
        # Start WebSocket stream
        print(f"📡 Initializing real-time WebSocket stream for {symbol}...")
        self.alpaca.start_streaming([symbol], is_crypto=is_crypto)
        time.sleep(2) # Give it a moment to connect
        
        print("="*70 + "\n")
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while True:
            try:
                # 1. Check sleep hours
                current_hour = time.localtime().tm_hour
                if 23 <= current_hour or current_hour < 7:
                    print(f"😴 Sleep hours detected ({current_hour}:00). Skipping trading.")
                    print("   Waiting 1 hour...\n")
                    time.sleep(3600)
                    continue
                
                # 2. Check market hours (for stocks)
                if not is_crypto:
                    if not self.alpaca.is_market_open():
                        print("🔴 Market is closed. Waiting...")
                        self.alpaca.wait_for_market_open(check_interval=300)
                        continue
                
                print(f"\n{'='*70}")
                print(f"  TRADING CYCLE - {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*70}\n")
                
                # 3. Get latest price from Alpaca
                print(f"📡 Fetching live price for {symbol}...")
                current_price = self.alpaca.get_latest_price(symbol, is_crypto=is_crypto)
                
                if current_price is None:
                    print(f"⚠️  Could not fetch price for {symbol}")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"❌ Too many consecutive errors ({consecutive_errors}). Stopping.")
                        break
                    time.sleep(60)
                    continue
                
                print(f"💵 Current Price: ${current_price:,.2f}\n")
                consecutive_errors = 0  # Reset on success
                
                # 4. Sync historical data for context
                print("📊 Syncing historical data...")
                df = self.sync_data()
                
                if df is None or df.empty:
                    print("⚠️  No historical data available")
                    time.sleep(300)
                    continue
                
                # 5. Create environment and warm up LSTM
                print("🧠 Warming up LSTM memory...")
                env = self.get_env(df.tail(200), training=False)  # Use last 200 days
                obs = env.reset()
                
                lstm_states = None
                episode_starts = np.ones((1,), dtype=bool)
                
                # Warm up LSTM with recent history
                for i in range(min(200, len(df))):
                    action, lstm_states = model.predict(
                        obs,
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=True
                    )
                    obs, reward, done, info = env.step(action)
                    episode_starts = done
                    if done[0]:
                        break
                
                # 6. Get final prediction
                action_val = action[0]
                action_names = ["NEUTRAL", "LONG", "SHORT"]
                predicted_action = action_names[action_val]
                
                print(f"\n🤖 AI PREDICTION: {predicted_action}")
                print(f"   Confidence: Based on {len(df)} days of data\n")
                
                # 7. Execute trade
                self.execute_trade_robust(
                    action=action_val,
                    symbol=symbol,
                    current_price=current_price,
                    is_crypto=is_crypto
                )
                
                # 8. Display updated account status
                status = self.alpaca.get_account_status()
                print(f"\n📊 Updated Account Status:")
                print(f"   Equity: ${status.get('equity', 0):,.2f}")
                print(f"   Positions: {status.get('num_positions', 0)}")
                
                # 9. Wait before next cycle
                wait_time = 300  # 5 minutes for crypto, adjust for stocks
                print(f"\n⏰ Next check in {wait_time//60} minutes...")
                print(f"{'='*70}\n")
                time.sleep(wait_time)
                
            except KeyboardInterrupt:
                print("\n\n🛑 Shutting down gracefully...")
                print("   Closing all positions...")
                self.alpaca.close_all_positions()
                print("   Goodbye! 👋\n")
                break
                
            except Exception as e:
                print(f"\n❌ Error in trading loop: {str(e)}")
                import traceback
                traceback.print_exc()
                consecutive_errors += 1
                
                if consecutive_errors >= max_consecutive_errors:
                    print(f"\n❌ Too many errors ({consecutive_errors}). Stopping for safety.")
                    break
                
                print(f"   Waiting 60 seconds before retry...\n")
                time.sleep(60)

    def execute_trade_robust(self, action: int, symbol: str, current_price: float, is_crypto: bool = True):
        """Execute trade with robust error handling and position management"""
        try:
            # Get current position
            current_qty = self.alpaca.get_position(symbol)
            account_equity = self.alpaca.get_account_balance()
            
            print(f"📍 Current Position: {current_qty} {symbol}")
            
            # Calculate position size (use 90% of equity for safety)
            max_position_value = account_equity * 0.90
            max_qty = max_position_value / current_price
            
            # Round quantity appropriately
            if is_crypto:
                max_qty = round(max_qty, 6)  # Crypto can have many decimals
            else:
                max_qty = int(max_qty)  # Stocks are whole shares
            
            # Action 0: NEUTRAL - Close all positions
            if action == 0:
                if abs(current_qty) > 0:
                    print(f"🔄 NEUTRAL signal - Closing position")
                    side = "sell" if current_qty > 0 else "buy"
                    self.alpaca.place_order(symbol, abs(current_qty), side)
                else:
                    print(f"✅ NEUTRAL signal - Already flat")
            
            # Action 1: LONG - Buy/hold long position
            elif action == 1:
                if current_qty < 0:
                    # Close short first
                    print(f"🔄 Closing short position...")
                    self.alpaca.place_order(symbol, abs(current_qty), "buy")
                    time.sleep(2)  # Wait for order to fill
                
                if current_qty <= 0:
                    # Open long
                    print(f"📈 LONG signal - Opening position")
                    print(f"   Buying {max_qty} {symbol} @ ${current_price:,.2f}")
                    self.alpaca.place_order(symbol, max_qty, "buy")
                else:
                    print(f"✅ LONG signal - Already long")
            
            # Action 2: SHORT - Sell/hold short position
            elif action == 2:
                if current_qty > 0:
                    # Close long first
                    print(f"🔄 Closing long position...")
                    self.alpaca.place_order(symbol, abs(current_qty), "sell")
                    time.sleep(2)
                
                if current_qty >= 0:
                    # Open short
                    print(f"📉 SHORT signal - Opening position")
                    print(f"   Shorting {max_qty} {symbol} @ ${current_price:,.2f}")
                    self.alpaca.place_order(symbol, max_qty, "sell")
                else:
                    print(f"✅ SHORT signal - Already short")
            
        except Exception as e:
            print(f"❌ Error executing trade: {str(e)}")
            import traceback
            traceback.print_exc()

    def execute_alpaca(self, action, current_price):
        """Converts AI signal to Alpaca orders."""
        if not self.alpaca.api:
            print("Alpaca API not configured. Skipping execution.")
            return

        # Simple logic: Go Long, Go Short, or Close Everything
        current_equity = self.alpaca.get_account_balance()
        symbol = self.ticker.replace("-", "")  # BTCUSD
        
        # Simplified crypto symbols for Alpaca
        if "BTC" in symbol: symbol = "BTCUSD"
            
        current_qty = self.alpaca.get_position(symbol)
        
        # 1. Neutral Action
        if action == 0:
            if current_qty != 0:
                side = 'sell' if current_qty > 0 else 'buy'
                self.alpaca.place_order(symbol, abs(current_qty), side)
        
        # 2. Long Action
        elif action == 1:
            if current_qty <= 0:
                # Close short if any
                if current_qty < 0: self.alpaca.place_order(symbol, abs(current_qty), 'buy')
                # Buy full position
                buy_qty = (current_equity * 0.95) / current_price # 95% of equity
                self.alpaca.place_order(symbol, round(buy_qty, 4), 'buy')
        
        # 3. Short Action
        elif action == 2:
            if current_qty >= 0:
                # Close long if any
                if current_qty > 0: self.alpaca.place_order(symbol, abs(current_qty), 'sell')
                # Shorting crypto on Alpaca is specific, but let's assume standard flow
                # For many assets, you just sell more than you have
                sell_qty = (current_equity * 0.95) / current_price
                self.alpaca.place_order(symbol, round(sell_qty, 4), 'sell')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--steps", type=int, default=50000, help="Training steps")
    parser.add_argument("--ticker", type=str, default="BTC-USD", help="Ticker to train on (BTC-USD, ETH-USD, SPY, QQQ)")
    parser.add_argument("--trade", action="store_true", help="Run autonomous trading")
    parser.add_argument("--test", action="store_true", help="Run rigorous robustness tests")
    args = parser.parse_args()

    bot = NeoBot(ticker=args.ticker)
    if args.train:
        print(f"\n{'='*60}")
        print(f"  TRAINING NEO v4 ON {args.ticker}")
        print(f"  Steps: {args.steps:,}")
        print(f"{'='*60}\n")
        bot.train(total_timesteps=args.steps)
    elif args.test:
        bot.run_rigorous_tests()
    elif args.trade:
        bot.run_autonomous()
    else:
        parser.print_help()
