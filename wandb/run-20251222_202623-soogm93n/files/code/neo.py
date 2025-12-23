import os
import argparse
import time
import pandas as pd
import numpy as np
import wandb
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

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Neo", unit="steps")

    def _on_step(self):
        # Calculate steps since last call
        steps = self.n_calls - self.last_step
        self.pbar.update(steps)
        self.last_step = self.n_calls
        
        # Pull metrics from the logger
        # Standard SB3 metrics are available in self.logger.name_to_value
        metrics = {}
        for key, value in self.model.logger.name_to_value.items():
            metrics[f"train/{key}"] = value
        
        # Add to WandB every 100 steps to avoid overhead
        if self.n_calls % 100 == 0:
            wandb.log(metrics, step=self.num_timesteps)
            
        return True

    def _on_training_end(self):
        self.pbar.close()

class NeoBot:
    def __init__(self, ticker="BTC-USD"):
        self.ticker = ticker
        self.model_path = "models/neo_recurrent_v4"
        self.stats_path = "models/neo_stats_v4.pkl"
        self.alpaca = AlpacaClient()

    def sync_data(self):
        """Pulls from yfinance, pushes to SQLite, and returns processed DF."""
        print(f"Syncing data for {self.ticker}...")
        return data_fetcher.get_processed_data(self.ticker)

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

    def train(self, total_timesteps=100000, use_wandb=True):
        """Trains the Recurrent PPO model."""
        df = self.sync_data()
        
        # Split 90/10 for Time Series
        split = int(len(df) * 0.9)
        train_df = df.iloc[:split]
        test_df = df.iloc[split:]
        
        env = self.get_env(train_df, training=True)
        eval_env = self.get_env(test_df, training=False)
        eval_env.obs_rms = env.obs_rms # Sync stats
        
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
                config={
                    "ticker": self.ticker,
                    "algorithm": "RecurrentPPO",
                    "total_timesteps": total_timesteps,
                },
                sync_tensorboard=True,
                save_code=True
            )
            # Standard WandbCallback for architectural logging
            callbacks.append(WandbCallback())

        # Recurrent PPO Model
        model = RecurrentPPO(
            "MlpLstmPolicy", 
            env, 
            verbose=0, # ZERO SPAM
            learning_rate=3e-4,
            n_steps=128, 
            batch_size=64,
            gae_lambda=0.95,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[128, 128])
        )
        
        print(f"Starting Training: {total_timesteps} steps...")
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
        
        # Save
        model.save(self.model_path)
        env.save(self.stats_path)
        print(f"Model and Stats saved to {self.model_path}")
        
        if use_wandb: run.finish()

    def run_autonomous(self):
        """The loop that runs the bot independently."""
        print("Starting Neo Autonomous Mode...")
        model = RecurrentPPO.load("./models/best_neo/best_model")
        
        while True:
            # 1. Sync latest data
            df = self.sync_data()
            
            # 2. Get latest observation
            env = self.get_env(df, training=False)
            obs = env.reset()
            
            # RecurrentPPO needs LSTM states
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            
            # Step through to the end of the history to get the latest state
            # This "warms up" the LSTM memory
            print("Warming up LSTM memory...")
            for i in range(len(df)):
                action, lstm_states = model.predict(
                    obs, 
                    state=lstm_states, 
                    episode_start=episode_starts,
                    deterministic=True
                )
                obs, reward, done, info = env.step(action)
                episode_starts = done
                if done[0]: break
            
            # 3. Final Signal
            last_info = info[0]
            action_val = action[0]
            action_str = ["NEUTRAL", "LONG", "SHORT"][action_val]
            print(f"LATEST SIGNAL: {action_str} | Price: ${last_info['current_price']:.2f}")
            
            # 4. Execute via Alpaca
            self.execute_alpaca(action_val, last_info['current_price'])
            
            # 5. Wait for next candle (Daily bot -> Wait 1 hour or check again)
            print("Sleeping for 1 hour...")
            time.sleep(3600)

    def execute_alpaca(self, action, current_price):
        """Converts AI signal to Alpaca orders."""
        if not self.alpaca.api:
            print("Alpaca API not configured. Skipping execution.")
            return

        # Simple logic: Go Long, Go Short, or Close Everything
        current_equity = self.alpaca.get_account_balance()
        symbol = self.ticker.replace("-", "") # BTCUSD
        
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
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--trade", action="store_true")
    args = parser.parse_args()

    bot = NeoBot()
    if args.train:
        bot.train(total_timesteps=args.steps)
    elif args.trade:
        bot.run_autonomous()
    else:
        parser.print_help()
