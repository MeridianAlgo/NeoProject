import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Advanced Trading Environment.
    Actions:
    0: Neutral (Cash)
    1: Long (Buy)
    2: Short (Sell)
    
    This allows the bot to make money when the price goes DOWN.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=10000):
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        
        # Action space: 0=Neutral, 1=Long, 2=Short
        self.action_space = spaces.Discrete(3)
        
        # Observation space matches data columns + 5 account states
        # [Market Features (25), Balance/Init, NetWorth/Init, Pos, Steps, UnrealizedPnL]
        num_features = len(df.columns) + 5
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32)
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.last_net_worth = self.initial_balance
        
        # Position tracking
        self.position = 0 # 0: None, 1: Long, -1: Short
        self.entry_price = 0
        self.shares_held = 0 
        self.steps_in_position = 0 # Time Horizon awareness
        
        return self._get_observation(), {}

    def _get_observation(self):
        current_data = self.df.iloc[self.current_step]
        
        # Calculate unrealized PnL % for current position as a feature
        unrealized_pnl_pct = 0
        if self.position != 0 and self.entry_price != 0:
            current_price = current_data['close']
            if self.position == 1: # Long
                unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
            else: # Short
                unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price

        # State: [Market Features, Balance/Initial, NetWorth/Initial, Position, StepsInPos, UnrealizedPnL]
        # Normalizing scalars helps the LSTM learn faster
        obs = np.concatenate((
            current_data.values,
            [
                self.balance / self.initial_balance,
                self.net_worth / self.initial_balance,
                self.position,
                self.steps_in_position / 100.0, # Scale down for network
                unrealized_pnl_pct
            ]
        ))
        return obs.astype(np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        
        # Logic:
        # If Action is same as current Position, do nothing (Hold).
        # If Action is different, Close current position (if any) and Open new one.
        
        reward = 0
        
        if self.position != action:
            # 1. Close existing
            if self.position != 0:
                self._close_position(current_price)
                self.steps_in_position = 0
            
            # 2. Open new
            if action != 0:
                self._open_position(action, current_price)
                self.steps_in_position = 0
        else:
            if self.position != 0:
                self.steps_in_position += 1
            
        # Update Step
        self.current_step += 1
        self._update_net_worth(current_price)
        
        # Reward: Change in Net Worth
        reward = self.net_worth - self.last_net_worth
        
        # Time Horizon Incentive: Penalize staying in a flat position too long 
        # (Encourages efficiency)
        if self.position != 0 and abs(reward) < 0.01:
            reward -= 0.001 * self.steps_in_position
            
        self.last_net_worth = self.net_worth
        
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        info = {
            'net_worth': self.net_worth,
            'current_price': current_price,
            'position': self.position,
            'steps_in_pos': self.steps_in_position
        }
        
        return self._get_observation(), reward, terminated, truncated, info
        
    def _open_position(self, action, price):
        # Commit full Net Worth (All-in strategy)
        if action == 1: # LONG
             self.shares_held = self.net_worth / price
             self.balance = 0 
             self.entry_price = price
             self.position = 1
        elif action == 2: # SHORT
             self.shares_held = self.net_worth / price
             self.entry_price = price
             self.position = -1
             
    def _close_position(self, current_price):
        if self.position == 1: # Closing Long
            self.balance = self.shares_held * current_price
        elif self.position == -1: # Closing Short
            pnl = (self.entry_price - current_price) * self.shares_held
            initial_value = self.shares_held * self.entry_price
            self.balance = initial_value + pnl
            
        self.position = 0
        self.shares_held = 0
        self.entry_price = 0
        
    def _update_net_worth(self, current_price):
        if self.position == 0:
            self.net_worth = self.balance
        elif self.position == 1: # Long
            self.net_worth = self.shares_held * current_price
        elif self.position == -1: # Short
            initial_value = self.shares_held * self.entry_price
            pnl = (self.entry_price - current_price) * self.shares_held
            self.net_worth = initial_value + pnl

