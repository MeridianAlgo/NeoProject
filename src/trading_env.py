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
        
        self.df = df.copy()
        self.initial_balance = initial_balance
        
        # Action space: 0=Neutral, 1=Long, 2=Short
        self.action_space = spaces.Discrete(3)
        
        # Calculate volatility for adaptive time horizons
        self.df['volatility'] = self.df['close'].pct_change().rolling(20).std()
        self.df['momentum'] = self.df['close'].pct_change(5)  # 5-period momentum
        self.df.fillna(0, inplace=True)
        
        # Observation space: Market Features + 8 account/position states
        # [Market Features, Balance/Init, NetWorth/Init, Pos, Steps, UnrealizedPnL, Volatility, Momentum, MaxDrawdown]
        num_features = len(self.df.columns) + 8
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32)
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.last_net_worth = self.initial_balance
        self.peak_net_worth = self.initial_balance  # Track for drawdown
        
        # Position tracking
        self.position = 0 # 0: None, 1: Long, -1: Short
        self.entry_price = 0
        self.shares_held = 0 
        self.steps_in_position = 0 # Time Horizon awareness
        
        # Metrics tracking
        self.total_trades = 0
        self.profitable_trades = 0
        self.net_worth_history = [self.initial_balance]
        
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

        # Calculate current drawdown from peak
        drawdown = (self.net_worth - self.peak_net_worth) / self.peak_net_worth if self.peak_net_worth > 0 else 0
        
        # Get current volatility and momentum
        current_volatility = current_data['volatility']
        current_momentum = current_data['momentum']
        
        # Adaptive time horizon based on volatility (higher vol = shorter horizon)
        adaptive_horizon = self.steps_in_position / (100.0 * (1 + current_volatility * 10))

        # State: [Market Features, Balance/Initial, NetWorth/Initial, Position, AdaptiveHorizon, UnrealizedPnL, Volatility, Momentum, Drawdown]
        obs = np.concatenate((
            current_data.values,
            [
                self.balance / self.initial_balance,
                self.net_worth / self.initial_balance,
                self.position,
                adaptive_horizon,
                unrealized_pnl_pct,
                current_volatility * 100,  # Scale up for visibility
                current_momentum * 100,
                drawdown
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
                pnl = self._close_position(current_price)
                self.total_trades += 1
                if pnl > 0:
                    self.profitable_trades += 1
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
        self.net_worth_history.append(self.net_worth)
        
        # Track peak for drawdown calculation
        if self.net_worth > self.peak_net_worth:
            self.peak_net_worth = self.net_worth
        
        # Reward: Percentage change in Net Worth (more stable than absolute $)
        pct_change = (self.net_worth - self.last_net_worth) / self.last_net_worth
        reward = pct_change * 100  # Scale to percentage points
        
        # Adaptive time horizon penalty based on volatility
        current_volatility = self.df.iloc[self.current_step]['volatility']
        if self.position != 0 and abs(pct_change) < 0.0001:  # Less than 0.01% change
            # Higher volatility = more tolerance for holding
            penalty_factor = max(0.5, 1.0 - current_volatility * 5)
            reward -= 0.01 * (self.steps_in_position / 10.0) * penalty_factor
        
        # Reward for reducing drawdown
        drawdown = (self.net_worth - self.peak_net_worth) / self.peak_net_worth
        if drawdown < -0.05:  # More than 5% drawdown
            reward -= abs(drawdown) * 10  # Penalize drawdowns
            
        self.last_net_worth = self.net_worth
        
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        # Close position at the end of the episode for metrics
        if terminated and self.position != 0:
             pnl = self._close_position(current_price)
             self.total_trades += 1
             if pnl > 0:
                self.profitable_trades += 1
        
        info = {
            'net_worth': self.net_worth,
            'current_price': current_price,
            'position': self.position,
            'steps_in_pos': self.steps_in_position,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'win_rate': (self.profitable_trades / self.total_trades) if self.total_trades > 0 else 0
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
        pnl = 0
        if self.position == 1: # Closing Long
            self.balance = self.shares_held * current_price
            pnl = self.balance - (self.shares_held * self.entry_price)
        elif self.position == -1: # Closing Short
            pnl = (self.entry_price - current_price) * self.shares_held
            initial_value = self.shares_held * self.entry_price
            self.balance = initial_value + pnl
            
        self.position = 0
        self.shares_held = 0
        self.entry_price = 0
        return pnl
        
    def _update_net_worth(self, current_price):
        if self.position == 0:
            self.net_worth = self.balance
        elif self.position == 1: # Long
            self.net_worth = self.shares_held * current_price
        elif self.position == -1: # Short
            initial_value = self.shares_held * self.entry_price
            pnl = (self.entry_price - current_price) * self.shares_held
            self.net_worth = initial_value + pnl

