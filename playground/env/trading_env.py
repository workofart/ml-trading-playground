import numpy as np
from playground.utilities.utils import read_data, cleanup_logs, get_minute_data, log_scalars

# Inspired by https://github.com/openai/gym/blob/master/gym/envs/algorithmic/algorithmic_env.py


# Trading Params
HOLD_PENALTY = 0
# TXN_COST = 0.0005
TXN_COST = 0
REPEAT_TRADE_THRESHOLD = 15

# Reward
MIN_REWARD = -2
MAX_REWARD = 2

class TradingEnv():

    def __init__(self, data_length, INIT_CASH):
        # Trading Params
        self.portfolio = []
        self.INIT_CASH = INIT_CASH
        self.cash = self.INIT_CASH

        self.time_step = 0
        self.data_length = data_length
        self.episode_total_reward = None
        self.action_space = [0, 1, 2]
        self.observation_space = read_data('crypto-test-data-82hrs.csv', 'ETHBTC', 'm')[0:data_length] * 1000
        self.previous_portfolio = self.cash
        self.reset()

    # Reset method called between episodes
    def reset(self):
        self.episode_total_reward = 0
        self.time_step = 0
        self.portfolio = []
        self.cash = self.INIT_CASH
        self.error_count = 0
        self.permitted_trades = []
        return self._get_obs()

    def step(self, action, isTrain = False):
        done = False
        state = self._get_obs()
        reward, mv = self.process_action(action, state)

        # Clip rewards between MIN_REWARD and MAX_REWARD
        # reward = max(MIN_REWARD, min(MAX_REWARD, reward))

        if self.time_step >= self.observation_space.shape[0] - 1:
            done = True
            
        self.episode_total_reward += reward
        self.time_step += 1
        return state, reward, done, {'marketValue': mv}

    def _get_obs(self, pos=None):
        if pos is None:
            return np.vstack(self.observation_space[self.time_step]).T            
        else:
            return np.vstack(self.observation_space[pos]).T

    def process_action(self, action, state):
        cur_price = state[0][2]
        error = False
        # Buy
        if action == 0:
            if cur_price > self.cash:
                error = True
            else:
                self.cash -= cur_price * (1+TXN_COST)# buy with current price of current state
                # self.portfolio += 1
                self.portfolio.append(cur_price)
                self.permitted_trades.append(0)
        # Sell
        elif action == 1:
            if self._get_holding_stock_count() == 0:
                error = True
            else:
                self.cash += cur_price * (1-TXN_COST)
                self.portfolio.append(-cur_price)
                self.permitted_trades.append(1)
        # Hold
        elif action == 2:
            self.cash = self.cash - HOLD_PENALTY
            self.portfolio = self.portfolio
            self.permitted_trades.append(2)

        book_val = self.cash + self._avg_holding_price() * self._get_holding_stock_count()
        market_val = self.cash + cur_price * self._get_holding_stock_count()
        
        reward = book_val + (market_val - self.previous_portfolio)

        if error:
            reward -= 0.03
            self.error_count += 1
            self.permitted_trades.append(2)

        # Previous reward should be the previous portfolio market value, not the difference
        self.previous_portfolio = market_val
        return reward, market_val

    def _avg_holding_price(self):
        if self._get_holding_stock_count() > 0:
            return np.sum(self.portfolio) / self._get_holding_stock_count()
        else:
            return 0
    
    def _get_holding_stock_count(self):
        return np.sum(np.array(self.portfolio) > 0, axis=0) - np.sum(np.array(self.portfolio) < 0, axis=0)