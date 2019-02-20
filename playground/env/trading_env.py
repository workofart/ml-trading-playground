import numpy as np
from playground.utilities.utils import read_data

# Inspired by https://github.com/openai/gym/blob/master/gym/envs/algorithmic/algorithmic_env.py


# Trading Params
INIT_CASH = 0
HOLD_PENALTY = 0.001
# HOLD_PENALTY = 0
# TXN_COST = 0.005
TXN_COST = 0
REPEAT_TRADE_THRESHOLD = 15

# Reward
MIN_REWARD = -1
MAX_REWARD = 1

class TradingEnv():

    def __init__(self, data_length):
        # Trading Params
        self.portfolio = 0
        self.cash = INIT_CASH

        self.time_step = 0
        self.data_length = data_length
        self.episode_total_reward = None
        self.action_space = [0, 1, 2]
        self.observation_space = read_data('crypto-test-data-82hrs.csv', 'ETHBTC')[0:data_length] * 1000
        self.previous_reward = 0
        self.reset()

        # Reward Function
        self.buys = 0
        self.sells = 0
        self.holds = 0

    # Reset method called between episodes
    def reset(self):
        self.episode_total_reward = 0
        self.time_step = 0
        self.portfolio = 0
        self.cash = INIT_CASH
        self.buys = 0
        self.sells = 0
        self.holds = 0
        return self._get_obs()

    def step(self, action, isTrain = False):
        done = False
        state = self._get_obs()
        reward = self.process_action(action, state)

        # Clip rewards between MIN_REWARD and MAX_REWARD
        reward = max(MIN_REWARD, min(MAX_REWARD, reward))

        if self.time_step >= self.observation_space.shape[0] - 1:
            done = True
        self.episode_total_reward += reward
        self.time_step += 1
        return state, reward, done, {}

    def _get_obs(self, pos=None):
        if pos is None:
            return np.reshape(self.observation_space[self.time_step], (1, 4))
        else:
            return np.reshape(self.observation_space[pos], (1, 4))

    def process_action(self, action, state):
        cur_price = state[0][2]
        # Buy
        if action == 0:
            # print('buying')
            self.cash -= cur_price * (1+TXN_COST)# buy with current price of current state
            self.portfolio += 1
            self.buys += 1
            self.sells = 0
            self.holds = 0
        # Sell
        elif action == 1:
            # print('selling')
            self.cash += cur_price * (1-TXN_COST) 
            self.portfolio -= 1
            self.sells += 1
            self.buys = 0
            self.holds = 0
        # Hold
        elif action == 2:
            # print('holding')
            self.cash = self.cash - HOLD_PENALTY
            self.portfolio = self.portfolio
            self.holds+= 1
            self.buys = 0
            self.sells = 0

        reward = self.cash + self.portfolio * cur_price - max(self.previous_reward, 0)

        # Enhance reward function based on behavior
        # if self.holds > REPEAT_TRADE_THRESHOLD or self.buys > REPEAT_TRADE_THRESHOLD or self.sells > REPEAT_TRADE_THRESHOLD:
        #     reward -= 0.2

        self.previous_reward = reward
        return reward
