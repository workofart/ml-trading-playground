import numpy as np
from utilities.utils import read_data

# Inspired by https://github.com/openai/gym/blob/master/gym/envs/algorithmic/algorithmic_env.py


# Trading Params
INIT_CASH = 5
HOLD_PENALTY = 0.0005
# TXN_COST = 0.005
TXN_COST = 0

class TradingEnv():

    def __init__(self):
        # Trading Params
        self.portfolio = 0
        self.cash = INIT_CASH

        self.time_step = 0
        self.episode_total_reward = None
        self.action_space = [0, 1, 2]
        self.observation_space = read_data('crypto-test-data-82hrs.csv', 'ETHBTC').values * 10
        self.STEPS_PER_EPISODE = 2000
        self.reset()


    # Reset method called between episodes
    def reset(self):
        self.episode_total_reward = 0
        self.time_step = 0
        self.portfolio = 0
        self.cash = INIT_CASH
        return self._get_obs()

    def step(self, action, isTrain = False):
        done = False
        state = self._get_obs()
        reward = self.process_action(action, state)
        # if self.time_step > self.STEPS_PER_EPISODE - 1:
        if self.time_step >= self.observation_space.shape[0] - 1:
            done = True
        self.episode_total_reward += reward
        self.time_step += 1
        return state, reward, done, {}

    def _get_obs(self, pos=None):
        if pos is None:
            return np.reshape(np.concatenate((self.observation_space[self.time_step], [self.cash], [self.portfolio])), (1, 6))
            # np.reshape(self.observation_space[self.time_step], (1, self.observation_space.shape[1]))

    def process_action(self, action, state):
        # Buy
        if action == 0:
            # print('buying')
            self.cash -= state[0][2] * (1+TXN_COST)# buy with current price of current state
            self.portfolio += 1
        # Sell
        elif action == 1:
            # print('selling')
            self.cash += state[0][2] * (1-TXN_COST) 
            self.portfolio -= 1
        # Hold
        elif action == 2:
            # print('holding')
            self.cash = self.cash - HOLD_PENALTY
            self.portfolio = self.portfolio
        
        reward = self.cash + self.portfolio * state[0][2]
        return reward
