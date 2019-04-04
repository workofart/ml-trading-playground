from playground.utilities.utils import read_data, generate_datasets, plot_trades, log_histogram, log_scalars, get_latest_run_count, test_trades
import numpy as np, pandas as pd, random, os
import tensorflow as tf
from tqdm import tqdm
from matplotlib import pyplot as plt
from playground.dqn.dqn_agent import DQN_Agent
from playground.env.trading_env import TradingEnv

INIT_CASH = 1000
EPISODE = 5000 # Episode limitation
BATCH_SIZE = 32 # size of minibatch
DATA_LENGTH = 250 # How many times steps to use in the data
TRAIN_EVERY_N_TIMESTEPS = int(DATA_LENGTH / 10)

# Logging
SAVED_LOG_PATH = "playground/logs/dqn"
RUN_COUNT = str(get_latest_run_count(SAVED_LOG_PATH))

# Test Runs
TEST_EVERY_N_EPISODES = int(EPISODE / 100)
TEST_RUNS = 10 # Run test for N times to smooth out noise
PLOT_FREQ = int(EPISODE / 10)

# Reproducibility
SEED = 1992

def main(isLoad=False):
    env = TradingEnv(data_length=DATA_LENGTH, INIT_CASH=INIT_CASH)
    agent = DQN_Agent(env, EPISODE, isLoad=isLoad, seed=SEED)
    for i in tqdm(range(EPISODE)):
        agent.isTrain = True
        agent.is_updated_target_net = False
        state = agent.env.reset() # To start the process
        done = False
        agent.replay_buffer.clear()
        avg_reward_list = []
        actions_list = []
        while done is False:
            action = agent.act(state)
            state, reward, done, _ = agent.env.step(action)
            actions_list.append(action)
            avg_reward_list.append(reward)
            if done is False:
                next_state = agent.env._get_obs() # Get the next state
                agent.perceive(state, action, reward, next_state, done)
                if agent.replay_buffer.size() > BATCH_SIZE and env.time_step % TRAIN_EVERY_N_TIMESTEPS == 0:
                    agent.train_dqn_network(i, batch_size=BATCH_SIZE)
        # Update epsilon after every episode
        if agent.isTrain is True and agent.epsilon > agent.final_epsilon:
            agent.epsilon -= (1 - agent.final_epsilon) / (EPISODE/1.2)
        log_histogram(agent.summary_writer, 'reward_dist', avg_reward_list, i)
        log_scalars(agent.summary_writer, 'avg_reward', np.mean(avg_reward_list), i)
        log_scalars(agent.summary_writer, 'drawdown', np.mean(np.sum(np.array(avg_reward_list) < INIT_CASH, axis=0)), i)
        log_scalars(agent.summary_writer, 'action_errors', np.mean(agent.env.error_count), i)
        
        # print('# Buys: {} | {}'.format(str(actions_list.count(0)), (actions_list.count(0)/len(actions_list))))
        # print('# Sells: {} | {}'.format(str(actions_list.count(1)), (actions_list.count(1)/len(actions_list))))
        # print('# Holds: {} | {}'.format(str(actions_list.count(2)), (actions_list.count(2)/len(actions_list))))
        if i % TEST_EVERY_N_EPISODES == 0 and i > 0:
            test_trades(agent, i, os.path.join(SAVED_LOG_PATH, RUN_COUNT, 'test_trades'), TEST_RUNS, PLOT_FREQ)

if __name__ == '__main__':
    main()