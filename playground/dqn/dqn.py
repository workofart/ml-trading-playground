from playground.utilities.utils import read_data, generate_datasets, plot_trades, log_histogram, log_scalars, get_latest_run_count, test_trades, copy_config
import numpy as np, pandas as pd, random, os, configparser
import tensorflow as tf
from tqdm import tqdm
from matplotlib import pyplot as plt
from playground.dqn.dqn_agent import DQN_Agent
from playground.env.trading_env import TradingEnv

####### Housing Keeping #######

config_path = 'playground/dqn/config.ini'
config = configparser.ConfigParser()
config.read(config_path)
cfg = config['dqn']
paths_cfg = config['paths']
logistics_cfg = config['logistics']

INIT_CASH = int(cfg['INIT_CASH'])
EPISODE = int(cfg['EPISODE']) # Episode limitation
BATCH_SIZE = int(cfg['BATCH_SIZE']) # size of minibatch
DATA_LENGTH = int(cfg['DATA_LENGTH']) # How many times steps to use in the data
TRAIN_EVERY_N_TIMESTEPS = int(logistics_cfg['TRAIN_EVERY_N_TIMESTEPS'])
SAVED_LOG_PATH = paths_cfg['SAVED_LOG_PATH']
TEST_RUNS = cfg['TEST_RUNS'] # Run test for N times to smooth out noise
SEED = int(logistics_cfg['SEED'])

RUN_COUNT = str(get_latest_run_count(SAVED_LOG_PATH))
TEST_EVERY_N_EPISODES = max(int(EPISODE / 100), 1)
PLOT_FREQ = max(int(EPISODE / 100), 1)
TEST_TRADE_PATH = os.path.join(SAVED_LOG_PATH, RUN_COUNT, 'test_trades')

copy_config(SAVED_LOG_PATH, config_path, RUN_COUNT)

#################################

def main(isLoad=False):
    env = TradingEnv(data_length=DATA_LENGTH, INIT_CASH=INIT_CASH)
    agent = DQN_Agent(env, EPISODE, isLoad=isLoad, seed=SEED)
    for i in tqdm(range(EPISODE)):
        agent.isTrain = True
        state = agent.env.reset() # To start the process
        done = False
        agent.replay_buffer.clear()
        avg_reward_list = []
        actions_list = []
        while done is False:
            agent.accum_steps += 1
            action = agent.act(state)
            state, reward, done, _ = agent.env.step(action)
            actions_list.append(action)
            avg_reward_list.append(reward)
            if done is False:
                next_state = agent.env._get_obs() # Get the next state
                agent.perceive(state, action, reward, next_state, done)
                if agent.replay_buffer.size() > BATCH_SIZE and agent.accum_steps % TRAIN_EVERY_N_TIMESTEPS == 0:
                    agent.train_dqn_network(batch_size=BATCH_SIZE)
            agent.update_target_q_net_if_needed()
        agent.current_episode += 1
        log_scalars(agent.summary_writer, 'avg_reward', np.mean(avg_reward_list), i)

        # Complex Reward Function Stats
        # log_histogram(agent.summary_writer, 'reward_dist', avg_reward_list, i)
        # log_scalars(agent.summary_writer, 'drawdown', np.mean(np.sum(np.array(avg_reward_list) < INIT_CASH, axis=0)), i)
        # log_scalars(agent.summary_writer, 'action_errors', np.mean(agent.env.error_count), i)
        
        if i % TEST_EVERY_N_EPISODES == 0 and i > 0:
            test_trades(agent, i, TEST_TRADE_PATH, TEST_RUNS, PLOT_FREQ)

if __name__ == '__main__':
    main(False)