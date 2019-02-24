from playground.utilities.utils import read_data, generate_datasets, plot_trades, log_histogram, log_scalars
import numpy as np, pandas as pd, random
import tensorflow as tf
from tqdm import tqdm
from matplotlib import pyplot as plt
from playground.dqn.dqn_agent import DQN_Agent
from playground.env.trading_env import TradingEnv

EPISODE = 3000 # Episode limitation
TRAIN_EVERY_STEPS = 16
TEST_EVERY_EP = 100
BATCH_SIZE = 32 # size of minibatch
DATA_LENGTH = 250 # How many times steps to use in the data

INIT_CASH = 100

# reproducible
random.seed(1992)
np.random.seed(1992)
tf.set_random_seed(1992)

def main(isLoad=False):
    env = TradingEnv(data_length=DATA_LENGTH, INIT_CASH=INIT_CASH)
    agent = DQN_Agent(env, isLoad=isLoad)
    # plt.ion()
    # plt.show()
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
                if agent.replay_buffer.size() > BATCH_SIZE and env.time_step % TRAIN_EVERY_STEPS == 0:
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
        if i % TEST_EVERY_EP == 0 and i > 0:
            test(agent, i)

def test(agent, ep = 0):
    agent.isTrain = False
    state = agent.env.reset() # To start the process

    prices = []
    actions = []
    for i in range(DATA_LENGTH):
        prices.append(state[0][2])
        action = agent.act(state)
        actions.append(action)
        state, reward, done, _ = agent.env.step(action)
    plot_trades(ep, prices, actions, agent.env.permitted_trades)



if __name__ == '__main__':
    main()