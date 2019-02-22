from playground.utilities.utils import read_data, generate_datasets, plot_trades, plot_data, log_histogram
import numpy as np, pandas as pd, random
import tensorflow as tf
from tqdm import tqdm
from matplotlib import pyplot as plt
from playground.dqn.dqn_agent import DQN_Agent
from playground.env.trading_env import TradingEnv

EPISODE = 3000 # Episode limitation
TRAIN_EVERY_STEPS = 12
TEST_EVERY_EP = 50
BATCH_SIZE = 256 # size of minibatch
DATA_LENGTH = 800 # How many times steps to use in the data

def main():
    env = TradingEnv(data_length = DATA_LENGTH)
    agent = DQN_Agent(env)
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
                    c = agent.train_dqn_network(i, BATCH_SIZE)
        # Update epsilon after every episode
        if agent.isTrain is True and agent.epsilon > agent.final_epsilon:
            agent.epsilon -= (1 - agent.final_epsilon) / (EPISODE/1.5)
        log_histogram(agent.summary_writer, 'reward_dist', avg_reward_list, i)
        avg_reward_summary = tf.Summary(value=[tf.Summary.Value(tag='avg_reward', simple_value=np.mean(avg_reward_list))])
        agent.summary_writer.add_summary(avg_reward_summary, i)
        drawdown_summary = tf.Summary(value=[tf.Summary.Value(tag='drawdown', simple_value=np.sum(np.array(avg_reward_list) < 0, axis=0))])
        agent.summary_writer.add_summary(drawdown_summary, i)
        agent.summary_writer.flush()

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
    plot_trades(ep, prices, actions)



if __name__ == '__main__':
    main()