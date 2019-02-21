from playground.utilities.utils import read_data, generate_datasets, plot_trades, plot_reward, plot_data
import numpy as np, pandas as pd, random
from matplotlib import pyplot as plt
from playground.dqn.dqn_agent import DQN_Agent
from playground.env.trading_env import TradingEnv

EPISODE = 2000 # Episode limitation
TEST = 3 # The number of experiment test every 1 episode, for reducing variance
TRAIN_EVERY_STEPS = 12
TEST_EVERY_EP = 10
BATCH_SIZE = 64 # size of minibatch
TEST_STEPS = 4999
DATA_LENGTH = 1000 # How many times steps to use in the data

def main():
    env = TradingEnv(data_length = DATA_LENGTH)
    agent = DQN_Agent(env)
    reward_plot_data = []
    cost_plot_data = []
    drawdown_data = []
    q_val_data = []
    plt.ion()
    plt.show()
    for i in range(EPISODE):
        print('---- Episode %d ----' %(i))
        if i > 450:
            agent.epsilon = agent.final_epsilon
        else:
            agent.epsilon = 1 # need to reset epsilon for every new episode
        agent.isTrain = True
        
        agent.q_vals = []
        state = agent.env.reset() # To start the process
        done = False
        agent.replay_buffer.clear()
        cost_per_episode = []
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
                    if c is not None:
                        cost_per_episode.append(c)
        reward_plot_data.append(np.mean(avg_reward_list))
        drawdown_data.append(np.sum(np.array(avg_reward_list) < 0, axis=0))
        # plot_reward(avg_reward_list)
        print('Avg Reward: %.8f' % (np.mean(avg_reward_list)))
        print('# Buys: {} | {}'.format(str(actions_list.count(0)), (actions_list.count(0)/len(actions_list))))
        print('# Sells: {} | {}'.format(str(actions_list.count(1)), (actions_list.count(1)/len(actions_list))))
        print('# Holds: {} | {}'.format(str(actions_list.count(2)), (actions_list.count(2)/len(actions_list))))
        cost_plot_data.append(np.mean(cost_per_episode))
        q_val_data.append(np.mean(agent.q_vals))


        # After one full timestep (timestep to end) is completed, test for TEST rounds
        # if i % TEST_EVERY_EP == 0:
        #     avg_reward_list = []
        #     for t in range(TEST):
        #         state = agent.env.reset()
        #         random_time_step = random.randint(0, DATA_LENGTH - TEST_STEPS - 1)
        #         agent.env.time_step = random_time_step
        #         avg_reward = 0
        #         # total_reward = 0
        #         done = False
        #         # while done is False:
        #         for step in range(TEST_STEPS):
        #             action = agent.act(state)
        #             state, reward, done, _ = agent.env.step(action)
        #         avg_reward = agent.env.episode_total_reward / TEST
        #         avg_reward_list.append(avg_reward)

        #     print('Avg Reward: %.8f' % (np.mean(avg_reward_list)))
        #     reward_plot_data.append(np.mean(avg_reward_list))
        plot_data(reward_plot_data, cost_plot_data, q_val_data, drawdown_data)
        if i % TEST_EVERY_EP == 0 and i > 0:
            test(agent, i)
    
    # plot_reward(reward_plot_data)


def test(agent, ep = 0):
    # env = TradingEnv(int(DATA_LENGTH / 4))
    # env = TradingEnv(DATA_LENGTH)
    # agent = DQN_Agent(env, False)
    agent.isTrain = False
    state = agent.env.reset() # To start the process

    prices = []
    actions = []
    # for i in range(int(DATA_LENGTH / 4)):
    for i in range(DATA_LENGTH):
        prices.append(state[0][2])
        action = agent.act(state)
        actions.append(action)
        state, reward, done, _ = agent.env.step(action)
    plot_trades(ep, prices, actions)



if __name__ == '__main__':
    main()
    # test()