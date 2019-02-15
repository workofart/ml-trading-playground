from playground.utilities.utils import read_data, generate_datasets, plot_trades, plot_reward
import numpy as np, pandas as pd
from playground.dqn.dqn_agent import DQN_Agent
from playground.env.trading_env import TradingEnv

EPISODE = 30 # Episode limitation
TEST = 3 # The number of experiment test every 1 episode, for reducing variance
TRAIN_EVERY_STEPS = 200
BATCH_SIZE = 32 # size of minibatch


def main():
    env = TradingEnv()
    agent = DQN_Agent(env)
    state = agent.env.reset() # To start the process
    reward_plot_data = []
    for i in range(EPISODE):
        done = False
        agent.replay_buffer.clear()
        state = agent.env.reset()
        print('---- Episode %d ----' %(i))
        while done is False:
            # TODO: everytime the reset is called, the timestep of the env
            # gets reset to 0, and the set of states in every iteration is always
            # the same set of prices, therefore, after a few episodes, the agent
            # is able to develop a winning policy, and stays there forever until
            # the state changes again
            action = agent.act(state)
            state, reward, done, _ = agent.env.step(action)
            agent.perceive(state, action, reward, agent.env.next_state, done)
            if len(agent.replay_buffer) > BATCH_SIZE and env.time_step % TRAIN_EVERY_STEPS == 0:
                agent.train_dqn_network(BATCH_SIZE)
    
        # After one full episode is completed, test for TEST rounds
        avg_reward_list = []
        for t in range(TEST):
            state = agent.env.reset()
            avg_reward = 0
            total_reward = 0
            done = False
            while done is False:
                action = agent.act(state)
                state, reward, done, _ = agent.env.step(action)
                total_reward += reward
            avg_reward = total_reward / TEST
            avg_reward_list.append(avg_reward)

        print('Avg Reward: %.8f' % (np.mean(avg_reward_list)))
        reward_plot_data.append(np.mean(avg_reward_list))
    
    plot_reward(reward_plot_data)


def test():
    env = TradingEnv()
    agent = DQN_Agent(env)
    agent.isTrain = False
    state = agent.env.reset() # To start the process

    prices = []
    actions = []
    for i in range(1000):
        prices.append(state[0][2])
        action = agent.act(state)
        actions.append(action)
        state, reward, done, _ = env.step(action)
    plot_trades(prices, actions)



if __name__ == '__main__':
    # main()
    test()