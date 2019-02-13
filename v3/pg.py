from utilityV3.utils import read_data, generate_datasets, plot_trades, plot_reward
import numpy as np, pandas as pd
from pg_agent import PG_Agent
from trading_env import TradingEnv
# ---------------------------------------------------------
EPISODE = 10000 # Episode limitation
TEST = 10 # The number of experiment test every TEST_EVERY_N_EPISODES episode, for reducing variance
TEST_EVERY_N_EPISODES = 100
ITERATION = 2
BATCH_SIZE = 32 # size of minibatch

def main():
    # initialize OpenAI Gym env and dqn agent
    episode_number = 0
    data = {}
    env = TradingEnv()
    agent = PG_Agent(env, data)
    state_list, reward_list, actions_taken = [], [], []
    avg_reward_list = []
    state = agent.env.reset() # To start the process

    for i in range(ITERATION):
        while True:
            # TODO: everytime the reset is called, the timestep of the env
            # gets reset to 0, and the set of states in every iteration is always
            # the same set of prices, therefore, after a few episodes, the agent
            # is able to develop a winning policy, and stays there forever until
            # the state changes again
            grad, action = agent.policy_forward(state)
            state, reward, done, _ = agent.env.step(action, agent)
            state_list.append(state)
            actions_taken.append(grad)
            reward_list.append(reward)

            # If the current episode is done, calculate discounted & normalized reward
            if done:
                episode_number += 1
                episode_reward = np.vstack(reward_list)
                discounted_epr = agent.discounted_rewards(episode_reward)
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)
                epdlogp = np.vstack(actions_taken)
                epdlogp *= discounted_epr
                agent.perceive(state_list, epdlogp)
                state = agent.env.reset()
                if episode_number % BATCH_SIZE == 0:
                    agent.train_pg_network()
        
                # After all the steps are completed, summarize the stats
                if episode_number % TEST_EVERY_N_EPISODES == 0 and episode_number >= TEST_EVERY_N_EPISODES:
                    total_reward = 0
                    for test_num in range(TEST):
                        state = agent.env.reset()
                        for step in range(agent.env.STEPS_PER_EPISODE):
                            grad, action = agent.policy_forward(state) # direct action for test
                            state, reward, done, _ = agent.env.step(action, agent, True)
                            total_reward += reward
                            if done:
                                break
                    avg_reward = total_reward / TEST
                    avg_reward_list.append(avg_reward)
                    print('Episode: %s | Avg Reward: %.8f' % (episode_number, avg_reward))
                    # if episode_number % 1000 == 0:
                        # plot_trades([s[0][2] for s in state_list], [np.argmax(action) for action in actions_taken])
                    state = agent.env.reset()

                state_list, reward_list, actions_taken = [], [], []
                if episode_number % 5000 == 0: plot_reward(avg_reward_list)

if __name__ == '__main__':
    main()