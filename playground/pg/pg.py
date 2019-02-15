from utilities.utils import read_data, generate_datasets, plot_trades, plot_reward
import numpy as np, pandas as pd
from pg.pg_agent import PG_Agent
from env.trading_env import TradingEnv
# ---------------------------------------------------------
EPISODE = 10000 # Episode limitation
TEST = 10 # The number of experiment test every TEST_EVERY_N_BATCHES episode, for reducing variance
TEST_EVERY_N_BATCHES = 10
ITERATION = 2
BATCH_SIZE = 512 # size of minibatch

def main():
    # initialize OpenAI Gym env and dqn agent
    batch_counter = 0
    data = {}
    env = TradingEnv()
    agent = PG_Agent(env, data)
    state_list, reward_list, actions_taken = [], [], []
    avg_reward_list = []
    agg_avg_reward_list = []
    state = agent.env.reset() # To start the process
    
    for i in range(EPISODE):
        done = False
        agent.env.time_step = 0
        batch_counter = 0
        agent.replay_buffer.clear()
        state = agent.env.reset()
        avg_reward_list = []
        print('---- Episode %d ----' %(i))
        while done is False:
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

            # ----> remove If the current episode is done, calculate discounted & normalized reward
            # If the current batch is done
            # if done:
            if agent.env.time_step % BATCH_SIZE == 0:
                # episode_number += 1
                batch_counter += 1
                episode_reward = np.vstack(reward_list)
                discounted_epr = agent.discounted_rewards(episode_reward)
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)
                epdlogp = np.vstack(actions_taken)
                epdlogp *= discounted_epr
                agent.perceive(state_list, epdlogp)
                # state = agent.env.reset()
                # if batch_counter % BATCH_SIZE == 0:
                agent.train_pg_network(BATCH_SIZE)
        
                # After all the steps are completed, summarize the stats
                if batch_counter % TEST_EVERY_N_BATCHES == 0 and batch_counter >= TEST_EVERY_N_BATCHES:
                    total_reward = 0
                    # for test_num in range(TEST):
                        # state = agent.env.reset()
                        # for step in range(agent.env.STEPS_PER_EPISODE):
                    grad, action = agent.policy_forward(state) # direct action for test
                    state, reward, done, _ = agent.env.step(action)
                    total_reward += reward
                    # if done:
                        # break
                    # avg_reward = total_reward / TEST
                    avg_reward = total_reward
                    avg_reward_list.append(avg_reward)
                    print('Timestep: %s | Avg Reward: %.8f' % (agent.env.time_step, avg_reward))
                    # if episode_number % 1000 == 0:
                        # plot_trades([s[0][2] for s in state_list], [np.argmax(action) for action in actions_taken])
                    state = agent.env.reset()
                    # if i % 10 == 0 and i > 0: plot_trades([s[0][2] for s in state_list], [np.argmax(action) for action in actions_taken])
                state_list, reward_list, actions_taken = [], [], []
                # if episode_number % 5000 == 0: plot_reward(avg_reward_list)
        # plot_reward(avg_reward_list)
        print('Avg reward of last episode: ' + str(np.mean(avg_reward_list)))
        agg_avg_reward_list.append(np.mean(avg_reward_list))
        # if i % 10 == 0 and i > 0: plot_reward(agg_avg_reward_list)

if __name__ == '__main__':
    main()