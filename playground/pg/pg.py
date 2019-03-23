from playground.utilities.utils import read_data, generate_datasets, plot_trades, plot_reward
import numpy as np, pandas as pd
from playground.pg.pg_agent import PG_Agent
from playground.env.trading_env import TradingEnv
# ---------------------------------------------------------
EPISODE = 200 # Episode limitation
TEST_EVERY_N_EPISODES = 10
BATCH_SIZE = 64 # size of minibatch
TRAIN_EVERY_TIMESTEP = 100

def main():
    env = TradingEnv(1000, 100)
    agent = PG_Agent(env)
    avg_reward_list = []
    state = agent.env.reset() # To start the process
    
    for i in range(EPISODE):
        done = False
        agent.replay_buffer.clear()
        state = agent.env.reset()
        state_list, reward_list, one_hot_actions = [], [], []
        print('---- Episode %d ----' %(i))
        while done is False:
            one_hot_action, action = agent.act(state)
            state, reward, done, _ = agent.env.step(action, agent)
            state_list.append(state)
            one_hot_actions.append(one_hot_action)
            reward_list.append(reward)

            # TODO: Need revision
            # episode_reward = np.vstack(reward_list)
            # discounted_epr = agent.discounted_rewards(episode_reward)
            # discounted_epr -= np.mean(discounted_epr)
            # discounted_epr /= np.std(discounted_epr)
            # epdlogp = np.vstack(one_hot_actions)
            # epdlogp *= discounted_epr
            # agent.perceive(state, epdlogp[-1])
            agent.perceive(state, one_hot_action)

            if len(agent.replay_buffer) >= BATCH_SIZE and agent.env.time_step % TRAIN_EVERY_TIMESTEP == 0:
                agent.train_pg_network(i, BATCH_SIZE)

            # After all the steps are completed, summarize the stats
            if done is True and i % TEST_EVERY_N_EPISODES == 0 and i >= 0:
                test(agent, i)
        
        # Summary after one EP
        avg_reward_list.append(np.mean(reward_list))    
        print('[TRAIN] - EP{0} | Avg reward: {1}'.format(i, avg_reward_list[-1]))

def test(agent, i):
    agent.isTrain = False
    state = agent.env.reset()
    reward_list = []
    done = False
    while done is False:
        one_hot_action, action = agent.act(state) # direct action for test
        state, reward, done, _ = agent.env.step(action)
        reward_list.append(reward)
    print('[TEST] - EP{0} | Avg Reward: {1}'.format(i, np.mean(reward_list)))

if __name__ == '__main__':
    main()