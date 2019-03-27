from playground.utilities.utils import read_data, generate_datasets, plot_trades, plot_reward, get_latest_run_count
import numpy as np, pandas as pd
import os
import tensorflow as tf
from playground.pg.pg_agent import PG_Agent
from playground.env.trading_env import TradingEnv
# ---------------------------------------------------------
EPISODE = 1000 # Episode limitation
TEST_EVERY_N_EPISODES = EPISODE / 10
BATCH_SIZE = 64 # size of minibatch
TRAIN_EVERY_TIMESTEP = 500

def main():
    with tf.Session() as sess:
        env = TradingEnv(1000, 10)
        agent = PG_Agent(env, sess)
        sess.run(tf.global_variables_initializer())
        state = agent.env.reset() # To start the process
    
        for i in range(EPISODE):
            done = False
            state = agent.env.reset()
            agent.one_hot_actions.clear()
            agent.rewards.clear()
            agent.states.clear()
            while done is False:
                one_hot_action, action = agent.act(state)
                state, reward, done, _ = agent.env.step(action, agent)
                agent.rewards.append(reward)
                agent.states.append(state)
                agent.one_hot_actions.append(one_hot_action)

                agent.discount_rewards()

                # if len(agent.states) >= BATCH_SIZE and agent.env.time_step % TRAIN_EVERY_TIMESTEP == 0:
            agent.train_pg_network(i, BATCH_SIZE)

            # Update epsilon after every episode
            if agent.isTrain is True and agent.epsilon > agent.FINAL_EPSILON:
                agent.epsilon -= (1 - agent.FINAL_EPSILON) / (EPISODE/1.2)

            # Summary after one EP, test and summarize the stats
            if i % TEST_EVERY_N_EPISODES == 0 and i >= 0:
                test(agent, i)
            summary = sess.run(agent.write_op, feed_dict={agent._inputs: np.vstack(np.array(agent.states)),
                                          agent._actions: np.vstack(np.array(agent.one_hot_actions)),
                                          agent._discounted_rewards: np.vstack(np.array(agent.discounted_rewards)),
                                          agent._mean_reward: np.mean(agent.rewards)
                                        })
            agent.writer.add_summary(summary, i)
            agent.writer.flush()

def test(agent, i):
    agent.isTrain = False
    state = agent.env.reset()
    reward_list = []
    actions = []
    prices = []
    done = False
    while done is False:
        prices.append(state[0][2])
        one_hot_action, action = agent.act(state) # direct action for test
        state, reward, done, _ = agent.env.step(action)
        actions.append(action)
        reward_list.append(reward)
    print('EP{0} | TEST | Avg Reward: {1}'.format(i, np.mean(reward_list)))
    plot_trades(i, prices, actions, None, name='PG_Test', path=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs', 'pg', 'test_trades')))
    agent.isTrain = True

if __name__ == '__main__':
    main()