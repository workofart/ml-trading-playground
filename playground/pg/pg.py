from playground.utilities.utils import read_data, generate_datasets, plot_trades, plot_reward, get_latest_run_count, log_scalars, test_trades
import numpy as np, pandas as pd
import os, tqdm, random
import tensorflow as tf
from playground.pg.pg_agent import PG_Agent
from playground.env.trading_env import TradingEnv
# ---------------------------------------------------------
EPISODE = 50000 # Episode limitation
TEST_EVERY_N_EPISODES = int(EPISODE / 100)
DATA_LENGTH = 250
INIT_CASH = 0
LOAD_MODEL=False
TEST_RUNS = 10 # Run test for N times to smooth out noise

# Logging
SAVED_LOG_PATH = "playground/logs/pg"
RUN_COUNT = str(get_latest_run_count(SAVED_LOG_PATH))

# Reproducibility
SEED = 1992

def main():
    tf.reset_default_graph()
    with tf.Session() as sess:
        env = TradingEnv(DATA_LENGTH, INIT_CASH)
        agent = PG_Agent(env, sess, SEED, LOAD_MODEL)
        for i in tqdm.tqdm(range(EPISODE)):
            agent.isTrain = True
            done = False
            state = agent.env.reset() # To start the process
            agent.one_hot_actions = []
            agent.rewards = []
            agent.states = []
            while done is False:
                action, one_hot_action = agent.act(state)
                state, reward, done, _ = agent.env.step(action, agent)
                agent.rewards.append(reward)
                agent.states.append(state)
                agent.one_hot_actions.append(one_hot_action)
                agent.discount_rewards()

            agent.train_pg_network(i)

            # Summary after one EP, test and summarize the stats
            if i % TEST_EVERY_N_EPISODES == 0 and i >= 0:
                test_trades(agent, i, os.path.join(SAVED_LOG_PATH, RUN_COUNT, 'test_trades'), TEST_RUNS, int(EPISODE / 10))
            summary = sess.run(agent.write_op, feed_dict={agent._inputs: np.vstack(np.array(agent.states)),
                                          agent._actions: np.vstack(np.array(agent.one_hot_actions)),
                                          agent._discounted_rewards: np.vstack(np.array(agent.discounted_rewards)),
                                          agent._mean_reward: np.mean(agent.rewards)
                                        })
            agent.writer.add_summary(summary, i)
            agent.writer.flush()

if __name__ == '__main__':
    main()