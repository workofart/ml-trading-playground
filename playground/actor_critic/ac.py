import numpy as np
import tensorflow as tf
from tqdm import tqdm

from playground.utilities.utils import plot_trades, log_scalars, get_latest_run_count, log_histogram
from playground.actor_critic.actor import Actor
from playground.actor_critic.critic import Critic
from playground.env.trading_env import TradingEnv

np.random.seed(2)
tf.set_random_seed(2)  # reproducible


def test(env, actor, ep = 0):
    state = env.reset() # To start the process

    prices = []
    actions = []
    for i in range(DATA_LENGTH):
        prices.append(state[0][2])
        action = actor.choose_action(state)
        actions.append(action)
        state, reward, done, _ = env.step(action)
    plot_trades(ep, prices, actions, env.permitted_trades)

# hyperparameters
OUTPUT_GRAPH = False
EPISODE = 9000
TEST_EVERY_EP = 100
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
DATA_LENGTH = 300   # maximum time step in one episode
LR_A = 1e-6    # learning rate for actor
LR_C = 5e-6     # learning rate for critic

# trading params
INIT_CASH = 100

# Network params
SAVE_NETWORK = 100

env = TradingEnv(data_length=DATA_LENGTH, INIT_CASH=INIT_CASH)

N_F = env.observation_space.shape[1]
N_A = len(env.action_space)

sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

summary_writer = tf.summary.FileWriter('logs/' + str(get_latest_run_count()))
summary_writer.add_graph(sess.graph)

print(get_latest_run_count())
# loading networks
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state('logs/' + str(get_latest_run_count()-2) + '/saved_networks')
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)
else:
    print("Could not find old network weights")
        
for i in tqdm(range(EPISODE)):
    state = env.reset()
    reward_list = []
    action_list = []
    done = False
    while done is False:
        action = actor.choose_action(state)
        action_list.append(action)

        state_next, reward, done, _ = env.step(action)
        reward_list.append(reward)

        td_error = critic.learn(state, reward, state_next)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(state, action, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        state = state_next # increment the current state

        # if done:
            # ep_rs_sum = sum(reward_list)

            # if 'running_reward' not in globals():
            #     running_reward = ep_rs_sum
            # else:
            #     running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            # if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            # print("episode:", i, "  reward:", int(running_reward))
    log_histogram(summary_writer, 'reward_dist', reward_list, i)
    log_scalars(summary_writer, 'avg_reward', np.mean(reward_list), i)
    log_scalars(summary_writer, 'drawdown', np.mean(np.sum(np.array(reward_list) < INIT_CASH, axis=0)), i)
    log_scalars(summary_writer, 'action_errors', np.mean(env.error_count), i)
    
    if i % TEST_EVERY_EP == 0 and i > 0:
        test(env, actor, i)
    
    # save network 9 times per episode
    if i % SAVE_NETWORK == 0 and i > 0:
        saver.save(sess, 'logs/' + str(get_latest_run_count()-1) + '/saved_networks/' + 'network' + '-ac', global_step = i)

