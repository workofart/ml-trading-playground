import numpy as np
import tensorflow as tf
from tqdm import tqdm

from playground.utilities.utils import plot_trades
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
EPISODE = 300
TEST_EVERY_EP = 10
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
DATA_LENGTH = 300   # maximum time step in one episode
LR_A = 1e-6    # learning rate for actor
LR_C = 5e-6     # learning rate for critic

# trading params
INIT_CASH = 100

env = TradingEnv(data_length=DATA_LENGTH, INIT_CASH=INIT_CASH)

N_F = env.observation_space.shape[1]
N_A = len(env.action_space)

sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

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

        if done:
            ep_rs_sum = sum(reward_list)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i, "  reward:", int(running_reward))
    
    if i % TEST_EVERY_EP == 0 and i > 0:
            test(env, actor, i)
