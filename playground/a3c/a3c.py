import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt

from playground.env.trading_env import TradingEnv
from playground.utilities.utils import plot_trades, log_scalars, get_latest_run_count, log_histogram


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

N_WORKERS = multiprocessing.cpu_count()
MAX_GLOBAL_EP = 1000
TEST_EVERY_EP = 100
SAVE_NETWORK = 100
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 1e-6    # learning rate for actor
LR_C = 1e-5    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

DATA_LENGTH = 300
INIT_CASH = 100

NN1_NEURONS = 4 * DATA_LENGTH
NN2_NEURONS = 2 * DATA_LENGTH

# only to get the following stats and testing
# each worker creates its own env
dummy_env = TradingEnv(DATA_LENGTH, INIT_CASH)
N_S = dummy_env.observation_space.shape[1]
N_A = len(dummy_env.action_space)


class ActorCriticNet(object):
    def __init__(self, scope, globalActorCritc=None):

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.state = tf.placeholder(tf.float32, [None, N_S], 'State')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.state = tf.placeholder(tf.float32, [None, N_S], 'State')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'Action')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.action_prob, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.action_prob + 1e-5) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.action_prob * tf.log(self.action_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalActorCritc.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalActorCritc.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalActorCritc.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalActorCritc.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a1 = tf.layers.dense(self.state, NN1_NEURONS, tf.nn.relu, kernel_initializer=w_init, name='layer_actor1')
            l_a2 = tf.layers.dense(l_a1, NN2_NEURONS, tf.nn.relu, kernel_initializer=w_init, name='layer_actor2')
            a_prob = tf.layers.dense(l_a2, N_A, tf.nn.softmax, kernel_initializer=w_init, name='action_prob')
        with tf.variable_scope('critic'):
            l_c1 = tf.layers.dense(self.state, NN1_NEURONS, tf.nn.relu, kernel_initializer=w_init, name='layer_critic1')
            l_c2 = tf.layers.dense(l_c1, NN2_NEURONS, tf.nn.relu, kernel_initializer=w_init, name='layer_critic2')
            v = tf.layers.dense(l_c2, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, state):  # run by a local
        # prob_weights = SESS.run(self.action_prob, feed_dict={self.s: s[np.newaxis, :]})
        prob_weights = SESS.run(self.action_prob, feed_dict={self.state: state})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action


class Worker(object):
    def __init__(self, name, globalActorCritc, SESS):
        self.env = TradingEnv(DATA_LENGTH, INIT_CASH)
        self.name = name
        self.AC = ActorCriticNet(name, globalActorCritc)
        self.session = SESS

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            print(self.name,"Ep:", GLOBAL_EP)
            state = self.env.reset()
            reward_list = []
            done = False
            while done is False:
                action = self.AC.choose_action(state)
                next_state, reward, done, info = self.env.step(action)
                reward_list.append(reward)
                buffer_s.append(state)
                buffer_a.append(action)
                buffer_r.append(reward)

                if total_step % UPDATE_GLOBAL_ITER == 0:   # update global and assign to local net
                    v_s_ = SESS.run(self.AC.v, {self.AC.state: next_state})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.state: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                state = next_state
                total_step += 1
                ### One episode End
            
            GLOBAL_EP += 1
            v_s_ = 0   # terminal
            log_scalars(summary_writer, 'profit', info['marketValue'] - INIT_CASH, GLOBAL_EP)
            log_scalars(summary_writer, 'avg_reward', np.mean(reward_list), GLOBAL_EP)
            log_histogram(summary_writer, 'reward_dist', reward_list, GLOBAL_EP)
            log_scalars(summary_writer, 'drawdown', np.mean(np.sum(np.array(reward_list) < INIT_CASH, axis=0)), i)

            # TODO: Might not be able to naively create test trades, because
            # the workers are working asynchronously, so the trades
            # might not make sense
            if GLOBAL_EP % TEST_EVERY_EP == 0 and GLOBAL_EP > 0:
                test(dummy_env, self.AC, GLOBAL_EP)
            
            # save network frequently
            if GLOBAL_EP % SAVE_NETWORK == 0 and GLOBAL_EP > 0:
                saver.save(self.session, 'logs/' + str(get_latest_run_count()-1) + '/saved_networks/' + 'network' + '-a3c', global_step = GLOBAL_EP)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--device', required=True, help='either "CPU" or "GPU" for training')
    args = vars(ap.parse_args())
    SESS = tf.Session()

    with tf.device("/{}:0".format(args['device'])):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_ACTORCRITIC = ActorCriticNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_ACTORCRITIC, SESS))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    # loading networks
    saver = tf.train.Saver()

    # Setup Tensorboard
    summary_writer = tf.summary.FileWriter('logs/' + str(get_latest_run_count()))
    summary_writer.add_graph(SESS.graph)

    worker_threads = []
    for worker in workers:
        checkpoint = tf.train.get_checkpoint_state('logs/' + str(get_latest_run_count()-2) + '/saved_networks_' + worker.name)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(SESS, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)