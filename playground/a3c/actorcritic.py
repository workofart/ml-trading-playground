import tensorflow as tf
import numpy as np
from playground.env.trading_env import TradingEnv
from playground.utilities.utils import variable_summaries

GLOBAL_NET_SCOPE = 'Global_Net'
ENTROPY_BETA = 0.001
DATA_LENGTH = 600
INIT_CASH = 100
NN1_NEURONS = min(4 * DATA_LENGTH, 512)
NN2_NEURONS = min(2 * DATA_LENGTH, 256)


# only to get the following stats and testing
# each worker creates its own env
dummy_env = TradingEnv(DATA_LENGTH, INIT_CASH)
N_S = dummy_env.observation_space.shape[1]
N_A = len(dummy_env.action_space)

LR_A = 1e-8    # learning rate for actor
LR_C = 3e-8    # learning rate for critic

class ActorCriticNet(object):
    def __init__(self, scope, sess, globalActorCritic=None):
        self.session = sess

        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.state = tf.placeholder(tf.float32, [None, N_S], 'State')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.state = tf.placeholder(tf.float32, [None, N_S], 'State')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'Action')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                # self.action_prob = self.policy
                self.action_prob, self.v, self.a_params, self.c_params = self._build_net(scope)
                # self.action_prob_summary = variable_summaries(self.action_prob)
                # self.v_summary = variable_summaries(self.v)
                # self.a_params_summary = variable_summaries(self.a_params)
                # self.c_params_summary = variable_summaries(self.c_params)
                
                # Advantage
                td = tf.subtract(self.v_target, self.v, name='TD_error')
                # self.td_summary = variable_summaries(td)

                # self.merged_summary = tf.summary.merge(
                #     self.action_prob_summary +
                #     self.v_summary +
                #     # self.a_params_summary +
                #     # self.c_params_summary +
                #     self.td_summary
                #     )

                with tf.name_scope('c_loss'):
                    # Critic's Value loss
                    self.c_loss = tf.reduce_mean(tf.square(td))
                    # self.c_loss_summary = variable_summaries(self.c_loss)
                    # self.merged_summary = tf.summary.merge(
                    #     self.c_loss_summary
                    # )

                with tf.name_scope('a_loss'):
                    # tf.log(self.responsible_outputs)
                    log_prob = tf.log(tf.reduce_sum(tf.log(self.action_prob) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True))

                    # Actor's Policy loss
                    exp_v = tf.reduce_sum(log_prob * tf.stop_gradient(td))
                    entropy = tf.reduce_sum(self.action_prob * tf.log(self.action_prob),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.a_loss = self.c_loss + exp_v - ENTROPY_BETA * entropy
                    
                    # self.a_loss_summary = variable_summaries(self.a_loss)
                    # self.merged_summary = tf.summary.merge(
                    #     self.a_loss_summary
                    # )
                    

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)
                    # self.a_grads_summary = variable_summaries(self.a_grads)
                    # self.c_grads_summary = variable_summaries(self.c_grads)
                    # self.merged_summary = tf.summary.merge(
                    #     self.a_grads_summary +
                    #     self.c_grads_summary
                    # )
        
            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalActorCritic.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalActorCritic.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalActorCritic.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalActorCritic.c_params))
        
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
        self.session.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.session.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, state):  # run by a local
        # prob_weights = self.session.run(self.action_prob, feed_dict={self.s: s[np.newaxis, :]})
        prob_weights = self.session.run(self.action_prob, feed_dict={self.state: state})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action
