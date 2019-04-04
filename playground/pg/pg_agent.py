import numpy as np, random
from collections import deque
import itertools, os, keras
from playground.utilities.utils import get_latest_run_count, log_scalars
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adagrad
from keras.regularizers import l2
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical

# Hyper Parameters for PG
GAMMA = 0.999999954 # discount factor for target Q 
LEARNING_RATE = 1e-5
NEURONS_PER_DIM = 32

SAVE_NETWORK_PER_N_EPISODES = 100
SAVED_MODEL_PATH = 'playground/saved_networks/pg'
SAVED_LOG_PATH = "playground/logs/pg"

# Use the log folder to determine the run count to ensure logs and models are in sync
RUN_COUNT = str(get_latest_run_count(SAVED_LOG_PATH))

class PG_Agent():

    def __init__(self, env, sess, seed=0, isLoad=False):
        # init some parameters
        self.seed = seed
        self.sess = sess
        self.env = env
        self.state_dim = env.observation_space.shape[1]
        self.action_dim = len(env.action_space)
        self.state_input = np.zeros((1, self.state_dim))
        self.y_input = np.zeros((1, self.action_dim))
        self.isTrain = True
        self.isLoad = isLoad        
        self.create_pg_network()
        sess.run(tf.global_variables_initializer())

        self.one_hot_actions = []
        self.rewards = []
        self.states = []
        self.discounted_rewards = []

        # Logging
        self.summary_writer = tf.summary.FileWriter('playground/logs/pg/' + RUN_COUNT)
        self.summary_writer.add_graph(self.sess.graph)
        tf.summary.scalar('Loss', self.loss)
        tf.summary.scalar('Mean Reward', self._mean_reward)
        self.write_op = tf.summary.merge_all()
    
    def create_pg_network(self):
        # Inspired by: https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Policy%20Gradients/Cartpole/Cartpole%20REINFORCE%20Monte%20Carlo%20Policy%20Gradients.ipynb
        with tf.name_scope('inputs'):
            self._inputs = tf.placeholder(tf.float32, [None, self.state_dim], name='input_state')
            self._actions = tf.placeholder(tf.int32, [None, self.action_dim], name='actions')
            self._discounted_rewards = tf.placeholder(tf.float32, [None, 1], name='discounted_rewards')

            # Add this placeholder for having this variable in tensorboard
            self._mean_reward = tf.placeholder(tf.float32 , name="mean_reward")

        with tf.name_scope('layer1'):
            layer1 = tf.contrib.layers.fully_connected(inputs=self._inputs,
                                                        num_outputs=NEURONS_PER_DIM,
                                                        activation_fn=tf.nn.elu,
                                                        weights_initializer=tf.initializers.he_normal(seed=self.seed),
                                                        biases_initializer=tf.constant_initializer(0.1))

        with tf.name_scope('layer2'):
            layer2 = tf.contrib.layers.fully_connected(inputs=layer1,
                                                        num_outputs=int(NEURONS_PER_DIM / 2),
                                                        activation_fn=tf.nn.elu,
                                                        weights_initializer=tf.initializers.he_normal(seed=self.seed),
                                                        biases_initializer=tf.constant_initializer(0.1))

        with tf.name_scope('layer3'):
            layer3 = tf.contrib.layers.fully_connected(inputs=layer2,
                                                        num_outputs=self.action_dim,
                                                        activation_fn=tf.nn.elu,
                                                        weights_initializer=tf.initializers.he_normal(seed=self.seed),
                                                        biases_initializer=tf.constant_initializer(0.1))

        

        with tf.name_scope('softmax'):
            self.action_output = tf.nn.softmax(layer3)

        with tf.name_scope('loss'):
            # self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer3, labels=self._actions)

            # The minus sign is to accommodate tensorflow only supporting minimize, not maximize
            self.neg_log_prob = tf.reduce_sum(-tf.log(self.action_output)*tf.cast(self._actions, tf.float32), axis=1)

            self.loss = tf.reduce_mean(self.neg_log_prob * self._discounted_rewards)
        
        with tf.name_scope('train'):
            self.train_opt = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(self.loss)

        self.saver = tf.train.Saver()
        model_dir = os.path.join(SAVED_MODEL_PATH, str(int(RUN_COUNT) - 1))
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        if (self.isTrain is False and checkpoint and checkpoint.model_checkpoint_path) or self.isLoad is True:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")            
    
    def train_pg_network(self, ep):
        loss, neg_prob, _ = self.sess.run([self.loss, self.neg_log_prob, self.train_opt], feed_dict={self._inputs: np.vstack(np.array(self.states)),
                                                    self._actions: np.vstack(np.array(self.one_hot_actions)),
                                                    self._discounted_rewards: np.vstack(np.array(self.discounted_rewards)),
                                                    })
        if ep % SAVE_NETWORK_PER_N_EPISODES == 0 and ep > 0:
            model_dir = os.path.join(SAVED_MODEL_PATH, RUN_COUNT)
            self.saver.save(self.sess, model_dir + '/network-pg-{0}.ckpt'.format(ep))

    def act(self, state):
        y = np.zeros([self.action_dim])
        pred_prob = self.sess.run(self.action_output, feed_dict={self._inputs:state})
        # Stochastic Actions, we don't take argmax
        action = np.random.choice(range(pred_prob.shape[1]), p=pred_prob.ravel())
        y[action] = 1
        return action, y # y is an one-hot array, action is the index selected

    def discount_rewards(self):
        reward_discounted = np.zeros_like(self.rewards)
        cumulative = 0
        for index in reversed(range(len(self.rewards))):
            cumulative = cumulative * GAMMA + self.rewards[index]
            reward_discounted[index] = cumulative

        # Normalization
        mean = np.mean(reward_discounted)
        std = np.std(reward_discounted)
        discounted_norm_rewards = (reward_discounted - mean) / (std + 1e-10) # don't divide by zero
       
        self.discounted_rewards = discounted_norm_rewards