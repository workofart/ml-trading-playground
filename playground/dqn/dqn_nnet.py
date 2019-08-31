import tensorflow as tf
import numpy as np
import configparser
from playground.utilities.utils import variable_summaries

####### Housing Keeping #######

config_path = 'playground/dqn/config.ini'
config = configparser.ConfigParser()
config.read(config_path)
pCfg = config['dqn_nnet']

# NN Parameters
NN1_NEURONS = int(pCfg['NN1_NEURONS'])
NN2_NEURONS = int(pCfg['NN2_NEURONS'])

#################################

# TODO: refactor this class to create components for layer creation - better stats

class DQN_NNET:

    def __init__(self, state_dim, action_dim, learning_rate, name, seed=0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.name = name
        
        # We use tf.variable_scope here to know which network we're using (DQN or target_net)
        # it will be useful when we will update our w- parameters (by copy the DQN parameters)
        with tf.variable_scope(self.name):
            
            # We create the placeholders
            self.state_input = tf.placeholder(dtype=tf.float64, shape=[None,self.state_dim], name='state_input')
            self.action_input = tf.placeholder(dtype=tf.float64, shape=[None, self.action_dim], name='action_input')
            self.Q_target = tf.placeholder(dtype=tf.float64, shape=[None, 1], name='target')

            self.W1 = tf.get_variable('W1',
                                      dtype=tf.float64,
                                      shape=[self.state_dim, NN1_NEURONS],
                                      trainable=True,
                                      initializer=tf.initializers.random_normal(seed=seed))
            self.B1 = tf.get_variable('B1',
                                dtype=tf.float64,
                                shape=[NN1_NEURONS],
                                trainable=True,
                                initializer=tf.initializers.zeros())
            self.layer1 = tf.nn.relu(tf.add(tf.matmul(self.state_input, self.W1), self.B1))

            self.W2 = tf.get_variable('W2',
                                      dtype=tf.float64,
                                      shape=[NN1_NEURONS, NN2_NEURONS],
                                      trainable=True,
                                      initializer=tf.initializers.random_normal(seed=seed))
            self.B2 = tf.get_variable('B2',
                                dtype=tf.float64,
                                shape=[NN2_NEURONS],
                                trainable=True,
                                initializer=tf.initializers.zeros())
            self.layer2 = tf.nn.relu(tf.add(tf.matmul(self.layer1, self.W2), self.B2))

            self.W_O = tf.get_variable('W_Output',
                                      dtype=tf.float64,
                                      shape=[NN2_NEURONS, self.action_dim],
                                      trainable=True,
                                      initializer=tf.initializers.random_normal(seed=seed)
                                    )
            self.B_O = tf.get_variable('B_Output',
                                dtype=tf.float64,
                                shape=[self.action_dim],
                                trainable=True,
                                initializer=tf.initializers.zeros())
            self.output = tf.add(tf.matmul(self.layer2, self.W_O), self.B_O, name='output')

            self.Q_value = tf.reduce_sum(tf.multiply(self.output, self.action_input), axis=1, name='Q_value')

            # Cost
            self.cost = tf.reduce_mean(tf.square(self.Q_target - self.Q_value), name='cost')

            # Optimizer
            with tf.name_scope("train"):
                self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)

                # Gradient Clipping
                # self.grads = self.optimizer.compute_gradients(self.cost)
                # self.grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.grads if grad is not None]
                # self.optimizer = self.optimizer.apply_gradients(self.grads)

            # Tensorboard Stats
            W1_summary = variable_summaries(self.W1)
            # W2_summary = variable_summaries(self.W2)
            # W_O_summary = variable_summaries(self.W_O)

            # B1_summary = variable_summaries(self.B1)
            # B2_summary = variable_summaries(self.B2)
            # B_O_summary = variable_summaries(self.B_O)

            # layer1_summary = variable_summaries(self.layer1)
            # layer2_summary = variable_summaries(self.layer2)

            output_summary = variable_summaries(self.output)
            Q_value_summary = variable_summaries(self.Q_value)
            cost_summary = variable_summaries(self.cost)

        self.merged_summary = tf.summary.merge(
                       W1_summary +
                    #    W2_summary +
                    #    W_O_summary +
                    #    B1_summary +
                    #    B2_summary +
                    #    B_O_summary +
                    #    layer1_summary +
                    #    layer2_summary +
                       output_summary +
                       Q_value_summary +
                       cost_summary
                       )
    