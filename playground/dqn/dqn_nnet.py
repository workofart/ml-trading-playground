import tensorflow as tf


# NN Parameters
NN1_NEURONS = 64
NN2_NEURONS = 32

class DQN_NNET:

    def __init__(self, state_dim, action_dim, learning_rate, name):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.name = name
        
        
        # We use tf.variable_scope here to know which network we're using (DQN or target_net)
        # it will be useful when we will update our w- parameters (by copy the DQN parameters)
        with tf.variable_scope(self.name):
            
            # We create the placeholders
            self.state_input = tf.placeholder(tf.float64,[None,self.state_dim], name='state_input')
            self.action_input = tf.placeholder(tf.float64, [None, self.action_dim], name='action_input')
            self.Q_input = tf.placeholder(tf.float64, [None, 1], name='target')

            self.W1 = tf.Variable(dtype=tf.float64, initial_value=tf.random_normal([self.state_dim, NN1_NEURONS], 0.0, 0.3, dtype=tf.float64))
            self.B1 = tf.Variable(dtype=tf.float64, initial_value=tf.zeros([NN1_NEURONS], tf.float64))
            self.layer1 = tf.nn.relu(tf.add(tf.matmul(self.state_input, self.W1), self.B1))

            self.W2 = tf.Variable(dtype=tf.float64, initial_value=tf.random_normal([NN1_NEURONS, NN2_NEURONS], 0.0, 0.3, dtype=tf.float64))
            self.B2 = tf.Variable(dtype=tf.float64, initial_value=tf.zeros([NN2_NEURONS], tf.float64))
            self.layer2 = tf.nn.relu(tf.add(tf.matmul(self.layer1, self.W2), self.B2))

            self.W_O = tf.Variable(dtype=tf.float64, initial_value=tf.random_normal([NN2_NEURONS, 1], 0.0, 0.3, dtype=tf.float64))
            self.B3 = tf.Variable(dtype=tf.float64, initial_value=tf.zeros([self.action_dim], tf.float64))
            self.output = tf.add(tf.matmul(self.layer2, self.W_O), self.B3)

            self.Q_value = tf.reduce_sum(tf.multiply(self.output, self.action_input), axis=1)

            # Cost
            # self.cost = tf.reduce_mean(tf.square(self.Q_input - tf.reduce_sum(
                # self.output * self.action_input)), reduction_indices=1)
            self.cost = tf.reduce_mean(tf.square(self.Q_input - self.Q_value))

            # Optimizer
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
    