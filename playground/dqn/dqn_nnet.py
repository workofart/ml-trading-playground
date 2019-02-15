import tensorflow as tf


# NN Parameters
NN1_NEURONS = 64
NN2_NEURONS = 32


class DQN_NNET:

    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name
        
        
        # We use tf.variable_scope here to know which network we're using (DQN or target_net)
        # it will be useful when we will update our w- parameters (by copy the DQN parameters)
        with tf.variable_scope(self.name):
            
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 100, 120, 4]

            # State Input tensor
            self.state_input = tf.placeholder("float",[None,self.state_dim], name='state_input')

            # Action Input tensor
            self.action_input = tf.placeholder('float', [None, self.action_dim], name='action_input')
            self.target_Q = tf.placeholder('float', [None, 1], name='target')

            self.W1 = tf.Variable(tf.random_normal([self.state_dim, NN1_NEURONS]))
            self.B1 = tf.Variable(tf.zeros([NN1_NEURONS]))
            self.layer1 = tf.nn.relu(tf.add(tf.matmul(self.state_input, self.W1), self.B1))

            self.W2 = tf.Variable(tf.random_normal([NN1_NEURONS, NN2_NEURONS]))
            self.B2 = tf.Variable(tf.zeros([NN2_NEURONS]))
            self.layer2 = tf.nn.relu(tf.add(tf.matmul(self.layer1, self.W2), self.B2))

            self.W_O = tf.Variable(tf.random_normal([NN2_NEURONS, 1]))
            self.B3 = tf.Variable(tf.zeros([1]))
            self.output = tf.add(tf.matmul(self.layer2, self.W_O), self.B3)

            # Cost
            self.cost = tf.reduce_mean(tf.square(self.y_input - tf.reduce_sum(
                self.output * self.action_input)), reduction_indices=1)

            # Optimizer
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)