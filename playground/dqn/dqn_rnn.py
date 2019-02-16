import tensorflow as tf

class DQN_RNN():
    def __init__(self, state_dim, num_time_steps, learning_rate, name):       
        self.state_dim = state_dim
        # self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.name = name

        # State Input tensor
        self.state_input = tf.placeholder("float",[None, num_time_steps, state_dim], name='state_input')

        # Output tensor
        self.y_input = tf.placeholder('float', [None, num_time_steps, 1], name='target')

        # self.W1 = tf.Variable(tf.random_normal([self.state_dim, 16]))
        # self.B1 = tf.Variable(tf.zeros([16]))
        # self.layer1 = tf.nn.relu(tf.add(tf.matmul(self.state_input, self.W1), self.B1))

        # self.W2 = tf.Variable(tf.random_normal([16, 10]))
        # self.B2 = tf.Variable(tf.zeros([10]))
        # self.layer2 = tf.nn.relu(tf.add(tf.matmul(self.layer1, self.W2), self.B2))

        # self.W_O = tf.Variable(tf.random_normal([10, 1]))
        # self.B3 = tf.Variable(tf.zeros([1]))
        # self.output = tf.add(tf.matmul(self.layer2, self.W_O), self.B3)

        # RNN specific cells
        #ANY RNN CELL TYPE
        cell = tf.contrib.rnn.OutputProjectionWrapper(
            tf.contrib.rnn.GRUCell(
                num_units=10,
                activation=tf.nn.relu
                ),
            output_size=1)

        self.outputs, states = tf.nn.dynamic_rnn(cell, self.state_input, dtype=tf.float32)

        #MSE
        self.cost=tf.reduce_mean(tf.square(self.outputs-self.y_input))
        self.optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        

        # Cost
        # self.cost = tf.reduce_mean(tf.square(self.y_input - tf.reduce_sum(
        #     self.output * self.action_input)), reduction_indices=1)

        # Optimizer
        # self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)