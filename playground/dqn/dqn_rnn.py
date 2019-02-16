import tensorflow as tf
import numpy as np

class DQN_RNN():
    def __init__(self, state_dim, num_time_steps, learning_rate, name):       
        self.state_dim = state_dim
        # self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.name = name

        # State Input tensor
        self.state_input = tf.placeholder("float",[None, num_time_steps, state_dim], name='state_input')

        # Output tensor
        self.y_input = tf.placeholder('float', [None, 1], name='target')

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
        
        ##############################
        num_layers = 3
        self.keep_prob = 0.8
        self.lstm_size = 10
        self.learning_rate = tf.placeholder(tf.float32, None)

        cell = tf.contrib.rnn.MultiRNNCell(
            [self._create_one_cell() for _ in range(num_layers)], 
            state_is_tuple=True
        ) if num_layers > 1 else self._create_one_cell()

        val, _ = tf.nn.dynamic_rnn(cell, self.state_input, dtype=tf.float32)

        # Before transpose, val.get_shape() = (batch_size, num_steps, lstm_size)
        # After transpose, val.get_shape() = (num_steps, batch_size, lstm_size)
        val = tf.transpose(val, [1, 0, 2])
        # last.get_shape() = (batch_size, lstm_size)
        last = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_lstm_output")

        weight = tf.Variable(tf.truncated_normal([self.lstm_size, state_dim]))
        bias = tf.Variable(tf.constant(0.1, shape=[state_dim]))
        self.prediction = tf.matmul(last, weight) + bias

        self.loss = tf.reduce_mean(tf.square(self.prediction - self.y_input))
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        self.minimize = optimizer.minimize(self.loss)


        # cell = tf.contrib.rnn.OutputProjectionWrapper(tf.nn.rnn_cell.BasicRNNCell(10), output_size=1)
        # self.outputs, self.states = tf.nn.dynamic_rnn(cell, self.state_input,dtype=tf.float32)

        # #MSE
        # self.cost=tf.reduce_mean(tf.square(self.outputs-self.y_input))
        # self.optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        

        # Cost
        # self.cost = tf.reduce_mean(tf.square(self.y_input - tf.reduce_sum(
        #     self.output * self.action_input)), reduction_indices=1)

        # Optimizer
        # self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def _create_one_cell(self):
        lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_size, state_is_tuple=True)
        return lstm_cell
        if self.keep_prob < 1.0:
            return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
