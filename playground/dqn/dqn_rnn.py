import tensorflow as tf

class DQN_RNN():
    def __init__(self, state_dim, action_dim, rnn_cell, learning_rate, name):
        # State Input tensor
        self.state_input = tf.placeholder("float",[None,self.state_dim], name='state_input')

        # Output tensor
        self.y_input = tf.placeholder('float', [None, 1], name='target')
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.name = name

        self.W1 = tf.Variable(tf.random_normal([self.state_dim, 16]))
        self.B1 = tf.Variable(tf.zeros([16]))
        self.layer1 = tf.nn.relu(tf.add(tf.matmul(self.state_input, self.W1), self.B1))

        self.W2 = tf.Variable(tf.random_normal([16, 10]))
        self.B2 = tf.Variable(tf.zeros([10]))
        self.layer2 = tf.nn.relu(tf.add(tf.matmul(self.layer1, self.W2), self.B2))

        self.W_O = tf.Variable(tf.random_normal([10, 1]))
        self.B3 = tf.Variable(tf.zeros([1]))
        self.output = tf.add(tf.matmul(self.layer2, self.W_O), self.B3)

        # RNN specific cells
        self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn,self.rnn_state = tf.nn.dynamic_rnn(\
                inputs=self.convFlat,cell=rnn_cell,dtype=tf.float32,initial_state=self.state_in,scope=myScope+'_rnn')
        self.rnn = tf.reshape(self.rnn,shape=[-1,h_size])

        # Cost
        self.cost = tf.reduce_mean(tf.square(self.y_input - tf.reduce_sum(
            self.output * self.action_input)), reduction_indices=1)

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)