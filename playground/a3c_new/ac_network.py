import tensorflow as tf
import numpy as np

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

DATA_LENGTH = 300
NN1_NEURONS = min(4 * DATA_LENGTH, 512)
NN2_NEURONS = min(2 * DATA_LENGTH, 256)


class AC_Network():
    def __init__(self,state_N,action_N,scope,trainer):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(shape=[None,state_N],dtype=tf.float32)
            
            #Recurrent network for temporal dependencies
            # lstm_cell = tf.contrib.rnn.BasicLSTMCell(256,state_is_tuple=True)
            # c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            # h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            # self.state_init = [c_init, h_init]
            # c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            # h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            # self.state_in = (c_in, h_in)
            # rnn_in = tf.expand_dims(hidden, [0])
            # step_size = tf.shape(self.imageIn)[:1]
            # state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            # lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            #     lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
            #     time_major=False)
            # lstm_c, lstm_h = lstm_state
            # self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            # rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            w_init = tf.random_normal_initializer(0., .1)
            with tf.variable_scope('actor'):
                l1 = tf.layers.dense(self.state, NN1_NEURONS, tf.nn.relu, kernel_initializer=w_init, name='layer_actor1')
                l2 = tf.layers.dense(l1, NN2_NEURONS, tf.nn.relu, kernel_initializer=w_init, name='layer_actor2')
            # a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
            # c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

            
            #Output layers for policy and value estimations
            self.policy = tf.layers.dense(l2,action_N,
                activation=tf.nn.softmax,
                kernel_initializer=normalized_columns_initializer(0.01),
                bias_initializer=None)
            self.value = tf.layers.dense(l2,1,
                activation=None,
                kernel_initializer=normalized_columns_initializer(1.0),
                bias_initializer=None)
            
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,action_N,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))