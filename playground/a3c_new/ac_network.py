import tensorflow as tf
import numpy as np

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

NN1_NEURONS = 64
NN2_NEURONS = 32

class AC_Network():
    def __init__(self,state_N,action_N,scope,trainer):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(shape=[None,state_N],dtype=tf.float32)

            w_init = tf.random_normal_initializer(0., .1)
            l1 = tf.layers.dense(self.state,
                                NN1_NEURONS, 
                                tf.nn.relu, 
                                kernel_initializer=w_init,
                                bias_initializer=tf.constant_initializer(0.1),
                                name='layer1')
            l2 = tf.layers.dense(l1, 
                                NN2_NEURONS, 
                                tf.nn.relu, 
                                kernel_initializer=w_init, 
                                bias_initializer=tf.constant_initializer(0.1),
                                name='layer2')
            
            #Output layers for policy (actor) and value estimations (critic)
            self.policy = tf.layers.dense(l2,
                action_N,
                activation=tf.nn.softmax,
                # kernel_initializer=normalized_columns_initializer(0.01),
                kernel_initializer=w_init,
                bias_initializer=None)
            self.value = tf.layers.dense(l2,
                1,
                activation=None,
                # kernel_initializer=normalized_columns_initializer(1.0),
                kernel_initializer=w_init,
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
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 10e-6))
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