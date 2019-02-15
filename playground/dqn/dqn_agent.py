import tensorflow as tf
import numpy as np
import random
from collections import deque
# Hyper Parameters for DQN
LEARNING_RATE = 1e-4
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.1 # ending value of epislon
DECAY = 0.99 # epsilon decay
GAMMA = 0.9 # discount factor for q value

# NN Parameters
NN1_NEURONS = 64
NN2_NEURONS = 32

class DQN_Agent():

    def __init__(self, env):
        # init some parameters
        self.epsilon = INITIAL_EPSILON
        self.env = env
        self.isTrain = True
        self.replay_buffer = []
        self.state_dim = env.observation_space.shape[1]
        self.action_dim = len(env.action_space)
        self.learning_rate = LEARNING_RATE

        # State Input tensor
        self.state_input = tf.placeholder("float",[None,self.state_dim], name='state_input')

        # Action Input tensor
        self.action_input = tf.placeholder('float', [None, self.action_dim], name='action_input')
        self.y_input = tf.placeholder('float', [None, 1], name='y_input')

        self.create_dqn_network()
        
        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initializers.global_variables())

        # loading networks
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.session, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
                print("Could not find old network weights")

        # global summary_writer
        # summary_writer = tf.summary.FileWriter('logs',graph=self.session.graph)
        

    def create_dqn_network(self):
        # network weights
        W1 = tf.Variable(tf.random_normal([self.state_dim, NN1_NEURONS]))
        B1 = tf.Variable(tf.zeros([NN1_NEURONS]))
        layer1 = tf.nn.relu(tf.add(tf.matmul(self.state_input, W1), B1))

        W2 = tf.Variable(tf.random_normal([NN1_NEURONS, NN1_NEURONS]))
        B2 = tf.Variable(tf.zeros([NN1_NEURONS]))
        layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, W2), B2))

        W3 = tf.Variable(tf.random_normal([NN1_NEURONS, NN2_NEURONS]))
        B3 = tf.Variable(tf.zeros([NN2_NEURONS]))
        layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, W3), B3))

        W4 = tf.Variable(tf.random_normal([NN2_NEURONS, NN2_NEURONS]))
        B4 = tf.Variable(tf.zeros([NN2_NEURONS]))
        layer4 = tf.nn.relu(tf.add(tf.matmul(layer3, W4), B4))

        W_O = tf.Variable(tf.random_normal([NN2_NEURONS, 1]))
        B5 = tf.Variable(tf.zeros([1]))
        output = tf.add(tf.matmul(layer4, W_O), B5)

        self.Q_value = output

        # Cost
        self.cost = tf.reduce_mean(tf.square(self.y_input - tf.reduce_sum(
            self.Q_value * self.action_input)), reduction_indices=1)

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def train_dqn_network(self, batch_size=32):
        # Assumes "replay_samples" contains [state, action, reward, next_state, done]
        # replay_samples = self.replay_buffer[random.randint(0, len(self.replay_buffer)-1)]
        # random.sample(self.replay_buffer, batch_size)
        rand = random.randint(0, len(self.replay_buffer) - batch_size)
        replay_samples = self.replay_buffer[rand:rand+batch_size]

        state_batch = np.reshape([data[0] for data in replay_samples], (batch_size, 4))
        action_batch = np.reshape([data[1] for data in replay_samples], (batch_size, self.action_dim))
        reward_batch = np.reshape([data[2] for data in replay_samples], (batch_size, 1))
        next_state_batch = np.reshape([data[3] for data in replay_samples], (batch_size, 4))

        # Predict next q-val, given next state
        q_val_batch = self.session.run(self.Q_value, feed_dict={self.state_input:next_state_batch})

        # Create a var to store "advantages/q-vals" derived from rewards
        y_batch = reward_batch + q_val_batch * GAMMA

        # Train on one batch
        _, c = self.session.run([self.optimizer, self.cost],
                            feed_dict={
                                self.y_input: np.reshape(y_batch, (batch_size, 1)),
                                self.action_input: action_batch,
                                self.state_input: state_batch
                            }
        )
        # print("Timestep:", '%04d' % (self.env.time_step+1), "cost={}".format(c))

        # save network every 1000 iteration
        if self.env.time_step % 1000 == 0:
            self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step = self.env.time_step)


    def act(self, state):
        if self.env.time_step > 200 and self.epsilon > FINAL_EPSILON:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 5000

        Q_value = self.Q_value.eval(feed_dict = {
			self.state_input:state
			})[0]
        
        if random.random() <= self.epsilon and self.isTrain is True:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(Q_value)

    def perceive(self, state, action, reward, next_state, done):
        # Assumes "replay_buffer" contains [state, action, reward, next_state, done]
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append([state, one_hot_action, reward, next_state, done])