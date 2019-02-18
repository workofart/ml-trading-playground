import tensorflow as tf
import numpy as np
import random
from collections import deque
from playground.dqn.dqn_nnet import DQN_NNET

# Hyper Parameters for DQN
LEARNING_RATE = 1e-4
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.1 # ending value of epislon
DECAY = 0.99 # epsilon decay
GAMMA = 0.9 # discount factor for q value
UPDATE_TARGET_FREQ = 200 # how many timesteps to update target network params

# NN Parameters
NN1_NEURONS = 64
NN2_NEURONS = 32

# TODO: Use Experience Buffer for sampling and adding experience

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
        self.update_target_net_freq = UPDATE_TARGET_FREQ

        # Reset the graph
        tf.reset_default_graph()
        self.network = DQN_NNET(self.state_dim, self.action_dim, self.learning_rate, 'q_network')
        self.target_network = DQN_NNET(self.state_dim, self.action_dim, self.learning_rate, 'target_q_network')
        
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
        
    def update_target_q_net_if_needed(self, step):
        if step % self.update_target_net_freq == 0 and step > 0:
            # Get the parameters of our DQNNetwork
            from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "q_network")
            
            # Get the parameters of our Target_network
            to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "target_q_network")

            op_holder = []
            
            # Update our target_q_network parameters with q_network parameters
            for from_var,to_var in zip(from_vars,to_vars):
                op_holder.append(to_var.assign(from_var))
            self.session.run(op_holder)
            # print('Timesteps:{} | Target Q-network has been updated.'.format(self.env.time_step))

    def train_dqn_network(self, batch_size=32):
        self.update_target_q_net_if_needed(self.env.time_step)
        # Assumes "replay_samples" contains [state, action, reward, next_state, done]
        rand = random.randint(0, len(self.replay_buffer) - batch_size)
        replay_samples = self.replay_buffer[rand:rand+batch_size]

        state_batch = np.reshape([data[0] for data in replay_samples], (batch_size, 4))
        action_batch = np.reshape([data[1] for data in replay_samples], (batch_size, self.action_dim))
        reward_batch = np.reshape([data[2] for data in replay_samples], (batch_size, 1))
        next_state_batch = np.reshape([data[3] for data in replay_samples], (batch_size, 4))

        # Get the Target Q-value for the next state using the target network,
        # by making a second forward-prop
        target_q_val_batch = self.session.run(self.target_network.output, feed_dict={self.target_network.state_input:next_state_batch})

        # Get Q values for next state using the q-network
        q_val_batch = self.session.run(self.network.output, feed_dict={self.network.state_input:next_state_batch})
        
        # Target Q-value - "advantages/q-vals" derived from rewards
        y_batch = []
        for i in range(0, batch_size):
            # Use Q-network to select the best action for next state
            action = np.argmax(q_val_batch[i])

            done = replay_samples[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * target_q_val_batch[i][action])

        # Train on one batch on the Q-network
        _, c = self.session.run([self.network.optimizer, self.network.cost],
                            feed_dict={
                                self.network.Q_input: np.reshape(y_batch, (batch_size, 1)),
                                self.network.action_input: action_batch,
                                self.network.state_input: state_batch
                            }
        )
        if self.env.time_step % 3000 == 0: print("Timestep:", '%04d' % (self.env.time_step+1), "cost={}".format(c))

        # save network every 1000 iteration
        if self.env.time_step % 1000 == 0:
            self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step = self.env.time_step)


    def act(self, state):
        if self.env.time_step > 200 and self.epsilon > FINAL_EPSILON:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 5000

        output = self.network.output.eval(feed_dict = {
			self.network.state_input:state
			})[0]
        
        if random.random() <= self.epsilon and self.isTrain is True:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(output)

    def perceive(self, state, action, reward, next_state, done):
        # Assumes "replay_buffer" contains [state, action, reward, next_state, done]
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append([state, one_hot_action, reward, next_state, done])