import tensorflow as tf
import numpy as np
import random, os
from collections import deque
from playground.dqn.dqn_nnet import DQN_NNET
from playground.dqn.experience_buffer import Experience_Buffer
from playground.utilities.utils import variable_summaries, get_latest_run_count, update_target_graph

# Hyper Parameters for DQN
LEARNING_RATE = 1e-5
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.05 # ending value of epislon
GAMMA = 0.99 # discount factor for q value

SAVED_MODEL_PATH = "playground/saved_networks/dqn"
SAVED_LOG_PATH = "playground/logs/dqn/"
RUN_COUNT = str(get_latest_run_count(SAVED_LOG_PATH))

class DQN_Agent():

    def __init__(self, env, eps, isTrain = True, isLoad = False, seed=0):
        # init some parameters
        self.epsilon = INITIAL_EPSILON
        self.final_epsilon = FINAL_EPSILON
        self.env = env
        self.isTrain = isTrain
        self.seed = seed
        self.replay_buffer = Experience_Buffer()
        self.state_dim = env.observation_space.shape[1]
        self.action_dim = len(env.action_space)
        self.learning_rate = LEARNING_RATE
        self.total_episodes = eps
        # self.update_target_net_freq = max(5, int(self.total_episodes / 100)) # how many episodes to update target network params
        self.update_target_net_freq = 3 # how many episodes to update target network params
        self.is_updated_target_net = False

        # Reset the graph
        tf.reset_default_graph()
        self.network = DQN_NNET(self.state_dim, self.action_dim, self.learning_rate, 'q_network', seed=seed)
        self.target_network = DQN_NNET(self.state_dim, self.action_dim, self.learning_rate, 'target_q_network', seed=seed)

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initializers.global_variables())

        # Tensorboard
        self.summary_writer = tf.summary.FileWriter(os.path.join(SAVED_LOG_PATH,RUN_COUNT))
        self.summary_writer.add_graph(self.session.graph)

        # loading networks
        self.saver = tf.train.Saver()
        model_dir = os.path.join(SAVED_MODEL_PATH, str(int(RUN_COUNT) - 1))
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        if (self.isTrain is False and checkpoint and checkpoint.model_checkpoint_path) or isLoad is True:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        
    def update_target_q_net_if_needed(self, ep):
        if ep % self.update_target_net_freq == 0 and ep > 0 and self.is_updated_target_net is False:
            op_holder = update_target_graph("q_network", "target_q_network")
            self.session.run(op_holder)

            self.is_updated_target_net = True
            # print('Timesteps:{} | Target Q-network has been updated.'.format(self.env.time_step))

    def train_dqn_network(self, ep, batch_size=32):
        self.update_target_q_net_if_needed(ep)
        # Assumes "replay_samples" contains [state, action, reward, next_state, done]
        replay_samples = self.replay_buffer.sample(batch_size)

        state_batch = np.reshape([data[0] for data in replay_samples], (batch_size, self.state_dim))
        action_batch = np.reshape([data[1] for data in replay_samples], (batch_size, self.action_dim))
        reward_batch = np.reshape([data[2] for data in replay_samples], (batch_size, 1))
        next_state_batch = np.reshape([data[3] for data in replay_samples], (batch_size, self.state_dim))

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
                y_batch.append(max(-1, min(reward_batch[i] + GAMMA * target_q_val_batch[i][action], 1)))

        # Train on one batch on the Q-network
        _, c, summary = self.session.run([self.network.optimizer, self.network.cost, self.network.merged_summary],
                            feed_dict={
                                self.network.Q_input: np.reshape(y_batch, (batch_size, 1)),
                                self.network.action_input: action_batch,
                                self.network.state_input: state_batch
                            }
        )
        self.summary_writer.add_summary(summary, ep)

        # save network 9 times per episode
        if ep % (self.total_episodes / 10) == 0:
            model_dir = os.path.join(SAVED_MODEL_PATH, RUN_COUNT)
            self.saver.save(self.session, model_dir + '/network-dqn', global_step = ep)

        return c


    def act(self, state):
        if self.isTrain is True and random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            output = self.network.output.eval(feed_dict = {
			    self.network.state_input:state
			})[0]
            action = np.argmax(output)
            return action

    def perceive(self, state, action, reward, next_state, done):
        # Assumes "replay_buffer" contains [state, action, reward, next_state, done]
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.add([state, one_hot_action, reward, next_state, done])