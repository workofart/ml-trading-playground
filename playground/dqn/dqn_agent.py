import tensorflow as tf
import numpy as np
import random, os, configparser
from collections import deque
from playground.dqn.dqn_nnet import DQN_NNET
from playground.dqn.experience_buffer import Experience_Buffer
from playground.utilities.utils import variable_summaries, get_latest_run_count, update_target_graph, log_scalars

####### Housing Keeping #######

config_path = 'playground/dqn/config.ini'
config = configparser.ConfigParser()
config.read(config_path)
cfg = config['dqn_agent']
paths_cfg = config['paths']
logistics_cfg = config['logistics']

LEARNING_RATE = float(cfg['LEARNING_RATE'])
INITIAL_EPSILON = float(cfg['INITIAL_EPSILON'])  # starting value of epsilon
FINAL_EPSILON = float(cfg['FINAL_EPSILON']) # ending value of epislon
GAMMA = float(cfg['GAMMA']) # discount factor for q value
UPDATE_TARGET_GRAPH_FREQ = int(logistics_cfg['UPDATE_TARGET_GRAPH_FREQ'])  # how many timesteps to update target network params
SAVED_MODEL_PATH = paths_cfg['SAVED_MODEL_PATH']
SAVED_LOG_PATH = paths_cfg['SAVED_LOG_PATH']
RUN_COUNT = str(get_latest_run_count(SAVED_LOG_PATH))
TENSORBOARD_LOG_PATH = os.path.join(SAVED_LOG_PATH, RUN_COUNT)

#################################

class DQN_Agent():

    def __init__(self, env, eps, isTrain = True, isLoad = False, seed=0):
        # init some parameters
        self.env = env
        self.epsilon = INITIAL_EPSILON
        self.final_epsilon = FINAL_EPSILON
        self.epsilon_decay = 1 / (env.data_length * eps)
        self.learning_rate = LEARNING_RATE
        self.total_episodes = eps
        self.isTrain = isTrain
        self.seed = seed
        
        self.replay_buffer = Experience_Buffer()
        self.state_dim = env.observation_space.shape[1]
        self.action_dim = len(env.action_space)
        
        # Counters
        self.current_episode = 0
        self.accum_steps = 0
        
        # Reset the graph
        tf.reset_default_graph()
        self.network = DQN_NNET(self.state_dim, self.action_dim, self.learning_rate, 'q_network', seed=seed)
        self.target_network = DQN_NNET(self.state_dim, self.action_dim, self.learning_rate, 'target_q_network', seed=seed)

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initializers.global_variables())

        # Update target graph with main graph
        self.update_q_network_op_holder = update_target_graph("q_network", "target_q_network")
        self.session.run(self.update_q_network_op_holder)

        # Tensorboard
        self.summary_writer = tf.summary.FileWriter(TENSORBOARD_LOG_PATH)
        self.summary_writer.add_graph(self.session.graph)

        # loading networks
        self.saver = tf.train.Saver()
        model_dir = os.path.join(SAVED_MODEL_PATH, str(int(RUN_COUNT) - 1))
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        if (self.isTrain is False or isLoad is True) and checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        
    def update_target_q_net_if_needed(self):
        if self.accum_steps % UPDATE_TARGET_GRAPH_FREQ == 0 and self.accum_steps > 0:
            self.session.run(self.update_q_network_op_holder)
            # print('Timesteps:{} | Target Q-network has been updated.'.format(self.accum_steps))

    def train_dqn_network(self, batch_size=32):
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
                # y_batch.append(max(-1, min(reward_batch[i] + GAMMA * target_q_val_batch[i][action], 1)))
                y_batch.append(reward_batch[i] + GAMMA * target_q_val_batch[i][action])

        # Train on one batch on the Q-network
        _, c, summary = self.session.run([self.network.optimizer, self.network.cost, self.network.merged_summary],
                            feed_dict={
                                self.network.Q_target: np.reshape(y_batch, (batch_size, 1)),
                                self.network.action_input: action_batch,
                                self.network.state_input: state_batch
                            }
                        )

        self.summary_writer.add_summary(summary, self.current_episode)

        # save network 9 times per episode
        if self.current_episode % (self.total_episodes / 10) == 0:
            model_dir = os.path.join(SAVED_MODEL_PATH, RUN_COUNT)
            self.saver.save(self.session, model_dir + '/network-dqn', global_step = self.current_episode)

        return c


    def act(self, state):
        if self.isTrain is True:
            self.epsilon = self.final_epsilon + (1 - self.final_epsilon) * np.exp(-self.epsilon_decay * self.accum_steps)
        if self.isTrain is True and random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            output = self.network.output.eval(feed_dict = {
			    self.network.state_input:state
			})[0]
            if self.current_episode % 100 == 0:
                log_scalars(self.summary_writer, 'output_buy', output[0], self.accum_steps)
                log_scalars(self.summary_writer, 'output_sell', output[1], self.accum_steps)
                log_scalars(self.summary_writer, 'output_hold', output[2], self.accum_steps)
            action = np.argmax(output)
            return action

    def perceive(self, state, action, reward, next_state, done):
        # Assumes "replay_buffer" contains [state, action, reward, next_state, done]
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.add([state, one_hot_action, reward, next_state, done])