import numpy as np, random
from collections import deque
import itertools, os, keras
from playground.utilities.utils import get_latest_run_count
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adagrad
from keras.regularizers import l2
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical

# Hyper Parameters for PG
GAMMA = 0.9 # discount factor for target Q 
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
LEARNING_RATE = 1e-4
TRAINING_EPOCHS = 5
NEURONS_PER_DIM = 32

SAVE_NETWORK_PER_N_EPISODES = 10
SAVED_MODEL_PATH = 'playground/saved_networks/pg'
SAVED_LOG_PATH = "playground/logs/pg"

BUFFER_SIZE = 1000

# Use the log folder to determine the run count to ensure logs and models are in sync
RUN_COUNT = str(get_latest_run_count(SAVED_LOG_PATH))
# Vector Shapes
INPUT_DIM = [1, 4]

# TODO: Too many files
tbCallback = keras.callbacks.TensorBoard(log_dir=os.path.join(SAVED_LOG_PATH, RUN_COUNT),
                                        write_graph=True,
                                        write_images=True)

class PG_Agent():

    def __init__(self, env):
        # init some parameters
        self.epsilon = INITIAL_EPSILON
        self.FINAL_EPSILON = FINAL_EPSILON
        self.env = env
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.state_dim = env.observation_space.shape[1]
        self.action_dim = len(env.action_space)
        self.state_input = np.zeros((1, self.state_dim))
        self.y_input = np.zeros((1, self.action_dim))
        self.isTrain = True
        self.create_pg_network()
    
    def create_pg_network(self):
        model = Sequential()
        model.add(Dense(self.state_dim*NEURONS_PER_DIM, input_shape=(self.state_dim,), activation='relu', bias_initializer='zero', kernel_initializer='glorot_normal'))
        model.add(Dense(self.action_dim, activation='softmax'))
        # model.compile(optimizer=Adagrad(lr=0.01), loss=categorical_crossentropy) # Adam
        model.compile(optimizer='rmsprop', loss=categorical_crossentropy) # rmsprop
        self.network = model
    
    def train_pg_network(self, ep, batch_size=32):
        # replay_samples = self.replay_buffer[random.randint(0, len(self.replay_buffer)-1)]
        replay_samples = random.sample(self.replay_buffer, batch_size)
        state_batch = np.squeeze([data[0] for data in replay_samples])
        y_batch = np.squeeze([data[1] for data in replay_samples])
        self.network.fit(state_batch, y_batch, epochs=TRAINING_EPOCHS, batch_size=batch_size, verbose=0)
        # , callbacks=[tbCallback])
        if ep % SAVE_NETWORK_PER_N_EPISODES == 0 and ep > 0:
            model_dir = os.path.join(SAVED_MODEL_PATH, RUN_COUNT)
            if os.path.isdir(model_dir) is False:
                os.mkdir(model_dir)
            self.network.save(model_dir + '/network-pg-{0}.model'.format(ep))

    def act(self, state):
        y = np.zeros([self.action_dim])
        pred_prob = self.network.predict(state)[0]
        if self.isTrain == True and random.random() < self.epsilon:
            action = random.randint(0, 2)
        else:
            action = np.argmax(pred_prob)
        y[action] = 1
        return y, action # y is an one-hot array, action is the index selected

    def perceive(self, state, action, reward):
        self.replay_buffer.append([state, action, reward])

    def discounted_norm_rewards(self, rewards):
        reward_discounted = np.zeros_like(rewards)
        cumulative = 0
        for index in reversed(range(len(rewards))):
            cumulative = cumulative * GAMMA + rewards[index]
            reward_discounted[index] = cumulative

        mean = np.mean(reward_discounted)
        std = np.std(reward_discounted)
        if std == 0:
            discounted_norm_rewards = [0] # Initially when std = 0
        else:
            discounted_norm_rewards = (reward_discounted - mean) / std
        return discounted_norm_rewards