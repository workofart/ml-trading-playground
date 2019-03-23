import numpy as np, random
from collections import deque
import itertools, os, keras
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

# Vector Shapes
INPUT_DIM = [1, 4]

# TODO: Tensorboard keras integration
tbCallback = keras.callbacks.TensorBoard(log_dir=SAVED_LOG_PATH, write_graph=True, write_images=True)

class PG_Agent():

    def __init__(self, env):
        # init some parameters
        self.epsilon = INITIAL_EPSILON
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
        self.network.fit(state_batch, y_batch, epochs=TRAINING_EPOCHS, batch_size=batch_size, verbose=0, callbacks=[tbCallback])
        if ep % SAVE_NETWORK_PER_N_EPISODES == 0 and ep > 0:
            # TODO: finish integrating run count into path
            self.network.save(os.path.join(SAVED_MODEL_PATH, 'network-pg-{0}.model'.format(ep)))

    def act(self, state):
        y = np.zeros([self.action_dim])
        pred_prob = self.network.predict(state)[0]
        if self.isTrain == True:
            action = np.random.choice(self.action_dim, size=1, p=np.squeeze(pred_prob))[0]
        else:
            action = np.argmax(pred_prob)
        y[action] = 1
        return y, action # y is an one-hot array, action is the index selected

    def perceive(self, state, action):
        self.replay_buffer.append([state, action])

    def discounted_rewards(self, reward):
        reward_discounted = np.zeros_like(reward)
        track = 0
        for index in reversed(range(len(reward))):
            track = track * GAMMA + reward[index]
            reward_discounted[index] = track
        return reward_discounted