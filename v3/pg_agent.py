import numpy as np, random
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
BATCH_SIZE = 16 # size of minibatch
LEARNING_RATE = 1e-4
TRAINING_EPOCHS = 100
NEURONS_PER_DIM = 32

# Vector Shapes
INPUT_DIM = [1, 6]
class PG_Agent():

    def __init__(self, env, data):
        # init some parameters
        self.epsilon = INITIAL_EPSILON
        self.env = env
        self.replay_buffer = []
        self.state_dim = env.observation_space.shape[1] + 2 # portfolio and cash
        self.action_dim = len(env.action_space)
        self.state_input = np.zeros((1, self.state_dim))
        self.y_input = np.zeros((1, self.action_dim))
        self.create_pg_network(data)
        # self.create_supervised_accuracy(self.model, x)
    
    def create_pg_network(self, data):
        model = Sequential()
        model.add(Dense(self.state_dim*NEURONS_PER_DIM, input_shape=(self.state_dim,), activation='relu', kernel_initializer='random_uniform'))
        model.add(Dense(self.action_dim, activation='softmax'))
        # model.compile(optimizer=Adagrad(lr=0.01), loss=categorical_crossentropy) # Adam
        model.compile(optimizer='rmsprop', loss=categorical_crossentropy) # Adam
        self.network = model
    
    def train_pg_network(self):
        replay_samples = random.sample(self.replay_buffer, BATCH_SIZE * 5)
        state_batch = np.squeeze([data[0] for data in replay_samples])
        y_batch = np.squeeze([data[1] for data in replay_samples])
        self.network.fit(state_batch, y_batch, epochs=TRAINING_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        self.replay_buffer = []

    # TODO: TBD
    def accuracy(self, model, x):
        return model.evaluate(x, self.y_input, verbose=1)
    
    def policy_forward(self, state):
        y = np.zeros([self.action_dim])
        pred_prob = self.network.predict(state)[0]
        action = np.random.choice(self.action_dim, size=1, p=np.squeeze(pred_prob))[0]
        y[action] = 1
        return y, action # y is an one-hot array, action is the index selected

    def act(self, state):
        y = np.zeros([self.action_dim])
        pred_prob = self.network.predict(state)
        action = np.argmax(pred_prob)
        y[action] = 1
        return y, action

    def perceive(self, states, actions):
        temp = []
        for index, value in enumerate(states):
            temp.append([states[index], [actions[index]]])
        self.replay_buffer += temp

    def discounted_rewards(self, reward):
        reward_discounted = np.zeros_like(reward)
        track = 0
        for index in reversed(range(len(reward))):
            track = track * GAMMA + reward[index]
            reward_discounted[index] = track
        return reward_discounted