import tensorflow as tf
import numpy as np


class DQN_Agent():

    def __init__(self, data):
        # init some parameters
        self.epsilon = INITIAL_EPSILON
        self.env = env
        self.replay_buffer = deque(maxlen=5000)
        self.state_dim = env.observation_space.shape[1] + 2 # portfolio and cash
        self.action_dim = len(env.action_space)
        self.state_input = np.zeros((1, self.state_dim))
        self.y_input = np.zeros((1, self.action_dim))
        self.create_pg_network(data)
        
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

        global summary_writer
        summary_writer = tf.summary.FileWriter('logs',graph=self.session.graph)
        

    def create_dqn_network(self, data):
        # network weights
		W1 = self.weight_variable([self.state_dim,data_dictionary["hidden_layer_1_size"]])
		variable_summaries(W1, "layer1/weights")
		b1 = self.bias_variable([data_dictionary["hidden_layer_1_size"]])
		variable_summaries(b1, "layer1/bias")
		W2 = self.weight_variable([data_dictionary["hidden_layer_1_size"],self.action_dim])
		variable_summaries(W2, "layer2/weights")
		b2 = self.bias_variable([self.action_dim])
		variable_summaries(b2, "layer2/bias")
		#tf.scalar_summary("second_layer_bias_scaler", b2)
		self.b2 = b2
		# input layer
		self.state_input = tf.placeholder("float",[None,self.state_dim])
		# hidden layers
		h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
		# Q Value layer

        self.Q_value