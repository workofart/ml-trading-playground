from playground.dqn.dqn_rnn import DQN_RNN
from playground.utilities.utils import read_data, generate_datasets
import tensorflow as tf, numpy as np

NUM_ITER = 10
NUM_TIME_STEPS = 10

def main():
    network = DQN_RNN(4, num_time_steps=NUM_TIME_STEPS, learning_rate=0.001, name='LSTM')

    with tf.Session() as sess:
        init=tf.global_variables_initializer()
        sess.run(init)
        
        for iter in range(NUM_ITER):
            data = read_data('crypto-test-data-82hrs.csv', 'ETHBTC')

            X_train, X_test, Y_train, Y_test = generate_datasets(data)

            X_train = np.array(X_train)[:, np.newaxis]
            Y_train = np.array(Y_train)[:, np.newaxis]
            
            sess.run(network,feed_dict={network.state_input:X_train, network.y_input:Y_train})
            
            if iter %100==0:
            
                mse=network.cost.eval(feed_dict={network.state_input:X_train, network.y_input:Y_train})
                print(iter, "\tMSE",mse)
                
            y_pred=sess.run(network.outputs,feed_dict={network.state_input:X_train})

main()