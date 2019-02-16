from playground.dqn.dqn_rnn import DQN_RNN
from playground.utilities.utils import read_data, generate_datasets, evaluate_result
import tensorflow as tf, numpy as np

NUM_ITER = 100
NUM_TIME_STEPS = 1
learning_rate_decay = 0.99
init_epoch = 5

def main():
    network = DQN_RNN(state_dim=4, num_time_steps=NUM_TIME_STEPS, learning_rate=0.001, name='LSTM')

    with tf.Session() as sess:
        init=tf.global_variables_initializer()
        sess.run(init)
        
        learning_rates_to_use = [
            0.001 * (
                learning_rate_decay ** max(float(i + 1 - init_epoch), 0.0)
            ) for i in range(NUM_ITER)]
        
        for epoch_step in range(NUM_ITER):
            current_lr = learning_rates_to_use[epoch_step]
            
            data = read_data('crypto-test-data-82hrs.csv', 'ETHBTC')
            X_train_orig, X_test_orig, Y_train_orig, Y_test_orig = generate_datasets(data)

            # Check https://github.com/lilianweng/stock-rnn/blob/master/data_wrapper.py
            # if you are curious to know what is StockDataSet and how generate_one_epoch() 
            # is implemented.

            # for batch_X, batch_y in stock_dataset.generate_one_epoch(config.batch_size):
            train_data_feed = {
                network.state_input: X_train_orig, 
                network.y_input: Y_train_orig, 
                network.learning_rate: current_lr
            }
            train_loss, _ = sess.run([network.loss, network.minimize], train_data_feed)
            if epoch_step % 10==0:
                mse=network.loss.eval(feed_dict={network.state_input:X_train_orig, network.y_input:Y_train_orig})
                print(epoch_step, "\tMSE",mse)
        
        # for i in range(NUM_ITER):
        #     data = read_data('crypto-test-data-82hrs.csv', 'ETHBTC')

        #     X_train_orig, X_test_orig, Y_train_orig, Y_test_orig = generate_datasets(data)

        #     X_train = np.reshape(X_train_orig, (X_train_orig.shape[0], X_train_orig.shape[1], 1))
        #     # Y_train = np.reshape(Y_train_orig, (Y_train_orig.shape[0], Y_train_orig.shape[1], 4))
        #     Y_train = Y_train_orig
            
        #     sess.run(network.optimizer,feed_dict={network.state_input:X_train, network.y_input:Y_train})
            
        #     if i % 10==0:
            
        #         mse=network.cost.eval(feed_dict={network.state_input:X_train, network.y_input:Y_train})
        #         print(i, "\tMSE",mse)
                
        y_pred=sess.run(network.prediction,feed_dict={network.state_input:X_train_orig})
        evaluate_result(y_pred, X_train_orig, Y_train_orig, mode='train')


main()