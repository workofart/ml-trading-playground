from utilityV3.utils import read_data, generate_datasets, evaluate_result
from sklearn.preprocessing import normalize
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

"""
This version 3 will attempt to use tensorflow to create a multi-layer neural network
to predict the next price a particular cryptocurrency given the current
price, volume, high price, low price of a given timestep
"""

# hyperparameters
batch_size = 32 # every how many episodes to do a param update?
learning_rate = 1e-4 # feel free to play with this to train faster or more stably.
episodes = 6000

def main():
    data = read_data('crypto-test-data-82hrs.csv', 'ETHBTC')

    X_train_orig, X_test_orig, Y_train_orig, Y_test_orig = generate_datasets(data)


    print('X_train shape: ' + str(X_train_orig.shape))
    print('X_test shape: ' + str(X_test_orig.shape))
    print('Y_train shape: ' + str(Y_train_orig.shape))
            
    tf.reset_default_graph()

    input_x = tf.placeholder('float', [None, X_train_orig.shape[1]], name='input_x')
    
    W1 = tf.Variable(tf.random_normal([X_train_orig.shape[1], 16]))
    B1 = tf.Variable(tf.zeros([16]))
    layer1 = tf.nn.relu(tf.add(tf.matmul(input_x, W1), B1))
    
    W2 = tf.Variable(tf.random_normal([16, 10]))
    B2 = tf.Variable(tf.zeros([10]))
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, W2), B2))
    
    W_O = tf.Variable(tf.random_normal([10, 1]))
    B3 = tf.Variable(tf.zeros([1]))
    output = tf.add(tf.matmul(layer2, W_O), B3)

    # Variables used to train the network
    tvars = tf.trainable_variables()
    input_y = tf.placeholder('float', [None, Y_train_orig.shape[1]], name='input_y')

    # Loss function
    cost = tf.reduce_mean(tf.square(output - input_y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()

        sess.run(init)
        for e in range(episodes):
            avg_cost = 0.0
            total_batch = int((X_train_orig.shape[0]) / batch_size)
            x_batches = np.array_split(X_train_orig, total_batch)
            y_batches = np.array_split(Y_train_orig, total_batch)

            for i in range(total_batch):
                batch_x, batch_y = x_batches[i], y_batches[i]
                _, c = sess.run([optimizer, cost], 
                                feed_dict={
                                    input_x: batch_x, 
                                    input_y: batch_y, 
                                })
                avg_cost += c / total_batch
            if e % 2 == 0:
                print("Epoch:", '%04d' % (e+1), "cost=", \
                    "{}".format(avg_cost))
                # print(sess.run(output))

        print("Optimization Finished!")

        pred_train = sess.run(output, feed_dict={input_x:X_train_orig[:100]})
        pred_test =  sess.run(output, feed_dict={input_x:X_test_orig[:100]})

        evaluate_result(pred_train, X_train_orig, Y_train_orig, 'train')
        evaluate_result(pred_test, X_test_orig, Y_test_orig, 'test')

main()