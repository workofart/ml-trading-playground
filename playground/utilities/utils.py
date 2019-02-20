import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize as norm2

def read_data(filename, ticker):
    data = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', filename)))
    if ticker:
        data = data[data['ticker'] == ticker]
    data['timestamp'] = pd.to_datetime(data.timestamp)
    data = data[['high', 'low', 'price', 'volume', 'timestamp']].sort_values(by='timestamp')
    data = data.set_index('timestamp')

    data = norm2(data.values, axis=0) # If 1, independently normalize each sample, otherwise (if 0) normalize each feature.
    return data

def normalize(data):
    """
    axis = 1, along each column, mean of all rows
    axis = 0, along each row, mean of all cols
    """
    return (data - np.mean(data, axis=0, keepdims=True)) / np.sqrt(np.var(data, axis=0, dtype=np.float64, keepdims=True))


def generate_datasets(data):
    # price, high, low, volume at time N are the x-vars
    # price at time N + 1 is the y-var
    X = data[['price', 'high', 'low', 'volume']][0: -1]
    Y = data[['price']][1:]
    X = (X.values)
    Y = (Y.values)
    assert (X.shape[0] == Y.shape[0])  # number of samples match
    assert (X.shape[1] == 4)
    assert (Y.shape[1] == 1)

    X = normalize(X)
    Y = normalize(Y)

    # X = norm2(X, axis=0) # Currently disabled
    # Y = norm2(Y, axis=0) # # Currently disabled


    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.33, random_state=42)

    # Due to the differences between Keras and the hand-coded forward/backward prop implementations
    # the orientation of the data is different. shape = (row, col), where row = # samples, col = # features
    # Therefore, transposition is not necessary
    # X_train = X_train.T
    # X_test = X_test.T
    # Y_train = Y_train.T
    # Y_test = Y_test.T

    return X_train, X_test, Y_train, Y_test

def evaluate_result(pred, x, y, mode):
    plt.plot(np.squeeze(pred)[0:100], marker=None,
            color='red', markersize=1, linewidth=1)
    plt.plot(np.squeeze(y)[0:100], marker=None,
            color='blue', markersize=1, linewidth=1)
    plt.ylabel('normalized price')
    plt.xlabel('time step')
    plt.title(mode + " Predicted Prices")
    plt.legend(['predict', 'true'], loc='upper left')
    plt.show()

def plot_trades(EP, prices, actions):
    plt.clf()
    plt.plot(prices, linewidth=1, color='#808080')
    buys, sells = {}, {}
    buys['x'] = []
    buys['y'] = []
    sells['x'] = []
    sells['y'] = []
    for i, action in enumerate(actions):
        if action == 0:
            buys['x'].append(i)
            buys['y'].append(prices[i])
        elif action == 1:
            sells['x'].append(i)
            sells['y'].append(prices[i])
    plt.plot(buys['x'], buys['y'], '^', markersize=3, color='g')
    plt.plot(sells['x'], sells['y'], 'v', markersize=3, color='r')
    plt.ylabel('Prices')
    plt.xlabel('Timesteps')
    plt.savefig(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'snapshots', 'test_trades_EP{}.png'.format(EP))))
    # plt.show()

def plot_reward(rewards):
    plt.clf()
    plt.plot(rewards)
    plt.ylabel('Avg Reward')
    plt.xlabel('Timesteps')
    plt.draw()
    plt.pause(0.001)
    # plt.show()

def plot_data(rewards, costs, q_vals, drawdown):
    plt.clf()
    avg_rewards = np.empty(len(rewards))
    avg_rewards.fill(np.mean(rewards))
    avg_rewards = avg_rewards.tolist()
    plt.subplot(4,1,1)
    plt.plot(rewards)
    plt.plot(avg_rewards, color='yellow')
    plt.ylabel('Rewards')
    plt.xlabel('Timesteps')
    
    plt.subplot(4,1,2)
    plt.plot(drawdown)
    plt.ylabel('Drawdown')
    plt.xlabel('Timesteps')

    plt.subplot(4,1,3)
    plt.plot(costs)
    plt.ylabel('Costs')
    plt.xlabel('Timesteps')

    plt.subplot(4,1,4)
    plt.plot(q_vals)
    plt.ylabel('Q Value Variance')
    plt.xlabel('Timesteps')

    plt.draw()
    plt.pause(0.001)
    

def generateTimeSeriesBatches(data, input_size, num_steps):
    seq = [np.array(data[i * input_size: (i + 1) * input_size]) 
       for i in range(len(data) // input_size)]

    # Split into groups of `num_steps`
    X = np.array([seq[i: i + num_steps] for i in range(len(seq) - num_steps)])
    y = np.array([seq[i + num_steps] for i in range(len(seq) - num_steps)])
    return X, y