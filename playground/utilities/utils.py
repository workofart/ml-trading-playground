import numpy as np
import matplotlib.pyplot as plt
import os, re, shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize as norm2
import tensorflow as tf

def get_minute_data(data):
    return data.asfreq('T', method='bfill')

def read_data(filename, ticker, freq='raw'):
    data = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', filename)))
    if ticker:
        data = data[data['ticker'] == ticker]
    data['timestamp'] = pd.to_datetime(data.timestamp)
    data = data[['high', 'low', 'price', 'volume', 'timestamp']].sort_values(by='timestamp')
    data = data.set_index('timestamp')

    if freq == 'm':
        data = get_minute_data(data)

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

def plot_trades(EP, prices, actions, permitted_trades, name=''):
    def get_buys_sells(actions):
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
        return buys, sells
    plt.clf()
    plt.subplot(2,1,1)
    plt.plot(prices, linewidth=1, color='#808080')
    buys, sells = get_buys_sells(actions)
    plt.plot(buys['x'], buys['y'], '.', markersize=2, color='g')
    plt.plot(sells['x'], sells['y'], '.', markersize=2, color='r')
    plt.ylabel('Prices')
    plt.xlabel('Timesteps')
    plt.title('Agent\'s Intended Actions')
    # Permitted Trades
    plt.subplot(2,1,2)
    plt.plot(prices, linewidth=1, color='#808080')
    p_buys, p_sells = get_buys_sells(permitted_trades)
    plt.plot(p_buys['x'], p_buys['y'], '.', markersize=2, color='g')
    plt.plot(p_sells['x'], p_sells['y'], '.', markersize=2, color='r')
    plt.ylabel('Prices')
    plt.xlabel('Timesteps')
    plt.title('Agent\'s Permitted Actions (Actual Trades)')
    
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'logs', str(get_latest_run_count() - 1), 'test_trades'))
    if os.path.isdir(path) is False:
        os.mkdir(path)
    plt.savefig(path + '/{0}EP{1}.png'.format(name, EP), dpi=400)

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

def variable_summaries(var):
    out = []
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    # Taken from https://www.tensorflow.org/guide/summaries_and_tensorboard
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        out.append(tf.summary.scalar(var.op.name + '_mean', mean))
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        out.append(tf.summary.scalar(var.op.name + '_stddev', stddev))
        out.append(tf.summary.scalar(var.op.name + '_max', tf.reduce_max(var)))
        out.append(tf.summary.scalar(var.op.name + '_min', tf.reduce_min(var)))
        out.append(tf.summary.histogram(var.op.name + '_histogram', var))
        return out

def log_histogram(writer, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)
        
        # Create histogram using numpy        
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        writer.add_summary(summary, step)
        writer.flush()

def log_scalars(writer, tag, values, step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=values)])
    writer.add_summary(summary, step)
    writer.flush()

def cleanup_logs():
    pattern = 'events.out.tfevents.*'
    log_dir_pattern = 'train_*'
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'logs'))

    parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    for f in os.listdir(path):
        if re.search(pattern, f):
            os.remove(os.path.join(path, f))

    for f in os.listdir(parent_path):
        if re.search(log_dir_pattern, f):
            shutil.rmtree(os.path.join(parent_path, f), ignore_errors=True)

def get_latest_run_count(root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'logs'))):
    dirs = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
    if len(dirs) == 0:
        return 0
    else:
        return int(max(dirs)) + 1

def update_target_graph(from_scope, to_scope):
    # Get the parameters of our DQNNetwork
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    
    # Get the parameters of our Target_network
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    
    # Update our target_q_network parameters with q_network parameters
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder