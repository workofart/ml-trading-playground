from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adagrad
from utilityV2.utils import read_data, generate_datasets, evaluate_result
from sklearn.preprocessing import normalize
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
This version 2 will attempt to use keras to create a multi-layer neural network
to predict the next price of a particular cryptocurrency given the current
price, volume, high price, low price of a given timestep
"""


data = read_data('crypto-test-data-82hrs.csv')
data = data[data['ticker'] == 'ETHBTC']
data['timestamp'] = pd.to_datetime(data.timestamp)
data = data[['high', 'low', 'price', 'volume', 'timestamp']].sort_values(by='timestamp')
data = data.set_index('timestamp')


X_train, X_test, Y_train, Y_test = generate_datasets(data)

print('X_train shape: ' + str(X_train.shape))
print('X_test shape: ' + str(X_test.shape))
print('Y_train shape: ' + str(Y_train.shape))

model = Sequential()
model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(Y_train.shape[1]))
model.compile(optimizer=SGD(lr=0.03), loss='mse') # Stochastic Gradient Descent
# model.compile(optimizer=Adagrad(lr=0.01), loss='mse') # Adam, to be in sync with v1, use SGD

model.fit(X_train, Y_train, epochs=6000, batch_size=X_train.shape[0]) # To be in sync with v1, no batch training

y_hat_train = model.predict(X_train, batch_size=X_train.shape[0])
evaluate_result(y_hat_train, X_train, Y_train, model, 'train')

y_hat_test = model.predict(X_test, batch_size=X_test.shape[0])
evaluate_result(y_hat_test, X_test, Y_test, model, 'test')