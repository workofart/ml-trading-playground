from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adagrad
from utility.utils import read_data, generate_datasets
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
# model.compile(optimizer=SGD(lr=0.03), loss='mse') # Stochastic Gradient Descent
model.compile(optimizer=Adagrad(lr=0.01), loss='mse') # Adam

model.fit(X_train, Y_train, epochs=50, batch_size=4)

model.evaluate(X_test, Y_test, batch_size=8)
y_hat = model.predict(X_train, batch_size=8)


plt.plot(np.squeeze(y_hat)[0:100], marker=None,
         color='red', markersize=1, linewidth=1)
plt.plot(np.squeeze(Y_train)[0:100], marker=None,
         color='blue', markersize=1, linewidth=1)
plt.ylabel('normalized price')
plt.xlabel('time step')
plt.title("Predicted Prices")
plt.legend(['predict', 'true'], loc='upper left')
plt.show()