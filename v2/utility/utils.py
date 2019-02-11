import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize as norm2


def read_data(filename):
    return pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', filename)))

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

    # X = norm2(X, axis=0)
    # Y = norm2(Y, axis=0)


    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.33, random_state=42)
    # X_train = X_train.T
    # X_test = X_test.T
    # Y_train = Y_train.T
    # Y_test = Y_test.T

    return X_train, X_test, Y_train, Y_test
