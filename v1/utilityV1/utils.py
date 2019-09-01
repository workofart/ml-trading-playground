import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from .parameter_utils import initialize_parameters_deep, update_parameters
from .forward_prop_utils import L_model_forward
from .backward_prop_utils import L_model_backward


def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cost resulting from the cost function
    """

    # m = Y.shape[1] # Not Used

    # Compute loss from aL and y.
    # TODO: refactor this to support multi-cost functions

    # cost for cross-entropy - sigmoid function/binary classification
    # cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))

    # Mean Squared Error
    cost = (np.square(AL - Y)).mean(axis=1)

    # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    return cost


def predict(X, y, parameters, mode):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    # n = len(parameters) // 2  # number of layers in the neural network, not used
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    # For linear values output
    for i in range(0, probas.shape[1]):
        p[0, i] = probas[0, i]

    # print results
    # print ("predictions: " + str(p[:10]))
    # print ("true labels: " + str(y[:10]))

    plt.plot(np.squeeze(p)[0:100], marker=None,
             color='red', markersize=1, linewidth=1)
    plt.plot(np.squeeze(y)[0:100], marker=None,
             color='blue', markersize=1, linewidth=1)
    plt.ylabel('normalized price')
    plt.xlabel('time step')
    plt.title(mode + " Predicted Prices")
    plt.legend(['predict', 'true'], loc='upper left')
    plt.show()

    return p


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=6000, print_cost=False):
    # lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []  # keep track of cost

    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        # Compute cost.
        cost = compute_cost(AL, Y)

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)

        # Update parameters.
        parameters = update_parameters(
            parameters, grads, learning_rate=learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per tens)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()

    return parameters


def normalize(data):
    """
    axis = 1, along each column, mean of all rows
    axis = 0, along each row, mean of all cols
    """
    return (data - np.mean(data, axis=0, keepdims=True)) / np.sqrt(np.var(data, axis=0, dtype=np.float64, keepdims=True))


def read_data(filename):
    return pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', filename)))


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
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.33, random_state=42)
    X_train = X_train.T
    X_test = X_test.T
    Y_train = Y_train.T
    Y_test = Y_test.T

    return X_train, X_test, Y_train, Y_test
