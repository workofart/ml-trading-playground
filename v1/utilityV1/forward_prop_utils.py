import numpy as np
from .activation_functions import relu, sigmoid, leaky_relu


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation, learning_rate=0.0075):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    elif activation == 'lrelu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = leaky_relu(Z, learning_rate)

    elif activation == 'linear':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A = activation_cache = Z

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters, last_activation='linear', learning_rate=0.0075):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    last_activation -- the last layer's activation function, usually different from the first "n" layers
    learning_rate -- how much each gradient descent step should

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(1, L):
        # print('Forward pass layer [%s]' % str(l))
        A_prev = A
        A, cache = linear_activation_forward(A_prev,
                                             parameters["W" + str(l)],
                                             parameters["b" + str(l)],
                                             activation='relu',
                                             learning_rate=learning_rate)
        caches.append(cache)

    AL, cache = linear_activation_forward(A,
                                          parameters["W" + str(L)],
                                          parameters["b" + str(L)],
                                          activation=last_activation,
                                          learning_rate=learning_rate)
    caches.append(cache)

    assert(AL.shape == (1, X.shape[1]))

    return AL, caches
