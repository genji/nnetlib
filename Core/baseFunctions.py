import numpy as np
import pickle


def sigm(x):
    # Returns the sigmoid of x.
    small_x = np.where(x < -20)  # Avoid overflows.
    sigm_x = 1/(1 + np.exp(-x))
    sigm_x[small_x] = 0.0
    return sigm_x


def pos(x):
    # Returns the positive part of x.
    return np.fmax(x, 0)


def softmax(x):
    # Returns the softmax of x along the rows.
    # The maximum of the array is removed to avoid underflows and overflows.
    max_x = np.amax(x, axis=1)
    x = x - max_x.reshape(-1, 1)
    exp_x = np.exp(x).T
    return (exp_x/np.sum(exp_x, 0)).T


def activ(layer, data):
    """ The activation function """
    if layer.activ == 'pos':
        return pos(data)
    elif layer.activ == 'sigm':
        return sigm(data)
    elif layer.activ == 'softmax':
        return softmax(data)
    elif layer.activ == 'none':
        return data
    else:
        print("Unknown activation function.")


def grad_activ(layer, data):
    """ Compute the derivative of the activation function """
    if layer.activ == 'pos':
        return data > 0
    elif layer.activ == 'sigm':
        return data*(1- data)
    elif layer.activ == 'softmax':
        return 1  # This is absolutely wrong and should be changed once I know how to design code.
    elif layer.activ == 'none':
        return np.ones(data.shape)
    else:
        print("Unknown activation function.")


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, -1)