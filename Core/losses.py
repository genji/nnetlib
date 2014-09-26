import numpy as np

'''
Lists all the possible losses.
'''


def compute_loss(loss_type, pred, labels):
    if loss_type == 'l2_loss':
        return l2_loss(pred, labels)
    elif loss_type == 'l1_loss':
        return l1_loss(pred, labels)
    elif loss_type == 'log_loss':
        return log_loss(pred, labels)
    elif loss_type == 'classif_loss':
        return classif_loss(pred, labels)
    else:
        print("Unknown loss_type")


def l2_loss(pred, labels):
    if pred.ndim > 1:
        return 0.5*np.sum((pred - labels)**2, axis=1)
    else:
        return 0.5*(pred - labels)**2


def l1_loss(pred, labels):
    if pred.ndim > 1:
        return np.sum(np.abs(pred - labels), axis=1)
    else:
        return np.abs(pred - labels)


def log_loss(pred, labels):
    # If there is only one column, it's the logistic loss. Otherwise, it's the softmax loss.
    eps = 1e-12  # To avoid log(0)
    if pred.ndim == 1:
        return -labels*np.log(pred + eps) - (1 - labels)*np.log(1 - pred + eps)
    else:
        return -np.sum(labels*np.log(pred + eps), axis=1)


def classif_loss(pred, labels):
    if pred.ndim == 1:
        return (pred > 0.5) != labels
    else:
        return np.argmax(pred, 1) != np.argmax(labels, 1)


def compute_gradient(loss_type, pred, labels):
    if loss_type == 'l2_loss':
        return grad_l2_loss(pred, labels)
    elif loss_type == 'l1_loss':
        return grad_l1_loss(pred, labels)
    elif loss_type == 'log_loss':
        return grad_log_loss(pred, labels)
    else:
        print("Unknown loss_type")


def grad_l2_loss(pred, labels):
    return pred - labels


def grad_l1_loss(pred, labels):
    return np.sign(pred - labels)


def grad_log_loss(pred, labels):
    # If there is only one column, it's the logistic loss. Otherwise, it's the softmax loss.
    eps = 1e-12  # To avoid log(0)
    if pred.ndim == 1:
        return -labels/(pred - eps) + (1 - labels)/(1 - pred + eps)
    else:
        return - labels + pred  # This is absolutely wrong and should be changed once I know how to design code.
