import numpy as np

from Core import baseFunctions


class Layer:
    def __init__(self, input_size, output_size, activ):
        self.input_size = input_size
        self.output_size = output_size
        self.activ = activ
        self.initialize()
    
    def initialize(self):
        input_size = self.input_size
        output_size = self.output_size
        self.biases = np.zeros(output_size)
        if output_size > 1:
            self.weights = (np.random.rand(input_size, output_size) - .5)/np.sqrt(input_size)
        else:
            self.weights = (np.random.rand(input_size) - .5)/np.sqrt(input_size)
    
    def fprop(self, input):
        """ We want to store the output for the computation of the gradient. """
        self.output = baseFunctions.activ(self, np.dot(input, self.weights) + self.biases)

    def get_gradient_sufficient_statistics(self, grad_output_after, input):
        grad_output_before = grad_output_after * baseFunctions.grad_activ(self, self.output)
        gradient_sufficient_statistics = []
        gradient_sufficient_statistics.append(grad_output_before)
        gradient_sufficient_statistics.append(input)
        return gradient_sufficient_statistics

    def get_gradient_from_sufficient_statistics(self, gradient_sufficient_statistics, input = None):
        if input == None:
            grad_output_before = gradient_sufficient_statistics[0]
            input = gradient_sufficient_statistics[1] # We don't store the input for the bottom layer.
        else:
            grad_output_before = gradient_sufficient_statistics
        n_data = grad_output_before.shape[0]
        grad_weights = np.dot(input.T, grad_output_before) / n_data
        grad_biases = np.mean(grad_output_before, axis=0)
        gradient = []
        gradient.append(grad_weights)
        gradient.append(grad_biases)
        return gradient
        
    def bprop(self, grad_output_before):
        if grad_output_before.ndim == 1:
            return np.outer(grad_output_before, self.weights)
        else:
            return np.dot(grad_output_before, self.weights.T)

    def update(self, gradient, stepsize, n_seen):
        self.weights -= gradient[0]*stepsize/n_seen
        self.biases -= gradient[1]*stepsize/n_seen

    def regularize(self, l1_regularizer, l2_regularizer, stepsize, n_seen):
        if l2_regularizer > 0:
            self.weights *= (1 - l2_regularizer*stepsize)
        if l1_regularizer > 0:
            self.weights *= np.fmax(0, 1 - l1_regularizer*stepsize/np.abs(self.weights))

    def compute_regularize(self, l1_regularizer, l2_regularizer):
        loss = 0
        if l2_regularizer > 0:
            loss += np.sum(self.weights**2) * 0.5 * l2_regularizer
        if l1_regularizer > 0:
            loss += np.sum(np.abs(self.weights)) * l1_regularizer
        return loss