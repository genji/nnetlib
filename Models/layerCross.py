import base_functions
import numpy as np

class LayerCross:
    def __init__(self, input_size, output_size, list_sizes, list_embeddings, activ):
        self.input_size = input_size
        self.output_size = output_size
        self.activ = activ

        # list_sizes is an array containing the number of dimensions for each variable
        # list_embeddings is the size of the embeddings for each variable. If set to 0, no embedding is created.
        self.list_sizes = list_sizes
        self.list_embeddings = list_embeddings

        self.initialize()

    def initialize(self):
        input_size = self.input_size
        output_size = self.output_size
        self.biases = np.zeros(output_size)
        if output_size > 1:
            self.weights = (np.random.rand(input_size, output_size) - .5)/np.sqrt(input_size)
        else:
            self.weights = (np.random.rand(input_size) - .5)/np.sqrt(input_size)

        # Initialize the matrices for all embeddings.
        self.embed_mat = []
        for subvar in range(len(self.list_sizes)):
            self.embed_mat.append(np.random.randn(list_sizes[subvar], list_embeddings[subvar]))

        # No other matrix for now, fuck it, we'll just do it with context.


    def fprop(self, input):
        """ We want to store the output for the computation of the gradient. """
        self.output = base_functions.activ(self, np.dot(input, self.weights) + self.biases)

    def get_gradient(self, input, grad_output_after):
        n_data = input.shape[0]
        grad_output_before = grad_output_after * base_functions.gradActiv(self, self.output)
        grad_weights = np.dot(input.T, grad_output_before) / n_data
        grad_biases = np.mean(grad_output_before, 0)
        gradient = dict(output_before = grad_output_before, weights = grad_weights, biases = grad_biases)
        return gradient

    def get_gradient_sufficient_statistics(self, input, grad_output_after):
        grad_output_before = grad_output_after * base_functions.gradActiv(self, self.output)
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
        grad_biases = np.mean(grad_output_before, 0)
        gradient = []
        gradient.append(grad_weights)
        gradient.append(grad_biases)
        return gradient

    def bprop(self, grad_output_before):
        if grad_output_before.ndim == 1:
            return np.outer(grad_output_before, self.weights)
        else:
            return np.dot(grad_output_before, self.weights.T)

    def update(self, gradient, l1_regularizer, l2_regularizer, stepsize, n_seen):
        self.weights -= gradient[0]*stepsize/n_seen
        if l2_regularizer > 0:
            self.weights *= (1 - l2_regularizer*stepsize)
        if l1_regularizer > 0:
            self.weights *= np.fmax(0, 1 - l1_regularizer*stepsize/np.abs(self.weights))
        self.biases -= gradient[1]*stepsize/n_seen

    def compute_regularize(self, l1_regularizer, l2_regularizer):
        loss = 0
        if l2_regularizer > 0:
            loss += np.sum(self.weights**2) * 0.5 * l2_regularizer
        if l1_regularizer > 0:
            loss += np.sum(np.abs(self.weights)) * l1_regularizer
        return loss