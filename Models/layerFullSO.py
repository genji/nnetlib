from Models.Model import *

import Core.baseFunctions
import numpy as np

class LayerFullSO(Model):
    def __init__(self, input_size, output_size, list_sizes, activ):
        self.input_size = input_size
        self.output_size = output_size
        self.activ = activ

        # list_sizes is an array containing the number of dimensions for each variable
        self.list_sizes = list_sizes
        self.initialize()

    def initialize(self):
        input_size = self.input_size
        output_size = self.output_size
        self.biases = np.zeros(output_size)
        if output_size > 1:
            self.weights = (np.random.rand(input_size, output_size) - .5)/np.sqrt(input_size)
        else:
            self.weights = (np.random.rand(input_size) - .5)/np.sqrt(input_size)

        # Initialize the matrices for all cross_features. Yes, it's hardcoded for now. It's 4:15AM, sue me.
        self.matrices = []
        self.matrices.append(np.random.rand(self.list_sizes[0], self.list_sizes[1]))
        self.matrices.append(np.random.rand(self.list_sizes[0], self.list_sizes[2]))
        self.matrices.append(np.random.rand(self.list_sizes[1], self.list_sizes[2]))

    def predict(self, input):
        """ We want to store the output for the computation of the gradient. """
        linear_part = np.dot(input, self.weights) + self.biases

        index_modality = 0
        input_var_1 = input[:, index_modality:index_modality+self.list_sizes[0]]
        index_modality += self.list_sizes[0]
        input_var_2 = input[:, index_modality:index_modality+self.list_sizes[1]]
        index_modality += self.list_sizes[1]
        input_var_3 = input[:, index_modality:index_modality+self.list_sizes[2]]

        quadratic_part = 0
        quadratic_part += np.sum(np.dot(input_var_1, self.matrices[0]) * input_var_2, axis=1)
        quadratic_part += np.sum(np.dot(input_var_1, self.matrices[1]) * input_var_3, axis=1)
        quadratic_part += np.sum(np.dot(input_var_2, self.matrices[2]) * input_var_3, axis=1)

        self.output = Core.baseFunctions.activ(self, linear_part + quadratic_part)

        return self.output

    def get_gradient_sufficient_statistics(self, grad_output_after, input):
        grad_output_before = grad_output_after * Core.baseFunctions.grad_activ(self, self.output)
        gradient_sufficient_statistics = grad_output_before
        return gradient_sufficient_statistics

    def get_gradient_from_sufficient_statistics(self, gradient_sufficient_statistics, input):
        grad_output_before = gradient_sufficient_statistics
        n_data = grad_output_before.shape[0]
        grad_weights = np.dot(input.T, grad_output_before) / n_data
        grad_biases = np.mean(grad_output_before, 0)

        index_modality = 0
        input_var_1 = input[:, index_modality:index_modality+self.list_sizes[0]]
        index_modality += self.list_sizes[0]
        input_var_2 = input[:, index_modality:index_modality+self.list_sizes[1]]
        index_modality += self.list_sizes[1]
        input_var_3 = input[:, index_modality:index_modality+self.list_sizes[2]]

        grad_matrix_1 = np.dot(input_var_1.T, input_var_2*grad_output_before.reshape(-1, 1)) / n_data
        grad_matrix_2 = np.dot(input_var_1.T, input_var_3*grad_output_before.reshape(-1, 1)) / n_data
        grad_matrix_3 = np.dot(input_var_2.T, input_var_3*grad_output_before.reshape(-1, 1)) / n_data

        gradient = []
        gradient.append(grad_weights)
        gradient.append(grad_biases)
        gradient.append(grad_matrix_1)
        gradient.append(grad_matrix_2)
        gradient.append(grad_matrix_3)
        return gradient

    def update(self, gradient, l1_regularizer, l2_regularizer, stepsize, n_seen):
        self.weights -= gradient[0]*stepsize/n_seen
        self.matrices[0] -= gradient[2]*stepsize/n_seen
        self.matrices[1] -= gradient[3]*stepsize/n_seen
        self.matrices[2] -= gradient[4]*stepsize/n_seen
        if l2_regularizer > 0:
            self.weights *= (1 - l2_regularizer*stepsize)
            self.matrices[0] *= (1 - l2_regularizer*stepsize)
            self.matrices[1] *= (1 - l2_regularizer*stepsize)
            self.matrices[2] *= (1 - l2_regularizer*stepsize)
        if l1_regularizer > 0:
            self.weights *= np.fmax(0, 1 - l1_regularizer*stepsize/np.abs(self.weights))
        self.biases -= gradient[1]*stepsize/n_seen

    def compute_regularize(self, l1_regularizer, l2_regularizer):
        loss = 0
        if l2_regularizer > 0:
            loss += np.sum(self.weights**2) * 0.5 * l2_regularizer
            loss += np.sum(self.matrices[0]**2) * 0.5 * l2_regularizer
            loss += np.sum(self.matrices[1]**2) * 0.5 * l2_regularizer
            loss += np.sum(self.matrices[2]**2) * 0.5 * l2_regularizer
        if l1_regularizer > 0:
            loss += np.sum(np.abs(self.weights)) * l1_regularizer
        return loss