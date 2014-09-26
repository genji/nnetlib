""" Neural networks """
from Models.Nnet import *
from Core.losses import *


class Crf(Model):

    def __init__(self, input_size, output_size, activation=["none"], hidden_sizes=np.zeros(0), regularizer=0, name="default_nnet"):
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        n_layers = len(hidden_sizes) + 1
        self.n_layers = n_layers
        self.layers = []

        # If there are no hidden layers, directly connect the input layer to the output layer.
        if self.n_layers == 1:
            self.layers.append(Layer(input_size, output_size, activation[0], regularizer=regularizer))
        else:
            self.layers.append(Layer(input_size, hidden_sizes[0], activation[0], regularizer=regularizer))
            for hidden in range(n_layers - 2):
                self.layers.append(Layer(hidden_sizes[hidden], hidden_sizes[hidden+1], activation[0], regularizer=regularizer))
            self.layers.append(Layer(hidden_sizes[-1], output_size, activation[1], regularizer=regularizer))

    def predict(self, data):
        self.layers[0].fprop(data)
        for hidden in range(1, self.n_layers):
            self.layers[hidden].fprop(self.layers[hidden-1].output)
        self.output = self.layers[-1].output
        return self.layers[-1].output

    def get_gradient(self, data, grad_output):
        ''' Returns the gradient with respect to the parameters of the entire model.
        grad_output is the gradient with respect to the output of the model.'''
        n_layers = self.n_layers
        gradient = []

        for hidden in range(n_layers -1, 0, -1):
            gradient.insert(0, self.layers[hidden].get_gradient(self.layers[hidden-1].output, grad_output))
            # gradient[0] because it's always the one we just introduced.
            grad_output = self.layers[hidden].bprop(gradient[0]["output_before"])

        gradient.insert(0, self.layers[0].get_gradient(data, grad_output))
        return gradient

    def get_gradient_sufficient_statistics(self, data, grad_output):
        ''' Returns the gradient with respect to the parameters of the entire model.
        grad_output is the gradient with respect to the output of the model.'''
        n_layers = self.n_layers
        gradient_sufficient_statistics = []

        for hidden in range(n_layers -1, 0, -1):
            gradient_sufficient_statistics.insert(0, self.layers[hidden].get_gradient_sufficient_statistics(self.layers[hidden-1].output, grad_output))
            # gradient[0] because it's always the one we just introduced.
            grad_output = self.layers[hidden].bprop(gradient_sufficient_statistics[0][0])

        # For the last layer, we only keep the gradient with respect to the output.
        gradient_sufficient_statistics.insert(0, self.layers[0].get_gradient_sufficient_statistics(data, grad_output)[0])
        return gradient_sufficient_statistics

    def get_gradient_from_sufficient_statistics(self, data, gradient_sufficient_statistics):
        gradient = []
        for hidden in range(self.n_layers):
            if hidden == 0:
                gradient.append(self.layers[hidden].get_gradient_from_sufficient_statistics(gradient_sufficient_statistics[hidden], data))
            else:
                gradient.append(self.layers[hidden].get_gradient_from_sufficient_statistics(gradient_sufficient_statistics[hidden]))
        return gradient

    def update(self, gradient, l1_regularizer, l2_regularizer, stepsize, n_seen):
        for hidden in range(self.n_layers):
            self.layers[hidden].update(gradient[hidden], l1_regularizer, l2_regularizer, stepsize, n_seen)

    def predict_batch(self, data, sparse_data = None):
        ''' Same as predict but uses small batches and accepts sparse data.'''

        n_data = data.shape[0]
        n_batches = np.floor(n_data/1000.0).astype(int)
        prediction = []

        # Let's grab the examples 1000 by 1000
        for i in range(n_batches):
            datum = data[i*1000:(i+1)*1000] # Extract the datapoints.
            if sparse_data != None:
                sparse_datum = np.array(sparse_data[i*1000:(i+1)*1000].todense())
                datum = np.hstack((datum, sparse_datum))
            prediction.append(self.predict(datum))

        # The last one.
        datum = data[1000*n_batches:] # Extract the datapoints.
        if sparse_data != None:
            sparse_datum = np.array(sparse_data[1000*n_batches:].todense())
            datum = np.hstack((datum, sparse_datum))
        prediction.append(self.predict(datum))
        self.output = np.hstack(prediction[:])
        return self.output

    def compute_regularize(self, l1_regularizer, l2_regularizer):
        return loss