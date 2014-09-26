""" Neural networks """
from Models.layer import *
from Models.Model import *
from Models.layer_embed import *


class NNetEmbed(Model):

    def __init__(self, input_size, output_size, activ = ["none"], hidden_sizes = np.zeros(0), name = "defaultNNet", list_embedding=None, list_sizes=None):
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        n_layers = len(hidden_sizes) + 2
        self.n_layers = n_layers
        self.layers = []
        self.list_sizes = list_sizes
        self.list_embedding = list_embedding
        
    
        # If there are no hidden layers, directly connect the input layer to the output layer.
        input_size = np.sum(list_embedding) + np.sum(list_sizes[np.where(list_embedding == 0)])
        self.layers.append(LayerEmbed(list_sizes, list_embedding ))
        hidden_sizes = np.array([self.layers[0].output_size, hidden_sizes[0]])
        for hidden in range(n_layers - 2):
            self.layers.append(Layer(hidden_sizes[hidden], hidden_sizes[hidden+1], activ[0]))
            
        self.layers.append(Layer(hidden_sizes[-1], output_size, activ[1]))

    def predict(self, data):
        self.layers[0].fprop(data)
        for hidden in range(1, self.n_layers):
            self.layers[hidden].fprop(self.layers[hidden-1].output)
        self.output = self.layers[-1].output
        return self.layers[-1].output

    def get_gradient_sufficient_statistics(self, grad_output, data):
        ''' Returns the gradient with respect to the parameters of the entire model.
        grad_output is the gradient with respect to the output of the model.'''
        n_layers = self.n_layers
        gradient_sufficient_statistics = []

        for hidden in range(n_layers -1, 0, -1):
            gradient_sufficient_statistics.insert(0, self.layers[hidden].get_gradient_sufficient_statistics(grad_output, self.layers[hidden-1].output))
            # gradient[0] because it's always the one we just introduced.
            grad_output = self.layers[hidden].bprop(gradient_sufficient_statistics[0][0])

        # For the last layer, we only keep the gradient with respect to the output.
        gradient_sufficient_statistics.insert(0, self.layers[0].get_gradient_sufficient_statistics(grad_output, data)[0])
        return gradient_sufficient_statistics

    def get_gradient_from_sufficient_statistics(self, gradient_sufficient_statistics, data):
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

    def compute_regularize(self, l1_regularizer, l2_regularizer):
        loss = 0
        for layer in self.layers:
            loss += layer.compute_regularize(l1_regularizer, l2_regularizer)
        return loss