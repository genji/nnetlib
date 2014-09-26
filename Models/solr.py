from models import *

class solr(model):
    """ A second-order logistic model"""
    def __init__(self, input_size, output_size = 1, l1reg=1.0, l2reg=.0001, loss="mse", name = "default_solr"):
        self.type = "solr"
        self.name = name
        self.input_size = input_size # Size of the input
        self.output_size = output_size
        self.weights = np.random.randn(input_size, input_size, output_size)/np.sqrt(input_size) # Weights of the model
        self.biases = np.zeros(output_size)
        self.l1reg = max(0.0, l1reg)
        self.l2reg = max(0.0, l2reg)
        self.loss = loss

    def predict(self, data):
        """ Predicts the output for any number of data """
        output_size = self.biases.shape[0]
        output = np.zeros((data.shape[0], output_size))
        
        for dim in range(output_size):
            output[:, dim] = np.sum(data*np.dot(data, self.weights[:,:,dim]), axis=1) + self.biases[dim]

        if self.loss == 'logistic':
            if self.output_size == 1:
                self.output = sigm(output)
            else:
                self.output = softmax(output)
        elif self.loss == 'mse':
            self.output = output
        else:
            print "Unknown loss."
    
    def get_gradient_sufficient_statistics(self, data, labels):
        """ Returns the sufficient statistics of the gradient. This is useful for the SAG training algorithm. """
        self.predict(data)
        grad_output = self.output - labels
        sufficient_statistics = dict(output = grad_output)
        return sufficient_statistics

    def get_gradient(self, data, sufficient_statistics):
        grad_output = sufficient_statistics["output"]
        grad_weights = np.zeros(self.weights.shape)
        index = 0
        for datapoint in data:
            for dim in xrange(self.output_size):
                grad_weights[:,:,dim] += np.outer(datapoint, datapoint) * grad_output[index, dim]
            index += 1
        grad_biases = np.mean(grad_output, 0)
        gradient = dict(weights = grad_weights, biases = grad_biases)
        return gradient

    def update(self, gradient, stepsize, n_seen):
        """ Update the parameters of the model based on the gradient """
        self.weights -= stepsize * gradient["weights"] / n_seen
        self.biases -= stepsize * gradient["biases"] / n_seen

        # Apply regularization. The biases are not regularized.
        if self.l2reg > 0:
            mult = 1 - stepsize*self.l2reg
            self.weights *= mult

        if self.l1reg > 0:
            self.weights *= np.maximum(0, 1 - stepsize*self.l1reg/(self.weights + eps))