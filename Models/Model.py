import numpy as np

'''
The super class for models. A model contains a way to predict and a way to compute the gradient with respect to the parameters.
'''


class Model:
    def __init__(self, name="defaultName"):
        self.name = name
        self.output = 0

    def predict(self, data):
        pass
        #return self.output

    def get_gradient_sufficient_statistics(self, data, grad_output):
        # Returns the gradient with respect to the parameters of the entire model.
        # grad_output is the gradient with respect to the output of the model.
        pass

    def get_gradient_from_sufficient_statistics(self, data, gradient_sufficient_statistics):
        pass

    def update(self, gradient, stepsize, n_seen):
        pass

    def regularize(self, l1_regularizer, l2_regularizer, stepsize, n_seen):
        pass

    def predict_batch(self, data, sparse_data=None):
        # Same as predict but uses small batches and accepts sparse data.

        n_data = data.shape[0]
        n_batches = np.floor(n_data/1000).astype(int)
        prediction = []

        # Let's grab the examples 1000 by 1000
        for i in range(n_batches):
            datum = data[i*1000:(i+1)*1000] # Extract the datapoints.
            if sparse_data is not None:
                sparse_datum = np.array(sparse_data[i*1000:(i+1)*1000].todense())
                datum = np.hstack((datum, sparse_datum))
            prediction.append(self.predict(datum))

        # The last one.
        datum = data[1000*n_batches:]  # Extract the datapoints.
        if sparse_data is not None:
            sparse_datum = np.array(sparse_data[1000*n_batches:].todense())
            datum = np.hstack((datum, sparse_datum))
        prediction.append(self.predict(datum))
        self.output = np.concatenate(prediction[:], axis=0)
        return self.output

    def compute_regularize(self, l1_regularizer, l2_regularizer):
        # Computes the regularization loss for the model.
        pass