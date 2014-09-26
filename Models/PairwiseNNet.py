""" Pairwise neural networks """
from Models.Model import *


class PairwiseNNet(Model):

    def __init__(self, nnetSingle, nnetPair, name = "defaultPairwiseNNet"):
        nnetSingle.layers[-1].activ = 'none'
        nnetPair.layers[-1].activ = 'none'
        self.name = name
        self.nnetSingle = nnetSingle
        self.nnetPair = nnetPair
    
    def predict(self, data):
        self.nnetSingle.predict(data)
        prediction = self.nnetSingle.output
        output = np.zeros(prediction.shape)

        for i in range(10):
            randomOrder = np.random.permutation(data.shape[0])
            dataRoll = data[randomOrder]

            if self.nnetPair.layers[0].input_size == data.shape[1]:
                pairData = np.abs(data - dataRoll)
            else:
                pairData = np.hstack((data, dataRoll, np.abs(data - dataRoll)))
            self.nnetPair.predict(pairData)

            # Then, for each datapoint, we marginalize over the class
            predictionRoll = prediction[randomOrder]
            predictionPair = self.nnetPair.output

            output += np.exp(prediction) * np.sum(np.exp(predictionRoll), axis=1).reshape(-1, 1)
            output += np.exp(prediction) * np.exp(predictionRoll) * (np.exp(predictionPair) - 1)

        return output / np.sum(output, axis=1).reshape(-1, 1)
            
    def get_gradient(self, data, grad_output):
        pass

    def get_gradient_sufficient_statistics(self, data, grad_output):
        pass

    def get_gradient_from_sufficient_statistics(self, data, gradient_sufficient_statistics):
        pass

    def update(self, gradient, l1_regularizer, l2_regularizer, stepsize, n_seen):
        pass

    def compute_regularize(self, l1_regularizer, l2_regularizer):
        pass