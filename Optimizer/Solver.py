from Core.losses import *

class Solver:
    def __init__(self, loss_type, model, l1_regularizer = 0.0, l2_regularizer = 0.0):
        self.loss_type = loss_type
        self.model = model
        self.l1_regularizer = l1_regularizer
        self.l2_regularizer = l2_regularizer
        
    def compute_loss_and_gradient(self, data, label, onlyLoss = False):
        '''
        Computes the loss, the gradient and the sufficient statistics of the gradient.
        However, we might sometimes only care about the loss, in which case the gradient is not computed.
        Right now, I shall return the regularized loss when asked for onlyLoss.
        '''
        output = self.model.predict(data)
        loss = compute_loss(self.loss_type, output, label)
        lossRegularizer = self.model.compute_regularize(self.l1_regularizer, self.l2_regularizer)
        totalLoss = loss + lossRegularizer

        if onlyLoss == False:
            grad_output = compute_gradient(self.loss_type, output, label)
            gradient_sufficient_statistics = self.model.get_gradient_sufficient_statistics(grad_output, data)
            gradient = self.get_gradient_from_sufficient_statistics(gradient_sufficient_statistics, data)
            return totalLoss, gradient, gradient_sufficient_statistics
        else:
            return totalLoss

    def get_gradient_from_sufficient_statistics(self, gradient_sufficient_statistics, data):
        gradient = self.model.get_gradient_from_sufficient_statistics(gradient_sufficient_statistics, data)
        return gradient

    def update(self, gradient, stepsize, n_seen):
        self.model.update(gradient, stepsize, n_seen)

    def regularize(self, stepsize, n_seen):
        self.model.regularize(self.l1_regularizer, self.l2_regularizer, stepsize, n_seen)