import sys

from DataAccess.DataAccess import *
from Core.losses import *

'''
An optimizer minimizes a loss by repeatedly calling a solver which outputs a value and a gradient.
'''


class Optimizer:
    """ The gradient descent optimizer used for learning. """
    def __init__(self, solver, method='sgd', read_type="random", stepsize=0.001, max_updates=10000, minibatch=50, display=500, max_data=sys.maxint):
        self.solver = solver
        self.method = method
        self.stepsize = stepsize # If stepsize = 0, it is determined automatically.
        if stepsize < 0 and method == 'sag':
            self.L_max = 0.0
        if method == 'sag-ls':
            # We initialize L_max to a very low value, it will be increased by backtracking early on.
            self.L_max = .000001
            self.stepsize = 1./(self.L_max + self.solver.l2_regularizer)
        self.max_updates = max_updates # Maximum number of iterations through the training set.
        self.minibatch = minibatch
        self.type = read_type # Do we get the data sequentially ("seq") or at random ("random")
        self.max_data = max_data
        self.display = display # Number of batches between two displays of the training error.
        self.seen = np.zeros(0)  # Have we seen the examples.
        self.n_seen = 0  # Number of examples seen.
        self.all_sufficient_statistics = []

    @staticmethod
    def add_gradient(gradient, sum_gradient):
        if type(gradient) != list:
            sum_gradient += gradient
        else:
            for i in range(len(gradient)):
                if type(gradient[i]) != list:
                    sum_gradient[i] += gradient[i]
                else:
                    for j in range(len(gradient[i])):
                        sum_gradient[i][j] += gradient[i][j]
        return sum_gradient

    @staticmethod
    def sq_norm_gradient(gradient):
        if type(gradient) != list:
            return np.sum(gradient**2)
        else:
            sq_norm_gradient = 0.
            for i in range(len(gradient)):
                if type(gradient[i]) != list:
                    sq_norm_gradient += np.sum(gradient[i]**2)
                else:
                    for j in range(len(gradient[i])):
                        sq_norm_gradient += np.sum(gradient[i][j]**2)
        return sq_norm_gradient

    @staticmethod
    def remove_gradient(gradient, sum_gradient):
        if type(gradient) != list:
            sum_gradient -= gradient
        else:
            for i in range(len(gradient)):
                if type(gradient[i]) != list:
                    sum_gradient[i] -= gradient[i]
                else:
                    for j in range(len(gradient[i])):
                        sum_gradient[i][j] -= gradient[i][j]

        return sum_gradient

    def train(self, data, labels, sparse_data=None):

        # Compute the average of the dataset.
        max_updates = self.max_updates
        minibatch = self.minibatch
        method = self.method
        stepsize = self.stepsize
        display = self.display
        solver = self.solver

        all_losses = np.zeros(0)  # The set of losses for all the batches in the dataset. We might not know in advance how many batches there will be.

        data_getter = DataAccess(data, labels, self.max_data, self.type, sparse_data)

        for i_batch in range(max_updates):
            # Draw a batch
            datum, label = data_getter.get(minibatch)
            current_batch = data_getter.current_batch

            # Compute the loss and the gradient for that batch.
            loss, gradient, gradient_sufficient_statistics = self.solver.compute_loss_and_gradient(datum, label)

            # We increase the sizes of "seen" and "all_losses" as we go along.
            if current_batch >= self.seen.shape[0]:
                self.seen.resize(np.floor(1.2*current_batch) + 1)
                all_losses.resize(np.floor(1.2*current_batch) + 1)

            # Similarly, we want to extend "all_sufficient_statistics" as we go along.
            if current_batch >= len(self.all_sufficient_statistics):
                self.all_sufficient_statistics.extend(np.zeros(current_batch + 1 - len(self.all_sufficient_statistics)))

            if self.seen[current_batch] == 1 and (method == 'sag' or method == 'sag-ls'):
                # Remove the current gradient from the sum of all gradients.
                old_gradient = self.solver.get_gradient_from_sufficient_statistics(self.all_sufficient_statistics[current_batch], datum)
                self.remove_gradient(old_gradient, self.sum_gradient)

            # If the batch has not been seen, update the counter now.
            # We will only update the status of that batch after the updates as it is used by some methods.
            if self.seen[current_batch] == 0:
                self.n_seen += 1

            if method == 'sgd':
                self.sgd_update(gradient)
            elif method == 'sag':
                self.sag_update(datum, gradient, i_batch, current_batch)
            elif method == 'sag-ls':
                self.sag_ls_update(datum, label, loss, gradient, i_batch, current_batch)
            else:
                print("Unknown method.")

            # Regularize the model.
            solver.regularize(stepsize, self.n_seen)

            # Update the status now if you need to.
            if self.seen[current_batch] == 0:
                self.seen[current_batch] = 1

            # Compute the loss on that batch.
            all_losses[current_batch] = np.mean(loss)

            # If using the line search, reduce L.
            # In the arXiv's paper, L is multiplied by 2^(-1/n) where n is the number of batches.
            # Since I do not know the number of batches beforehand, I use n_seen instead.
            if method == 'sag-ls':
                self.L_max *= 2**(-1. / float(n_seen))
                stepsize = 1./(self.L_max + self.solver.l2_regularizer)

            if (i_batch+1) % display == 0:
                print('Example {}/{} - Average train loss = {}'.format((i_batch+1)*minibatch, max_updates*minibatch, np.sum(all_losses)/n_seen))

    def sag_update(self, datum, gradient, i_batch, current_batch):
        # Stochastic average gradient
        if self.seen[current_batch] == 1:
            # Remove the current gradient from the sum of all gradients.
            old_gradient = self.solver.get_gradient_from_sufficient_statistics(self.all_sufficient_statistics[current_batch], datum)
            self.remove_gradient(old_gradient, self.sum_gradient)

            # When using SAG and an automatic stepsize, this might be the time to update the Lipschitz constant.
            if self.stepsize < 0:
                L = np.mean(np.sum(datum**2, axis=1))
                if L > self.L_max:
                    if self.solver.loss_type == 'log_loss':
                        self.L_max = L/4
                    elif self.solver.loss_type == 'l2_loss':
                        self.L_max = L
                self.stepsize = np.abs(self.stepsize)/(self.L_max + self.solver.l2_regularizer)

        # Insert the sufficient statistics.
        self.all_sufficient_statistics[current_batch] = self.gradient_sufficient_statistics

        if i_batch > 0:
            # Add the gradient to the sum of gradients.
            self.add_gradient(gradient, self.sum_gradient)
        else:
            self.sum_gradient = gradient

        self.solver.update(self.sum_gradient, self.stepsize, self.n_seen)

    def sag_ls_update(self, datum, label, loss, gradient, i_batch, current_batch):
        # Stochastic average gradient with line search.
        if self.seen[current_batch] == 1:
            # Remove the current gradient from the sum of all gradients.
            old_gradient = self.solver.get_gradient_from_sufficient_statistics(self.all_sufficient_statistics[current_batch], datum)
            self.remove_gradient(old_gradient, self.sum_gradient)

        # Insert the sufficient statistics.
        self.all_sufficient_statistics[current_batch] = self.gradient_sufficient_statistics

        if i_batch > 0:
            # Add the gradient to the sum of gradients.
            self.add_gradient(gradient, self.sum_gradient)
        else:
            self.sum_gradient = gradient

        self.solver.update(self.sum_gradient, self.stepsize, self.n_seen)
      # Make sure you respect Armijo rule.
        # Compute the squared norm of the gradient.
        sq_norm_gradient = self.sq_norm_gradient(self.sum_gradient)

        # Recompute the loss at the new location.
        new_loss = self.solver.compute_loss_and_gradient(datum, label, onlyLoss=True)

        while sq_norm_gradient > 10**-8 and np.mean(new_loss) > np.mean(loss) - sq_norm_gradient/(2 * self.stepsize):
            print("New loss = {} Threshold = {}".format(np.mean(new_loss), np.mean(loss) - sq_norm_gradient*self.stepsize/2))
            print("Backtracking step. Stepsize = {}".format(stepsize))
            old_stepsize = stepsize
            self.L_max *= 2
            stepsize = 1./(self.L_max + self.solver.l2_regularizer)
            self.solver.update(self.sum_gradient, -old_stepsize + stepsize, self.n_seen)
            new_loss = self.solver.compute_loss_and_gradient(datum, label, onlyLoss=True)

    def sgd_update(self, gradient):
        self.solver.update(gradient, self.stepsize, 1)

    def layer_ls_update(self, ):