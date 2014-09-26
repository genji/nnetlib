import numpy as np
import random, cPickle, gzip
import pdb

eps = .00001 # To avoid log(0)

def expand(labels):
	''' Takes a vector of labels and transforms it into a matrix of one-hot encodings '''
	label_matrix = np.zeros((labels.size, max(labels)+1))
	for line in xrange(labels.shape[0]):
		label_matrix[line][labels[line]] = 1
	return label_matrix

def sigm(x):
	return 1/(1 + np.exp(-x))

def pos(x):
	return (x + np.abs(x))/2

def softmax(x):
	expXT = np.exp(x).transpose()
	return (expXT/sum(expXT, 0)).transpose()

def load_data(dataset):
	''' Loads the dataset

	:type dataset: string
	:param dataset: the path to the dataset (here MNIST)
	'''
	#############
	# LOAD DATA #
	#############

	# Load the dataset
	f = gzip.open(dataset, 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()
	train_set_x = train_set[0]
	train_set_y = train_set[1]
	valid_set_x = valid_set[0]
	valid_set_y = valid_set[1]
	test_set_x = test_set[0]
	test_set_y = test_set[1]

	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),(test_set_x, test_set_y)]
	return rval

class nnet:
	""" A fuckin neural network! """
	def __init__(self, input_size, hidden_size = 500, output_size = 1, l1reg=1.0, l2reg=.0001):
		self.input_size = input_size # Size of the input
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.input_weights = np.random.randn(input_size, hidden_size)/np.sqrt(input_size) # Weights of the model
		self.output_weights = np.zeros((hidden_size, output_size)) # Weights of the model
		self.input_biases = np.zeros(hidden_size)
		self.output_biases = np.zeros(output_size)
		self.l1reg = max(0.0, l1reg)
		self.l2reg = max(0.0, l2reg)

	def predict(self, data):
		n_data = data.shape[0]
		""" Predicts the output for any number of data """
		hidden = pos(np.dot(data, self.input_weights) + np.tile(self.input_biases, (n_data, 1)))
		return softmax(np.dot(hidden, self.output_weights) + np.tile(self.output_biases, (n_data, 1))), hidden # The softmax is the multidimensional equivalent of the sigmoid
	
	def get_gradient(self, data, labels):
		n_data = data.shape[0]
		output, hidden = self.predict(data)
		grad_output = output - labels
		grad_output_weights = np.dot(hidden.transpose(), grad_output)/n_data
		grad_output_biases = np.mean(grad_output, 0)
		grad_hidden_after = np.dot(grad_output, self.output_weights.transpose())
		grad_hidden_before = grad_hidden_after * (hidden > 0)
		grad_input_weights = np.dot(data.transpose(), grad_hidden_before)
		grad_input_biases = np.mean(grad_hidden_before, 0)
		return grad_output, grad_input_weights, grad_output_weights, grad_input_biases, grad_output_biases

	def add_neuron(self, data, gradient):
		""" Finds the datapoint in the minibatch with the largest gradient and adds it to the input layer """

	def train(self, data, labels, optimizer):
		n_data = data.shape[0]

		# Compute the average of the dataset.
		avg_data = np.mean(data, 0)
		stepsize = optimizer.stepsize
		l2reg = self.l2reg
		max_iter = optimizer.max_iter
		start_dynamic = 5
		minibatch = 20
		count = 0
		count_max = 50
		if optimizer.method == 'sgd':
			for iter in xrange(max_iter):
				for i in xrange(n_data/minibatch):
					l1reg = self.l1reg
					datum = data[i*minibatch:i*minibatch + minibatch]
					label = labels[i*minibatch:i*minibatch + minibatch]
					grad_output, grad_input_weights, grad_output_weights, grad_input_biases, grad_output_biases = self.get_gradient(datum, label)
					self.input_weights -= stepsize * grad_input_weights
					self.output_weights -= stepsize * grad_output_weights
					self.input_biases -= stepsize * grad_input_biases
					self.output_biases -= stepsize * grad_output_biases

					if iter >= start_dynamic:
						# Add an extra neuron
						worst_point = np.argmax(np.sum(np.abs(grad_output), 1))
						norm_worst = datum[worst_point]/np.linalg.norm(datum[worst_point])
	
						# Append the datapoint itself at the end of the input weights.
						self.input_weights = np.column_stack((self.input_weights, norm_worst))
	
						# Add the bias so that the activation is 0 on average.
						self.input_biases = np.append(self.input_biases, -np.dot(norm_worst, avg_data))
	
						# Do a gradient step for the output weights.
						hidden = pos(np.dot(datum, norm_worst) + self.input_biases[-1])
						self.output_weights = np.row_stack((self.output_weights, -stepsize*np.dot(hidden, grad_output/minibatch)))
	
						# Now do an iteration of group L1 of the output weights.
						norms = np.sqrt(np.sum(self.output_weights**2, 1))
	
						# We only keep the neurons whose output weights are not all 0.
						to_be_kept = np.nonzero(norms > stepsize*l1reg)[0]
						self.output_weights = np.take(self.output_weights, to_be_kept, axis=0)
						self.input_weights = np.take(self.input_weights, to_be_kept, axis=1)
						self.input_biases = self.input_biases[to_be_kept]
	
						# Project the remaining weights
						self.output_weights *= (1 - stepsize*l1reg/norms[to_be_kept]).reshape((-1, 1))
						if self.output_weights.shape[0] > 600 and count > count_max:
							self.l1reg *= 2
							count = 0
						elif self.output_weights.shape[0] < 400 and count > count_max:
							self.l1reg /= 2
							count = 0
						else:
							count += 1
						
					self.input_weights *= 1 - stepsize*l2reg

				if optimizer.display > 0 and iter%optimizer.display == 0:
					log_loss, classif_loss = self.test(data, labels)
					print "Iteration {}: {} units, log_loss = {}, classif_loss = {}".format(iter, self.input_biases.shape[0], log_loss, classif_loss)
		elif optimizer.method == 'fg':
			for iter in xrange(max_iter):
				grad_input_weights, grad_output_weights, grad_input_biases, grad_output_biases = self.get_gradient(data, labels)
				self.input_weights -= stepsize * grad_input_weights
				self.output_weights -= stepsize * grad_output_weights
				self.input_biases -= stepsize * grad_input_biases
				self.output_biases -= stepsize * grad_output_biases
				if optimizer.display > 0 and iter%optimizer.display == 0:
					log_loss, classif_loss = self.test(data, labels)
					print "Iteration {}: log_loss = {}, classif_loss = {}".format(iter, log_loss, classif_loss)
		else:
			print "Nope"
	
	def test(self, data, labels):
		n_data = data.shape[0]
		""" Compute the error on a test set """
		prediction, _ = self.predict(data)
		log_loss = -np.sum(labels*np.log(prediction))/n_data
		classif_loss = np.mean(np.argmax(prediction, 1) != np.argmax(labels, 1))
		return log_loss, classif_loss


class optimizer:
	""" The gradient descent optimizer used for learning. """
	def __init__(self, method = 'sag', stepsize = 0.001, max_iter = 30, display = 1):
		self.method = method
		self.stepsize = stepsize
		self.max_iter = max_iter # Maximum number of iterations through the training set.
		self.display = display # Do we display the training loss after each iteration?

def mlp_train():
	rval = load_data('/home/nicolas/data/mnist.pkl.gz')
	train_set_x, train_set_y = rval[0]
	valid_set_x, valid_set_y = rval[1]
	test_set_x, test_set_y = rval[2]
	train_set_y = expand(train_set_y)
	valid_set_y = expand(valid_set_y)
	test_set_y = expand(test_set_y)

	used_set_x = train_set_x # The set actually used for training
	used_set_y = train_set_y
	max_norm = max(np.sum(used_set_x**2, 1))

	model = nnet(used_set_x.shape[1], hidden_size = 500, output_size = used_set_y.shape[1], l1reg = .002, l2reg=.0001)
	optim = optimizer('sgd', max_iter = 200, stepsize = .01, display = 5)
	model.train(used_set_x, used_set_y, optim)
	log_loss, classif_loss = model.test(test_set_x, test_set_y)
	print "Final model: log_loss = {}, classif_loss = {}\n".format(log_loss, classif_loss)

if __name__ == '__main__':
	mlp_train()
