import gzip
import sys
import getopt

from Core.baseFunctions import *


eps = .000001 # To avoid log(0)

def expand(labels):
	''' Takes a vector of labels and transforms it into a matrix of one-hot encodings '''
	label_matrix = np.zeros((labels.size, max(labels)+1))
	for line in xrange(labels.shape[0]):
		label_matrix[line][labels[line]] = 1
	return label_matrix

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
	def __init__(self, input_size, hidden_size = 500, output_size = 1, l1reg=1.0, l2reg=.0001, start_dynamic = 0):
		self.input_size = input_size # Size of the input
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.input_weights = np.random.randn(hidden_size, input_size)/np.sqrt(input_size) # Weights of the model
		self.output_weights = np.zeros((hidden_size, output_size)) # Weights of the model
		self.input_biases = np.zeros(hidden_size)
		self.output_biases = np.zeros(output_size)
		self.l1reg = max(0.0, l1reg)
		self.l2reg = max(0.0, l2reg)
		self.start_dynamic = start_dynamic

	def predict(self, data):
		n_data = data.shape[0]
		""" Predicts the output for any number of data """
		hidden = pos(np.dot(data, self.input_weights.T) + np.tile(self.input_biases, (n_data, 1)))
		return softmax(np.dot(hidden, self.output_weights) + np.tile(self.output_biases, (n_data, 1))), hidden # The softmax is the multidimensional equivalent of the sigmoid
	
	def get_gradient(self, data, labels):
		n_data = data.shape[0]
		output, hidden = self.predict(data)
		grad_output = output - labels
		grad_output_weights = np.dot(hidden.T, grad_output)/n_data
		grad_output_biases = np.mean(grad_output, 0)
		grad_hidden_after = np.dot(grad_output, self.output_weights.T)
		grad_hidden_before = grad_hidden_after * (hidden > 0)
		grad_input_weights = np.dot(grad_hidden_before.T, data)
		grad_input_biases = np.mean(grad_hidden_before, 0)
		return grad_output, grad_input_weights, grad_output_weights, grad_input_biases, grad_output_biases, output

	def add_neuron(self, data, labels, output, stepsize):
		""" Do a small optimization of the input weights added """
		# Randomly initialize the new neuron
		input_weight = np.random.randn(self.input_size)
		input_bias = 0
		output_weight = np.zeros(self.output_size)
		for inside_iter in xrange(5):
			hidden = pos(np.dot(data, input_weight) + input_bias)
			new_output = output + np.outer(hidden, output_weight)
			grad_output = new_output - labels
			grad_output_weight = np.dot(grad_output.T, hidden)
			grad_hidden_after = np.dot(grad_output, output_weight)
			grad_hidden_before = grad_hidden_after * (hidden > 0)
			grad_input_weight = np.dot(grad_hidden_before.T, data)
			grad_input_bias = np.mean(grad_hidden_before, 0)
			input_weight -= stepsize * grad_input_weight
			output_weight -= stepsize * grad_output_weight
			input_bias -= stepsize * grad_input_bias
		return input_weight, input_bias


	def train(self, data, labels, optimizer):
		n_data = data.shape[0]
		# Compute the average of the dataset.
		avg_data = np.mean(data, 0)
		stepsize = optimizer.stepsize
		l1reg = self.l1reg
		l2reg = self.l2reg
		max_iter = optimizer.max_iter
		start_dynamic = self.start_dynamic
		minibatch = 50
		count_zero = np.zeros(self.hidden_size)
		count_zero_max = 100

		if optimizer.method == 'sgd':
			for iter in xrange(1, max_iter):
				for i in xrange(n_data/minibatch):
					datum = data[i*minibatch:i*minibatch + minibatch]
					label = labels[i*minibatch:i*minibatch + minibatch]
					grad_output, grad_input_weights, grad_output_weights, grad_input_biases, grad_output_biases, output = self.get_gradient(datum, label)
					self.input_weights -= stepsize * grad_input_weights
					self.output_weights -= stepsize * grad_output_weights
					self.input_biases -= stepsize * grad_input_biases
					self.output_biases -= stepsize * grad_output_biases


					if iter >= start_dynamic:
						do_add = np.random.rand() < .01
						if do_add == 1 and self.input_biases.shape[0] < 700:
							count_zero = np.append(count_zero, 0)
		
							# Append the datapoint itself at the end of the input weights.
							new_weight, new_bias = np.random.randn(self.input_size)/np.sqrt(self.input_size), 0
							new_weight, new_bias = self.add_neuron(datum, label, output, stepsize)

							self.input_weights = np.row_stack((self.input_weights, new_weight))
							self.input_biases = np.append(self.input_biases, new_bias)
		
							# Do a gradient step for the output weights.
							hidden = pos(np.dot(datum, new_weight) + new_bias)
							self.output_weights = np.row_stack((self.output_weights, -stepsize*np.dot(hidden, grad_output/minibatch)))
	
						# Now do an iteration of group L1 of the output weights.
						norms = np.sqrt(np.sum(self.output_weights**2, 1))
	
						# We only keep the neurons whose output weights are not all 0.

						nonzero = np.nonzero(norms > stepsize*l1reg)[0]
						count_zero += 1
						count_zero[nonzero] = 0
						to_be_kept = np.nonzero(count_zero != count_zero_max)[0]
						
						if len(to_be_kept) < self.input_biases.shape[0]:
							count_zero = count_zero[to_be_kept]
							self.output_weights = np.take(self.output_weights, to_be_kept, axis=0)
							self.input_weights = np.take(self.input_weights, to_be_kept, axis=0)
							self.input_biases = self.input_biases[to_be_kept]
	
						# Project the remaining weights
						self.output_weights *= np.maximum(0, 1 - stepsize*l1reg/(norms[to_be_kept]+eps)).reshape((-1, 1))
						
					self.input_weights *= 1 - stepsize*l2reg

				if optimizer.display > 0 and iter%optimizer.display == 0:
					log_loss, classif_loss = self.test(data, labels)
					print "Iteration {}: {} units, log_loss = {}, classif_loss = {}".format(iter, self.input_biases.shape[0], log_loss, classif_loss)
	
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

def mlp_train(argv):
	try:
		opts, args = getopt.getopt(argv, "s:h:l:m:d:")
	except getopt.GetoptError:
		print 'dynamic.py -s <train_set> -h <hidden_size> -l <l1penalty> -m <max_iter> -d <dynamic>'
		sys.exit(2)

	rval = load_data('/home/nicolas/data/mnist.pkl.gz')
	train_set_x, train_set_y = rval[0]
	valid_set_x, valid_set_y = rval[1]
	test_set_x, test_set_y = rval[2]
	train_set_y = expand(train_set_y)
	valid_set_y = expand(valid_set_y)
	test_set_y = expand(test_set_y)

	for opt, arg in opts:
		if opt == '-s':
			if arg == 'train':
				used_set_x = train_set_x # The set actually used for training
				used_set_y = train_set_y
			elif arg == 'valid':
				used_set_x = valid_set_x # The set actually used for training
				used_set_y = valid_set_y
			else:
				print 'Wrong argument'
				sys.exit()
		elif opt == '-h':
			hidden_size = int(arg)
		elif opt == '-l':
			l1reg = float(arg)
		elif opt == '-m':
			max_iter = int(arg)
		elif opt == '-d':
			start_dynamic = int(arg)
				
	max_norm = max(np.sum(used_set_x**2, 1))

	model = nnet(used_set_x.shape[1], hidden_size = hidden_size, output_size = used_set_y.shape[1], l1reg = l1reg, l2reg=.0001, start_dynamic = start_dynamic)
	optim = optimizer('sgd', max_iter = max_iter, stepsize = .01, display = 5)
	model.train(used_set_x, used_set_y, optim)
	log_loss, classif_loss = model.test(test_set_x, test_set_y)
	print "Final model: log_loss = {}, classif_loss = {}\n".format(log_loss, classif_loss)

if __name__ == '__main__':
	mlp_train(sys.argv[1:])
