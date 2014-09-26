import gzip

from Optimizer.Optimizer import *
from Optimizer.Solver import *
from Core.baseFunctions import *
from Models.Nnet import *
from Models.PairwiseNNet import *


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

def train():
    rval = load_data('e:/Data/MNist/mnist.pkl.gz')
    train_set_x, train_set_y = rval[0]
    valid_set_x, valid_set_y = rval[1]
    test_set_x, test_set_y = rval[2]
    train_set_y = expand(train_set_y)
    valid_set_y = expand(valid_set_y)
    test_set_y = expand(test_set_y)
    used_set_x = train_set_x # The set actually used for training
    used_set_y = train_set_y

    # Create the unitary model.
    model = Nnet(input_size = used_set_x.shape[1], output_size = used_set_y.shape[1], activation = ['softmax'], hidden_sizes = [])
    solver = Solver("logistic", model, l1_regularizer = 0, l2_regularizer = 0.0001)
    optimizer = Optimizer(solver, method = 'sag', read_type = "random", stepsize = -1.0, max_updates = 50000, minibatch = 50, display = 500, max_data = sys.maxint)
    optimizer.train(used_set_x, used_set_y)

    # Create the pairwise dataset.
    randomOrder = np.random.permutation(used_set_x.shape[0])
    xRoll = used_set_x[randomOrder]
    yRoll = used_set_y[randomOrder]
    #pairData = np.hstack((used_set_x, xRoll, np.abs(used_set_x - xRoll)))
    pairData = np.abs(used_set_x - xRoll)
    pairLabel = np.sum(used_set_y * yRoll, axis = 1) # Ugly but it works.

    modelPair = Nnet(input_size = pairData.shape[1], output_size = 1, activation = ['sigm'], hidden_sizes = [])
    solverPair = Solver("logistic", modelPair, l1_regularizer = 0, l2_regularizer = 0.0001)
    optimizerPair = Optimizer(solverPair, method = 'sag', read_type = "random", stepsize = -1.0, max_updates = 50000, minibatch = 50, display = 500, max_data = sys.maxint)
    optimizerPair.train(pairData, pairLabel)

    # Predict on the train set.
    model.predict_batch(used_set_x)
    training_nll = np.mean(log_loss(model.output, used_set_y))
    training_classif_error = np.mean(classif_loss(model.output, used_set_y))
    print "Single train NLL = {}, Single train classif error = {}".format(training_nll,training_classif_error)

    model.predict_batch(valid_set_x)
    valid_nll = np.mean(log_loss(model.output, valid_set_y))
    valid_classif_error = np.mean(classif_loss(model.output, valid_set_y))
    print "Single valid NLL = {}, Single valid classif error = {}".format(valid_nll,valid_classif_error)

    # Create the joint model.
    pairwiseNNet = PairwiseNNet(model, modelPair)

    # Predict on the train set.
    pairwiseNNet.predict_batch(used_set_x)
    training_nll = np.mean(log_loss(pairwiseNNet.output, used_set_y))
    training_classif_error = np.mean(classif_loss(pairwiseNNet.output, used_set_y))
    print "Pair train NLL = {}, Pair train classif error = {}".format(training_nll,training_classif_error)

    pairwiseNNet.predict_batch(valid_set_x)
    valid_nll = np.mean(log_loss(pairwiseNNet.output, valid_set_y))
    valid_classif_error = np.mean(classif_loss(pairwiseNNet.output, valid_set_y))
    print "Pair valid NLL = {}, Pair valid classif error = {}".format(valid_nll,valid_classif_error)

if __name__ == '__main__':
    train()
