import csv
import numpy as np
import scipy.sparse


class DataAccess:
    # Class which sends data to an optimizer. Data can be stored in RAM or come from a file.
    def __init__(self, data, labels, max_data, read_type, sparse_data=None):
        self.data = data  # The path to the data or the numpy array itself
        self.labels = labels  # The path to the labels or the numpy array itself
        self.read_type = read_type  # Sequential retrieval of data
        self.max_data = max_data  # The number of datapoints to get.

        if sparse_data is not None:
            self.sparse_data = sparse_data
            self.is_sparse_data = 1
        else:
            self.is_sparse_data = 0
        
        # If "data" is a filename, check once and for all that it exists.
        if str(data):
            self.data_type = "file"
            try:
                self.data_reader = csv.reader(open(data, "rb"))
                self.labels_reader = csv.reader(open(labels, "rb"))
            except IOError:
                print("The data file does not exist.")
        else:
            self.data_type = "var"
            
        if self.read_type == "seq":
            self.current_batch = 0  # The position of the current datapoint in the file.
        
    def get(self, minibatch):    # Get a new minibatch of data
        data = self.data
        labels = self.labels
        max_data = self.max_data

        if self.data_type == "var":  # The dataset is stored in memory.
            n_data = data.shape[0]
            
            if self.read_type == "seq":  # If we get the data sequentially, we need to make sure we don't get too far.
                current_batch = self.current_batch
                limit = np.min((max_data, n_data))
                end_data = current_batch*minibatch + minibatch        
                if end_data < limit:
                    datum = data[current_batch*minibatch:end_data] # Extract the datapoints.
                    label = labels[current_batch*minibatch:end_data] # Extract the labels.

                    if self.is_sparse_data:
                        sparse_datum = np.array(self.sparse_data[current_batch*minibatch:end_data].todense())
                        datum = np.hstack((datum, sparse_datum))

                    self.current_batch += 1
                else:
                    datum = data[current_batch:limit] # Extract the datapoints.
                    label = labels[current_batch:limit] # Extract the labels.
                    if self.is_sparse_data:
                        sparse_datum = np.array(self.sparse_data[current_batch:limit].todense())
                        datum = np.hstack((datum, sparse_datum))

                    self.current_batch = 0  # We start from the beginning again.

            elif self.read_type == "random":
                n_batches = int(n_data/minibatch)  # The number of batches in the dataset.
                batch = np.random.randint(n_batches)
                datum = data[batch*minibatch:batch*minibatch + minibatch]  # Extract the datapoints.
                label = labels[batch*minibatch:batch*minibatch + minibatch]
                if self.is_sparse_data:
                    sparse_datum = np.array(self.sparse_data[batch*minibatch:batch*minibatch + minibatch].todense())
                    datum = np.hstack((datum, sparse_datum))
                self.current_batch = batch
            else:
                print("Unknown data_getter type.")
        else:  # This must be a filename
            datum = np.zeros()
            label = np.zeros()
            # Do nothing for now.
            print("Not yet implemented.")
            
        return datum, label