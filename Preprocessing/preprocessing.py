import numpy as np
import scipy.sparse

def special_zero(dataset):
    ''' Returns a dataset with double the number of inputs of the original one.
    Every time an input is 0, it is transformed to [1 0].
    Every time an input is x != 0, it is transformed to [0 x].
    If a column does not contain any zero, no additional column is created.
    '''

    dataset_zero = []

    for column in dataset.T:
        if len(np.unique(column)) > 2 and np.sum(column == 0) > 0:
            dataset_zero.append(scipy.sparse.csc_matrix((column == 0).astype(float)))

    new_dataset = scipy.sparse.vstack(dataset_zero[:]).T
    return new_dataset.tocsr()

def special_nan(dataset):
    ''' Every time there is a NaN, treat it as a special value.
    '''

    dataset_nan = []

    for column in dataset.T:
        is_nan = np.isnan(np.sum(column)) # Is there at least one NaN?
        if is_nan:
            dataset_nan.append(scipy.sparse.csc_matrix(np.isnan(column).astype(float)))

    new_dataset = scipy.sparse.vstack(dataset_nan[:]).T
    return new_dataset.tocsr()

def bucketize(dataset, labels, n = 10):
    ''' If there are too many categories, bucketize them according to their average output.
    '''

    dataset_bucketized = []
    n_columns = dataset.shape[1]
    for index in range(n_columns):
        if (index+1)%np.floor(n_columns/20.0 + 1) == 0:
            print "Variable {}/{}".format(index+1, n_columns)
        variable = dataset[:, index]
        new_variable = bucketize_variable(variable, labels, n)
        if new_variable.ndim > 1: # This means we actually modified the variable
            dataset_bucketized.append(new_variable)
    new_dataset = scipy.sparse.hstack(dataset_bucketized[:])
    return new_dataset.tocsr()

def bucketize_variable(variable, labels, n = 10):
    ''' If there are too many categories, bucketize them according to their average output.'''

    n_data = len(variable) # Number of datapoints.
    n_labels = len(labels) # Number of labeled points.
    uniques = np.unique(variable[:n_labels]) # List of distinct modalities for the variable.
    n_uniques = len(uniques) # Number of distinct modalities for the variable.

    if n_uniques > 2 and n_uniques <= n:
        # In that case, bucketize for each unique.
        new_variable = np.zeros((n_data, n_uniques))
        index = 0
        for unique in uniques:
            new_variable[:, index] = (variable == unique)
            index += 1
        return scipy.sparse.csc_matrix(new_variable.astype(np.float))
    elif n_uniques > n and n_uniques <= n_data/100:
        # Compute the average output for each value, as well as the number for each value.
        avg_output = np.zeros(n_uniques)
        n_values = np.zeros(n_uniques)
        #avg_dataset = np.mean(labels)
        index = 0
        for unique in uniques:
            n_values[index] = np.sum(variable[:n_labels] == unique)
            avg_output[index] = np.mean(labels[variable[:n_labels] == unique])
            index += 1

        sorted_avg = np.argsort(avg_output) # Modalities with increasing outputs.
        new_variable = np.zeros((n_data, n)) # The transformed variable.
        n_non_nan = np.sum(n_values)

        current_sample = 0 # Current number of examples seen.
        current_column = 0
        n_modalities_seen = 0
        while current_sample < n_non_nan:
            list_modalities = [] # The list of modalities for that column.
            n_samples_seen = current_sample # Number of examples seen through the previous columns.
            # Each column will have around n_data/n elements associated to it. n_data is counted on the training set.
            list_values = []
            while current_sample < np.minimum(n_samples_seen + n_non_nan/float(n), n_non_nan):
                current_modality_index = sorted_avg[n_modalities_seen]
                list_values.append(uniques[current_modality_index])
                current_sample += n_values[current_modality_index]
                n_modalities_seen += 1
            new_variable[:, current_column] = np.in1d(variable, list_values)
            current_column += 1

        # Remove the unused columns, if any.
        new_variable = new_variable[:, :current_column]
        return scipy.sparse.csc_matrix(new_variable.astype(np.float))
    else:
        return variable