import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import *
from scipy.sparse.linalg import svds

# Choose the number and the size of the datapoints.
n = 2000
p = 400
q = 100
n_max = 100
avg_modality = 5

X  = lil_matrix((n, p))
sp_1 = np.zeros(n_max)
sp_2 = np.zeros(n_max)

# What is the probability of accessing each of the p elements.
prob_p = np.ones(p) * 1/p
prob_q = np.ones(q) * 1/q

# Create the original data matrix
for i in xrange(0, n):
	# For each datapoint, draw the number of modalities equal to 1 from a Poisson distribution.
	n_modalities = 1 + np.minimum(np.random.poisson(avg_modality - 1), p - 1)
	for modality in xrange(0, n_modalities):
		mass = np.random.uniform()
		index = 0
		while mass > 0:
			mass -= prob_p[index]
			index += 1
		X[i, index - 1] = 1

for n_tests in xrange(0, n_max):

	H_1 = lil_matrix((p, q))
	H_2 = lil_matrix((p, q))

	# Now create the two hashing dictionaries
	for i in xrange(0, p):
		mass = np.random.uniform()
		index = 0
		while mass > 0:
			mass -= prob_q[index]
			index += 1
		H_1[i, index - 1] = 1
	
	for i in xrange(0, p):
		mass = np.random.uniform()
		index = 0
		while mass > 0:
			mass -= prob_q[index]
			index += 1
		H_2[i, index - 1] = 1


	# Create the hashed data matrices
	X = X.tocsr()
	H_1 = H_1.tocsr()
	H_2 = H_2.tocsr()
	X_1 = X * H_1
	X_2 = X * (H_1 + H_2)

	#k_1 = svds(X_1, return_singular_vectors=False)
	#k_2 = svds(X_2, return_singular_vectors=False)
	#sp_1[n_tests] = np.sum(k_1 > .5)
	#sp_2[n_tests] = np.sum(k_2 > .5)
	sp_1[n_tests] = np.linalg.matrix_rank(X_1.todense())
	sp_2[n_tests] = np.linalg.matrix_rank(X_2.todense())

count_1 = np.bincount(sp_1.astype(int))
count_2 = np.bincount(sp_2.astype(int))
print count_1
print count_2

plt.bar(range(0, np.size(count_1)), count_1, width=.2, color='b')
plt.bar(np.arange(0, np.size(count_2))+.2, count_2, width=.2, color='r')
plt.show()
