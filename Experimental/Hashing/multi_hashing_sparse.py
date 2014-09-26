import numpy as np
import matplotlib.pyplot as plt
import pysparse as ps
from pysparse.eig import jdsym

# Choose the number and the size of the datapoints.
n = 2000
p = 200
q = 100
n_max = 1000
avg_modality = 10

X  = ps.spmatrix.ll_mat(n, p, avg_modality)
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

	H_1 = ps.spmatrix.ll_mat(p, q, 1)
	H_2 = ps.spmatrix.ll_mat(p, q, 1)

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
	X_1 = ps.spmatrix.matrixmultiply(X, H_1)
	X_2 = ps.spmatrix.matrixmultiply(X, H_2)

	# Compute the rank of each matrix. 
	
	dummy_1, ev_1 = jdsym.jdsym(X_1, None)
	dummy_2, ev_2 = jdsym.jdsym(X_2, None)
	sp_1[n_tests] = np.sum(ev_1.astype(int))
	sp_1[n_tests] = np.sum(ev_2.astype(int))


count_1 = np.bincount(sp_1.astype(int)).astype(float)/n_max
count_2 = np.bincount(sp_2.astype(int)).astype(float)/n_max
print count_1
print count_2

plt.bar(range(0, np.size(count_1)), count_1, width=.2, color='b')
plt.bar(np.arange(0, np.size(count_2))+.2, count_2, width=.2, color='r')
plt.show()
