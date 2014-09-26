import numpy as np
import copy
#import matplotlib.pyplot as plt
iter = 100
p = 20
A = np.random.randn(p, p)
H = np.dot(A.T, A)
H = H + H.T
v = np.linalg.eigvals(H)
L = np.max(v)
m = np.min(v)
print(L, m, m/L)
theta_f = np.random.randn(p)
theta_n = copy.deepcopy(theta_f)
errs = np.zeros((2, iter))
for i in range(iter):
    grad_f = H.dot(theta_f)
    grad_n = H.dot(theta_n)
    theta_f -= grad_f/L
    theta_n -= grad_n * np.dot(grad_n, grad_n) / np.dot(grad_n, H.dot(grad_n))
    errs[0, i] = np.log(np.dot(theta_f, H.dot(theta_f)))
    errs[1, i] = np.log(np.dot(theta_n, H.dot(theta_n)))
    print (errs[:, i])
#plt.plot(range(iter), errs[0], 'r--', range(iter), errs[1], 'bs')