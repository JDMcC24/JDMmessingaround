import numpy as np


def prob6(A):
    s = A.sum(axis =1)
    s = np.array([s]).transpose()
    M = np.tile(s, len(A[1,:]))
    M = np.divide(A,M)
    return M

print(prob6(A))
