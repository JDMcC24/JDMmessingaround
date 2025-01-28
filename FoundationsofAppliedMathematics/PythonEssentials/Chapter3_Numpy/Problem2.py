import numpy as np

def prob2():
    """ Define the matrix A as an array. Return the matrix -A^3 + 9A^2 - 15A. """
    A = np.array([[3,1,4], [1, 5, 9],[-5,3,1]])
    C = A.dot(A)
    C = -C.dot(A) + 9* C - 15*A
    return C
