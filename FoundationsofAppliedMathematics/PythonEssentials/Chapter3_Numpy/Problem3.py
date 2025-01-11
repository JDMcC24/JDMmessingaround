import numpy as np


def prob3():
    A = np.ones((7,7))
    B = -1 * np.tril(A) + 5* np.triu(A) -5* np.diag(np.diag(A))
    A = np.triu(A)
    C = A.dot(B.dot(A))
    return C.astype(np.int64)