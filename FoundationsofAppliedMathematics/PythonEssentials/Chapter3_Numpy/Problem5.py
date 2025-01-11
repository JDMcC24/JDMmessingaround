import numpy as np

def Prob5():
    A = np.arange(6).reshape(2,3)
    B = np.tril(3 * np.ones(3))
    C = -2 * np.eye(3)
    R1 = np.hstack( (np.zeros((3,3)), A.T, np.eye(3)))
    R2 = np.hstack((A, np.zeros((2,5))))
    R3 = np.hstack((B, np.zeros((3,2)),C))

    return np.vstack((R1,R2,R3))


print(Prob5())