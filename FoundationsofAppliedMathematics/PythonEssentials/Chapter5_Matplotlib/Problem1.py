from matplotlib import pyplot as plt
import numpy as np

def prob1(n):
    M = np.random.normal(size=(n,n))
    means = np.mean(M, axis= 1)



    return np.var(means)

print(prob1(2))