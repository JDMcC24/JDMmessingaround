import numpy as np

def prob4(A):
    A_copy = A
    mask = A < 0
    A_copy[mask] = 0
    return A_copy

