import numpy as np
import matplotlib.pyplot as plt
import pandas
import sympy as sp
from my_functions import *




e = np.array([[25.03, 33.33, 1.73, 31.20, 33.33, 2.2],[37, 50, 5, 46, 50, 6]])

print(np.sum(e, axis = 1)[1] /60 )


# Bill = 0.05 + 0.5/20 +0.5/10 + 1/2.5
# Bob = .75/8 + .5/16 + .5/8 + 1/2
# print(Bill, Bob)
