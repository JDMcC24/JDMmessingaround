import numpy as np
import matplotlib.pyplot as plt
import pandas
import sympy as sp
from my_functions import *






e = np.array([[29.21,25.72, 26.83, 33.33, 8.26, 20.69, 33.33, 6.28, 18,30 ],[43,38, 40, 50, 24,30, 50, 18, 27,46]])
# print( np.sum(e, axis=1))
# print(prime_list(10))

x = sp.Symbol("x") 
t1 = sp.Eq(15*x, 8*x + 1/50 - 1/100)
s = sp.solveset(t1,x)
# print(s)
# print(list(s)[0] * 60)
# print(8/60 * 2.4)
print(np.sum(e,axis = 1))
