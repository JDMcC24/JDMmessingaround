import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.plotting import plot3d
from my_functions import *




# e = np.array([[36.25, 11.23, 20.32, 33.33, 12.39, 14.50,26.91, 16.46, 30.82, 7.95, 21.51],[39, 17, 30, 50, 36, 21,40, 24,46, 12, 32]])

# print(np.sum(e, axis = 1), e.shape )


# Bill = 0.05 + 0.5/20 +0.5/10 + 1/2.5
# Bob = .75/8 + .5/16 + .5/8 + 1/2
# print(Bill, Bob)

# seq = [3]
# for i in range(0,7):
#     seq.append((seq[i])**2 - seq[i] )

# print(seq)
# x = sp.Symbol('x')
# eq = sp.Eq( x/15, (x-.25)/5)
# print(sp.solveset(eq))
    
#print(110/15)

#plot3d(1- x**2 + y**2, (x,-1,1), (y,-1,1) )
e = [37.80, 43.5, 38.83,41.31, 48.40, 47.59, 53.55, 53.78]
print(sum(e) + 180+150)
