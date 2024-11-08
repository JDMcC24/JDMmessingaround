import numpy as np
import matplotlib.pyplot as plt
import pandas
import sympy as sp
from sympy.plotting import plot
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
    
print(110/15)
x,y = sp.symbols('x y')

#plot3d(1- x**2 - y**2, (x,-1,1), (y,-1,1) )


e = [47.2, 48.16, 45.21, 31.24, 47.51, 46.94, 44.52, 51.65, 42.72]
b = [75, 75]

print(sum(e) + sum(b))

# plot(sp.diff( x**2 * sp.sin(x), x, 5))
# plt.show()

# eq = sp.Eq( x**4 - 10 * x**2 + 9, 0)


# f = 0
# for i in range(0,11):
#     print((1+1j)**i)

# eq2 = sp.Eq(x**2 + y**2,1)
# sp.plotting.plot_implicit(eq2)
# plt.show()

print(sp.solveset(sp.Eq(x**2 - 1000*x + 1000, 0 )))

def factor_list(n):
    fl = []
    for i in range(1,math.ceil(n/2)+1):
        if n%i == 0 :
            fl.append(i)
    if n >1:
        fl.append(n)
    return fl

for i in range(1,12):
    print([i, sum(factor_list(i))])