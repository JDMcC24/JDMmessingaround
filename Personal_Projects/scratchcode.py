import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import sympy as sp
from sympy.plotting import plot
from sympy.plotting import plot3d
import scipy
from my_functions import *
#import sklearn
import time


def minsectominutes(m,s):
    return m + s/60


x = sp.Symbol('x')

def poly1(x):
    return (x-1)**2 *(x+1)**2 *(x+3)

#print(poly1(x))



"""Math Bio Problem(s)"""

## x(t+1) = MPx(t)

x, y, z = sp.symbols('x y z')
M = np.array([ [.75 , .5], [0.25,.5]])
r1 = .5
r2 = 1.2
#D = np.array([[r1*x*(1-x)], [r2*y*(1-y)], [r1 *y*(1-y) ]])
D0 = np.array([[r1, 0],[0, r2]])
#M = sp.Matrix(M)
print(sp.latex(sp.Matrix(M)))
#print(M@D0)
#print((D0))
#print(sp.latex(M))


def find_lcm(a, b):
    # Calculate the Greatest Common Divisor (GCD) using math.gcd()
    gcd = math.gcd(a, b)
    
    # Calculate the Least Common Multiple (LCM)
    lcm = abs(a * b) // gcd
    
    return lcm
# print(find_lcm(439582,97850))

e = [minsectominutes(2,21),minsectominutes(18,13),minsectominutes(20,58),minsectominutes(6,33), minsectominutes(24,7)]

# """" Limits """
# def seqfun(x):
#     return .5*x*(1-x)
# a = 2/3 
# for i in range(0,10): 
#     a = seqfun(a)
#     print(a)pip in

f = 1/(x**2 * sp.log(x))
#result = sp.integrate(f,(x,1,sp.oo))
#print(result)

# dom = np.linspace(0,10)
# sp.plotting.plot(f,(x,1,10))
# sp.plotting(plot(f,(x,0,1)))
# plt.show()
print(outlierearnings(e))

