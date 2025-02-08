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

# def poly1(x):
#     return (x**2+1) *(x+1)**2

# print(sp.latex(sp.expand(poly1(x))))

Z = 11

def poly2(x):
    return (x**3 + 5*x + 1)#*(x+2)**3

for i in range(Z):
    print(i, poly2(i), poly2(i) % Z)
print(sp.latex(sp.expand(poly2(x))))
print(sp.latex(sp.expand((x**2 + 1)*(x**2-x+1))))

# D = [1,1,-1,-1]
# J = np.diag(D)
# print(sp.latex(sp.Matrix(J)))


"""Math Bio Problem(s)"""

## x(t+1) = MPx(t)

# x, y, z = sp.symbols('x y z')
# M = np.array([ [.25 , .75], [.75,.25]])
# r1 = .6
# r2 = 1.2
#D = np.array([[r1*x*(1-x)], [r2*y*(1-y)], [r1 *y*(1-y) ]])
# D0 = np.array([[r1, 0],[0, r2]])
#M = sp.Matrix(M)
# print(sp.latex(sp.Matrix(M@D0)))
# print(np.linalg.eigvals(M@D0))
#print(M@D0)
#print((D0))
#print(sp.latex(M))


# def find_lcm(a, b):
#     # Calculate the Greatest Common Divisor (GCD) using math.gcd()
#     gcd = math.gcd(a, b)
    
#     # Calculate the Least Common Multiple (LCM)
#     lcm = abs(a * b) // gcd
    
#     return lcm
# # print(find_lcm(439582,97850))

# e = [minsectominutes(2,21),minsectominutes(18,13),minsectominutes(20,58),minsectominutes(6,33), minsectominutes(24,7)]

# # """" Limits """
# # def seqfun(x):
# #     return .5*x*(1-x)
# # a = 2/3 
# # for i in range(0,10): 
# #     a = seqfun(a)
# #     print(a)pip in

# f = 1/(x**2 * sp.log(x))
# #result = sp.integrate(f,(x,1,sp.oo))
# #print(result)

# # dom = np.linspace(0,10)
# # sp.plotting.plot(f,(x,1,10))
# # sp.plotting(plot(f,(x,0,1)))
# # plt.show()
# print(outlierearnings(e))



e = [minsectominutes(27,52), minsectominutes(40,46), minsectominutes(42,24), minsectominutes(47,50), minsectominutes(38,46), minsectominutes(34,4), minsectominutes(22,48),minsectominutes(25,31)
     , minsectominutes(29,7)] 
print(outlierearnings(e))

# def sec_to_time(sec):
#     hours = sec // 3600
#     remaining = sec % 3600
#     minutes = remaining // 60
#     seconds = remaining %60
#     return f' Total time is {hours} hours, {minutes} minutes, and {seconds}, seconds.'

# print(sec_to_time(2098.0845415592194))



from datetime import datetime

def timestamp_to_float(timestamp):
    # Convert the timestamp to a float representing seconds since the epoch
    timestamp_float = timestamp.timestamp()
    return timestamp_float

# Example usage
timestamp = datetime(2025, 2, 6, 20, 17)
timestamp_float = timestamp_to_float(timestamp)
print(f"The timestamp as a float is: {timestamp_float}")

