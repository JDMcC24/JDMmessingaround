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


#x,y,z = sp.symbols(' x y z')
starttime = time.time()

def totient(n):
    answr = 0
    for i in range(1,n):
        if math.gcd(n,i) == 1:
            answr+=1
    return answr

print(prime_factors(125))

def euler_totient(n):
    result = n
    # Check for factors from 2 to √n
    factor = 2
    while factor * factor <= n:
        # If factor divides n, it's a prime factor
        if n % factor == 0:
            # Subtract multiples of the prime factor from result
            while n % factor == 0:
                n //= factor
            result -= result // factor
        factor += 1
    # If n is a prime number greater than 1, apply totient formula to it
    if n > 1:
        result -= result // n
    return result

starttime = time.time()
print(totient(10000000), time.time()-starttime)

starttime = time.time()
print(euler_totient(10000000), time.time()-starttime)
