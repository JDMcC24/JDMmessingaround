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


x,y,z = sp.symbols(' x y z')

M = np.array([[1,4,-1],[6,1,3],[-2,5,7]])
N = np.array([[3,0,2],[-1,4,1],[5,-2,3]])
# print(np.linalg.det(N))
P =  M* - 5*M.dot(N) + 3*N.transpose()
# print(np.linalg.det(P))
# print(scipy.linalg.det(P))
# pol1 = 5 + 19 * x - 9 * x**2 + x**3
# pol2 = (x+2)*(x-7)*(x-4)
# print(sp.latex(sp.expand(pol2)))
# print(sp.latex(pol1))


A = np.array([[1,1],[0,1]])
A = sp.Matrix(A)
B = np.array([[1,0,0],[1,1,0],[1,1,1]])
B = sp.Matrix(B)
C = np.array([[-1,5,6],[1,5,9],[1,0,-2]])
C = sp.Matrix(C)

# print(sp.latex(sp.Matrix(A)))
# # print((A**2))
# # print(A**3)
# #print(scipy.linalg.expm(A))
# #print(math.e**3)
# M = sp.Matrix(scipy.linalg.expm(A))
# P, J = M.jordan_form()
# print(sp.latex(J))

# a = 6
# print((a**3 + 5*a**2 + 2* a + 1)%7) 
# print(216 + 180 + 12 + 1)
# print(sp.latex(A))
# print(sp.latex(B))
# print(sp.latex(C))
 

# print(sp.latex(sp.Matrix(M)))
# print( sp.latex(sp.Matrix(N)))
# print( sp.latex(sp.Matrix(P)))
# print(sp.det(M))


def minsectominutes(m,s):
    return m + s/60

# B = A.T
# C = -A*B
# print(sp.latex(C))
# print(sp.latex(C.jordan_form()))

Z = 5

def pol(a):
    return  ((Z+3)* a**3 + (Z+2)*a**2 -(Z+1)* a + 1)
# for i in range(Z):
#     print(i, pol(i),  pol(i) % Z )


def pol2(a):
    return  ((a+1)* (a-1)**4)
for i in range(Z):
    print(i, pol2(i),  pol2(i) % Z )
print( sp.latex(sp.expand(pol2(x))))


# A = np.array([[1,0,0,0], [0,2,1,0], [0,0,2,1],[0,0,0,2]])
# A = sp.Matrix(A)
# print(sp.latex(A.T))
# B = np.array([[1,0,0], [0,2,1], [0,0,2],])
# B = sp.Matrix(B)
# print(sp.latex(B.T))
# A1 = np.array([[1,1], [1,0]])
# A1 = sp.Matrix(A1)
# print(sp.latex(A1))


e = [18.75, 48.25,16]
print(outlierearnings(e))



# px = (x+1)**2 * (x+2)**2 * (x+1)
# mx = (x-1) *(x-3)*(x-2)**3
# print(sp.latex(sp.expand(px)))


# A = np.array([[1,1],[0,1]])
# A = sp.Matrix(A)
# E = np.zeros([2,2])
# E = sp.Matrix(E)
# for i in range (0,20):
#     E+= A**i / math.factorial(i)
# print(E)
# print(sp.latex(A.T))

