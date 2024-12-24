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

def pol(a):
    return  ((4* a**3+ 6)*(a-2))
for i in range(7):
    print(i, pol(i),  pol(i) % 7 )



print(sp.latex(sp.expand((x+1)**4)))


# A = np.array([[1,1,0,0], [0,1,1,0], [0,0,1,1],[0,0,0,1]])
# A = sp.Matrix(A)
# print(sp.latex(A.T))
# A1 = np.array([[1,1], [1,0]])
# A1 = sp.Matrix(A1)
# print(sp.latex(A1))

#e = [minsectominutes(10,21) ,minsectominutes(59,27), minsectominutes(27,47), minsectominutes(63,42),minsectominutes(32,18) ,minsectominutes(14,6), minsectominutes(34,30),minsectominutes(18,19), minsectominutes(56,26), minsectominutes(17,24), minsectominutes(33,6), minsectominutes(14,2), minsectominutes(61,23)]

e = [minsectominutes(38,17),minsectominutes(56,9), minsectominutes(16,6), minsectominutes(11,15), minsectominutes(54,29), minsectominutes(11,56)]
print(outlierearnings(e))

# for i in range(11):
#     print( i**2 %11)
# #    print (str(10) +" x " + str(i)+ " is " + str(10*i % 11))


# print(len(e))
# p = (x-2)**2 * (x-4)**3
# m = (x-2)**2 * (x-4)
# print(sp.latex(sp.expand(p)))
# print(sp.latex(sp.expand(m)))
print( 38+56+11+54+11+16)