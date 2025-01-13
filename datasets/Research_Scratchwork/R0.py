import numpy as np
from matplotlib import pyplot as plt
import random
from my_functions import *
from numpy.linalg import inv

def spradius(A):
    #Input: n x n Matrix, A, as an array
    #Output: spectral radius of A
    if np.shape(A)[0] != np.shape(A)[1]:
        return print( "Error: Input must be a square matrix")
    return np.max(np.abs(np.linalg.eig(A)[0]))

def prod(n):
    #Mulitples elemenst of a list
    product = 1
    for i in n:
        product *= i
    return product

def StochM(n):
    #Input: natrual number n
    #Outpit: random nxn column stochastic matrix
    M = np.random.rand(n,n)
    colsums = M.sum(axis=0)
    colsum = np.array([colsums]).transpose()
    A = np.tile(colsums, len(M[:,1])).reshape(n,n)
    return np.divide(M,A)


def Tmatrixmig(n):
    # Input : natural number N
    # Output: nxn positive Transition matrix representing movment between n- main loci
    M = np.random.rand(n,n)
    colsums = M.sum(axis=0)
    colsum = np.array([colsums]).transpose()
    A = np.tile(colsums, len(M[:,1])).reshape(n,n)
    while prod(prod(np.divide(M,A))) == 0 :
            M = np.random.rand(n,n)
            colsums = M.sum(axis=0)
            colsum = np.array([colsums]).transpose()
            A = np.tile(colsums, len(M[:,1])).reshape(n,n)
    return np.divide(M,A)

def Tmatrixage(n):
    # Input : natural number N
    # Output: nxn positive Transition matrix representing movment between n-main age groups
    # Note: Any age group can move into any older age group. Not a Leslie matrix. 
    
    M = np.random.rand(n,n)
    colsums = M.sum(axis=0)
    colsum = np.array([colsums]).transpose()
    A = np.tile(colsums, len(M[:,1])).reshape(n,n)
    while prod(prod(np.divide(M,A))) == 0 :
            M = np.random.rand(n,n)
            colsums = M.sum(axis=0)
            colsum = np.array([colsums]).transpose()
            A = np.tile(colsums, len(M[:,1])).reshape(n,n)
    M = np.divide(M,A)
    return np.tril(M)

def Fmatrix(n,r):
    # Input : natural number n, and postivie number r 
    # Output: nxn positive Fecunditiy matrix with maximum value r
    F = np.random.rand(1,n)
    F = r * F / np.max(F)
    M = np.zeros((n,n))
    M[0,:] = F

    return M
def GenMatrixmig(n,r):
    # Input : natural number n, and postivie number r 
    # Output: An nxn matrix A = T + F where T is a transition matrix for
    # movement between loci and F is a fecundity matrix with largest entry r.
    F = Fmatrix(n,r)
    T = Tmatrixmig(n)
    A = T + F
    return A
def GenMatrixage(n,r):
    # Input : natural number n, and postivie number r 
    # Output: An nxn matrix A = T + F where T is a transition matrix for
    # movement between age groups and F is 
    # a fecundity matrix with largest entry r.
    # Note: Any age group can move into any older age group. Not a Leslie matrix. 
    
    F = Fmatrix(n,r)
    T = Tmatrixage(n)
    A = T + F
    return A


# Experiment demonstrating R0 is always further from 1 than r is
m = 100
n= 3
f = 1
eye = np.eye(n)
rlist = []
R0list = []
ones = list(np.ones((m,1)))
for x in range(m):
     F = Fmatrix(n,f)
     T = Tmatrixage(n)
     A = F+T
     r = spradius(A)
     Nextgenmat = F @ inv((eye - T))
     R0 = spradius(Nextgenmat)
     R0list.append(R0)
     print(r)
     rlist.append(r)
     if np.abs(r-1) > np.abs(R0-1):
         print("Eureka! R0 is further from 1 than r", 'T = '+ str(T), 'F = '+ str(F), sep = '\n')
         break
plt.plot(range(m), rlist, 'b', label = 'Spectral Radius')
plt.plot(range(m), R0list, 'r', label = 'R0')
plt.plot(range(m), ones, 'g', label = 'y = 1')
plt.legend(loc = 'upper left')
plt.title('R0 is almost more extreme than r \n n = '+ str(n)+ '. Maximum Fecundity = '+ str(f)+'.')
plt.show()

