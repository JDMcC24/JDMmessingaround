import numpy as np
from matplotlib import pyplot as plt
import random
from my_functions import *

def spradius(A):
    #Input: n x n Matrix, A, as an array
    #Output: spectral radius of A
    if np.shape(A)[0] != np.shape(A)[1]:
        return print( "Error: Input must be a square matrix")
    return max(np.linalg.eig(A)[0])

def StochM(n):
    #Input: natrual number n
    #Outpit: random positive nxn column stochastic matrix
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

def prod(n):
    #Mulitples elemenst of a list
    product = 1
    for i in n:
        product *= i
    return product

def DiagM(n,r):
    #Input: natrual number n
    #Output: Diagonal nxn matrix with postiive diagonal values between 0 and r
    if r <= 0:
        return print("Error! The second argurment must be positive.")
    D = r* np.diag(np.random.rand(1,n)[0])
    while prod(np.diag(D)) == 0:
        D = r* np.diag(np.random.rand(1,n)[0])
    return D

def Mmatrix(P,D,t):
    return (1-t)*D + t * P @ D


def RadnomReductionPlot(n,r):
    P = StochM(n)
    D = DiagM(n,r)
    dom = np.linspace(0,1)
    radii = []
    for x in dom:
        M = Mmatrix(P,D,x)
        radii.append(spradius(M))
    plt.plot(dom,radii)
    plt.ylim((0,r))
    plt.xlim((0,1))
    plt.title("Reduction Phenomon for a Random "+str(n)+' x ' + str(n)+ ' Matrix' )
    print( "P matrix is", P, sep = "\n")
    print("D matrix is ", D, sep = "\n")
    plt.show()


def ReductionPlot(P,D):
    dom = np.linspace(0,1)
    radii = []
    for x in dom:
        M = Mmatrix(P,D,x)
        radii.append(spradius(M))
    plt.plot(dom,radii)
    plt.ylim((0,np.ceil(1.5*np.max(D))))
    plt.xlim((0,1))
    plt.title("Reduction Phenomon for given Matrices")
    print( "P matrix is", P, sep = "\n")
    print("D matrix is ", D, sep = "\n")
    plt.show()

P = np.array([[0.16531415, 0.32417886, 0.01862905, 0.23562475],
 [0.3851423,  0.14766764, 0.26246465, 0.0016378, ],
 [0.05894469, 0.22489492, 0.34313725, 0.28811765],
 [0.39059886, 0.30325858, 0.37576905, 0.4746198, ]])
D = np.array([[3.97121496,0,0,0,],
 [0.   ,      1.60956246, 0. ,        0.  ,      ],
 [0.  ,       0.    ,     2.03084684, 0. ,       ],
 [0.    ,     0.    ,     0.    ,     3.97046905]])
ReductionPlot(P,D)