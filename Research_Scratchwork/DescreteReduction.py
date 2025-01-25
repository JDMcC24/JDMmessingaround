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
    D = np.random.rand(1,n)
    D = D/ np.max(D)
    D = r* np.diag(D[0])
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
    # print( "P matrix is", P, sep = "\n")
    # print("D matrix is ", D, sep = "\n")
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
    r0 = float(spradius(Mmatrix(P,D,0)))
    r1 = float(spradius(Mmatrix(P,D,1)))
    plt.xlabel(f't in {0,1}')
    plt.annotate(f'r(t=0) = {round(r0,3)}', (0.1,r0+.1))
    plt.annotate(f'r(t=1) = {round(r1,3)}', (0.85,r1+.1))
    print( "P matrix is", P, sep = "\n")
    print("D matrix is ", D, sep = "\n")
    return plt.plot(dom,radii), P, D

# P = np.array([[0.16531415, 0.32417886, 0.01862905, 0.23562475],
#  [0.3851423,  0.14766764, 0.26246465, 0.0016378, ],
#  [0.05894469, 0.22489492, 0.34313725, 0.28811765],
#  [0.39059886, 0.30325858, 0.37576905, 0.4746198, ]])
# D = np.array([[3.97121496,0,0,0,],
#  [0.   ,      1.60956246, 0. ,        0.  ,      ],
#  [0.  ,       0.    ,     2.03084684, 0. ,       ],
#  [0.    ,     0.    ,     0.    ,     3.97046905]])
# ReductionPlot(P,D)

# n = 2
# r = 1.01
# t = .5
# x0 = np.ones((1,n)).transpose()
# P= StochM(n)
# D = DiagM(n,r)
# M1 = Mmatrix(P,D,1)
# M2= Mmatrix(P,D,t)
# M3 = Mmatrix(P,D,0)
# x0 = np.ones((1,n)).transpose()
# x1, x2, x3 = x0, x0, x0
# M1path = x1
# M2path = x2
# M3path = x3

# for i in range(10):
#     x1 = M1 @ x1
#     M1path = np.append(M1path,x1,axis = 1) 
#     x2 = M2 @ x2
#     M2path = np.append(M2path,x2,axis = 1) 
#     x3 = M3 @ x3
#     M3path = np.append(M3path,x3,axis = 1) 
# M1xs = M1path[0,:]
# M1ys = M1path[1,:]
# M2xs = M2path[0,:]
# M2ys = M2path[1,:]
# M3xs = M3path[0,:]
# M3ys = M3path[1,:]
# lim = 1.1* max([max(M1xs),max(M2xs),max(M3xs),max(M1ys),max(M2ys),max(M3ys)])

# plt.figure(1)
# plt.xlim((-.1,1.1*lim))
# plt.ylim((-.1,1.1* lim))
# plt.scatter(M1xs,M1ys, label  = f"Pure Dispersal. r = {round(spradius(M1),4)}")
# plt.scatter(M2xs,M2ys, label  = f" {round(100* t)}  percent Dispersal. r = {round(spradius(M2),4)}")
# plt.scatter(M3xs,M3ys, label = f'No Dispersal. r = {round(spradius(M3),4)}')

# for i in range(len(M1xs)):
#     plt.annotate( f'{i}', (M1xs[i]+.005, M1ys[i]+.005))
#     plt.annotate( f'{i}', (M2xs[i]+.005, M2ys[i]+.005), )
#     plt.annotate( f'{i}', (M3xs[i]+.005, M3ys[i]+.005), )

# plt.legend(loc = 'upper right')

# plt.figure(2)
# ReductionPlot(P,D)

# plt.show()


n = 10
D = DiagM(n,2)
P = StochM(n)
# P =np.diag([.5,.5,.5], 1) + np.diag([.5,.5,.5], -1)
# P[0,0]=1
# P[1,0]=0
# P[3,3]=.5
# print(P)
ax1, P, D = ReductionPlot(P,D)
print( "P matrix is", P, sep = "\n")
print("D matrix is ", D, sep = "\n")
plt.show()