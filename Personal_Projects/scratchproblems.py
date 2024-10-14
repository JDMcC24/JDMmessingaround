import numpy as np
import matplotlib.pyplot as plt
import pandas
from my_functions import *


def CzPlot(n):
    xs = []
    ys = []
    
    for i in range(2,n,1):
        xs.append(i)
        ys.append(Czlist(i)[1])
    plt.scatter(xs,ys,s=5)
    plt.xlabel("Number")
    plt.ylabel("Number of Steps Until 1")
    plt.title( "Collatz Sequences" )
    plt.show()

Gblist(494)
