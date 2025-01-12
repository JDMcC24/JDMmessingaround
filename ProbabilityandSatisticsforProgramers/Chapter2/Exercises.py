import numpy as np
import pandas as pd



def mean(v):
    return sum(v)/len(v)

def variance(v):
    dev = []
    xbar = mean(v)
    for x in v:
        dev.append((x-xbar)**2)
    return sum(dev)/len(v)
def sd(v):
    return np.sqrt(variance(v))

def Pumpkin():
    pumpkins = [1,1,1,3,3,591]

    print("The mean is " + str(mean(pumpkins))
        + ". The variance is " + str(variance(pumpkins)) + ". The standard deviation is " + str(sd(pumpkins)) )

#Pumpkin()

NSFGdata = pd.read_csv("ProbabilityandSatisticsforProgramers/Chapter1/NSFG_2022_2023_FemRespPUFData.csv")
print(NSFGdata.describe())