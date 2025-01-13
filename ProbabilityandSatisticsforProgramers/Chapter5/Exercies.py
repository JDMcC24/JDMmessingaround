import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
import itertools 

""" Started going down a rabit hole here just playing around.
Serious attempts to do the exercies in the book are forth coming"""



def diceroll():
    return random.randint(1,6)

def twodiceexperiment(n):
    count8 = 0
    score = 0
    for i in range(n):
        a = diceroll()
        b = diceroll()
        if a+b == 8:
            count8+=1
            if a==6 or b==6:
                score+=1

    return  score/count8



def multiexperiment(n,m):
    record = []
    for i in range(m):
        record.append(twodiceexperiment(n))
    return record


# data = multiexperiment(10000,100)
# bins = np.linspace(0,1,100)
# plt.xlim((0,1))
# plt.hist(data,bins)
# plt.show()

def bernoullidist(p):
    roll = random.random()
    if roll < p:
        return 1
    else:
        return 0



def binomdist(n,p):
    record = 0 
    for i in range(n):
        record += bernoullidist(p)
    return record

def binomdata(n,p,m):
    record = []
    for i in range(m):
        record.append(binomdist(n,p))
    return record
# n = 20
# p = .5
# data = binomdata(n,p,1000)
# bins = np.linspace(0,n,n)
# plt.hist(data,bins)
# plt.show()

#Exercise 1.
dice = [1,2,3,4,5,6]
rolls = itertools.product(dice,repeat=2)
count8 = 0
count6 = 0
for roll in rolls:
    if sum(roll) == 8:
        count8+=1
        if roll[0]==6 or roll[1]==6:
            count6+=1
#print(count6/count8)
    
#Exercise2
#print(1/ 6**100)

#Exercise 3
#These get repeatitive.... and I've done most of these in high school...
sexes = ["b", "g"]
children = itertools.product(sexes, repeat =2)
count = 0
for siblings in children:
    if siblings[0]=='g' or siblings[1]=='g':
        count+=1
#print(1/count)

#Exercise 4 Monty Hall Problem
def MontyHallEstimate(n):
    Doors = {"A", "B", "C"}
    stayer = 0
    changer = 0
    for i in range(n):
        initialpick = random.choice(list(Doors))
        winner = "A"

        opendoor = {random.choice(list({"B","C"}.difference({initialpick})))}
        NewDoors = Doors.difference(opendoor)
        secondpick = NewDoors.difference(initialpick)
        if initialpick ==winner:
            stayer +=1
        if winner in secondpick:
            changer+=1

    return 'Changer won '+ str(100* changer/n)+ "percent of the time. Stayer won "+ str(100* stayer/n ) +" percent of the time."

print(MontyHallEstimate(100000))





