import time
from my_functions import *
import itertools




starttime = time.time()
limit = 10**7
minratio = 10
for n in range(1,limit+1):
    m = totient(n)
    if m in digit_permutations(n):
        if n/m < minratio:
            minratio = n/m
            answer = n
print(answer, minratio, time.time()- starttime)




