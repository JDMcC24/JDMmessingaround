import time
from my_functions import *

def relative_totient(n):
    return n/totient(n)



starttime = time.time()
limit = 10**6
maxratio = 1
for n in range(2, limit):
    m = relative_totient(n)
    if m > maxratio:
        maxratio = m
        answr =n
print(answr, maxratio)
print(time.time() - starttime)




