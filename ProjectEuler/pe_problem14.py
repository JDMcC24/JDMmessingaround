from my_functions import *
import time


startime = time.time()
m = 0
for n in range(2,1000000):
    if Czlist(n)[1] > m:
        m = Czlist(n)[1]
        aswr = n
print(aswr,m)
print( time.time() - startime)

