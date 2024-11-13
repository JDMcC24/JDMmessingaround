import time
from my_functions import *

starttime = time.time()
n = 0
for i in range(1,1001):
    n+= i**i %10**10
n = n % 10**10

print(n)
print(time.time() - starttime)
