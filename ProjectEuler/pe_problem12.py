from my_functions import*
import math
import time

starttime=time.time()

numlimit = 500
tri = 1
n=1
factors = number_of_divisors(tri)
while factors <=numlimit:
     tri = n*(n+1)/2
     factors = number_of_divisors(tri)
     n+=1
print(tri)
print(time.time() - starttime)






        
    