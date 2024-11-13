import time
from my_functions import *

starttime = time.time()
limit = 28123
limit2 = 7000
numsum = 0
nums = []
for n in range(0,limit):
    if sum_of_abundants(n) == False:
        numsum+= n
        #nums.append(n)
print(sum(nums))
#print(numsum)
print(time.time()- starttime)


#37.5256929397583