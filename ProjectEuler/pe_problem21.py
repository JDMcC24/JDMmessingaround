import time
from my_functions import *

starttime = time.time()
limit = 10000
am_list = []
for a in range(10,limit+1):
    for b in range(2,a):
        if is_amicable(b,a) == True:
            am_list.append(b)
            am_list.append(a)

print(sum(am_list), am_list)
print(time.time()- starttime)
