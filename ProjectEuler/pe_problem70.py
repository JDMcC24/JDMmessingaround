import time
from my_functions import *
import itertools



# starttime = time.time()
# limit = 10**5
# minratio = 10
# answer = 1
# for n in range(2,limit):
#     m = totient(n)
#     if m in digit_permutations(n) and n/m < minratio:
#         answer = n
#         minratio = m
#         print(n)
# print(answer, time.time()- starttime)

starttime = time.time()
limit = 10**7
minratio = 10
answer = 1
for n in range(10**6,limit):
    m = euler_totient(n)
    if m in digit_permutations(n) and n/m < minratio:
        answer = n
        minratio = n/m
        print(n)
print(answer, time.time()- starttime)


