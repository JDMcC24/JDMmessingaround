import time
from my_functions import *

def relative_totient(n):
    return n/totient(n)



starttime = time.time()
# limit = 7*10**3
# maxratio = 1
# for n in range(2, limit):
#     m = relative_totient(n)
#     if m > maxratio:
#         maxratio = m
#         answr =n
# print(answr, maxratio)

last_prime = 2
max=2
while max < 10**6:
    p = next_prime(last_prime)
    max *= p
    last_prime = p
print(max/prime_factors(max)[-1])

print(time.time() - starttime)


