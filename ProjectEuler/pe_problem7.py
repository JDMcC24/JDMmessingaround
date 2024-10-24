from my_functions import *



n = 10001
primes=[2]
while len(primes) <n:
    primes.append(next_prime(primes[-1]))

print(primes[-1])
  

