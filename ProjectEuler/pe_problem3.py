from my_functions import *
import math

big_prime = 0

for i in range(1,math.ceil(math.sqrt(600851475143))):
    if 600851475143 % i == 0 and is_prime(i):
        big_prime = i

print(big_prime)