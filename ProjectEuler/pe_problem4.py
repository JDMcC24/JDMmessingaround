from my_functions import *


palindromes = []
for i in range(100,1000):
    for j in range(i,1000):
        if is_palindrome(i*j):
            palindromes.append(i*j)


print(max(palindromes))
