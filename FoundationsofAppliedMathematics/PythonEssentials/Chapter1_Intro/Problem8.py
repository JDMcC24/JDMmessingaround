import math
from time import time
starttime = time()

# def alt_harmonic(n):
#     terms = []
#     for i in range(1,n+1):
#         terms.append((-1)**(i+1) *1/i) 
#     return sum(terms)



def alt_harmonic2(n):
    s=0
    for i in range(1, n+1):
        s+=(-1)**(i+1) *1/i
    return s

print(alt_harmonic2(500000), time() - starttime)
