
""" By conisdering the terms of the fibonacci sequence whoes values do not exceed four million, 
Find the sum of the even-valued terms"""

fib = [1 , 1]
while fib[-1]<4000000:
    fib.append(fib[-1] + fib[-2])
if fib[-1] >= 4000000:
    fib.remove(fib[-1])

evens = []
for i in fib:
    if i % 2 == 0:
        evens.append(i)
print(sum(evens))
