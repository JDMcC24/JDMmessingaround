import math

n = math.factorial(100)

n_string = str(n)
digits = []
for i in n_string:
    digits.append(int(i))
print(sum(digits))