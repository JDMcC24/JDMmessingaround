

n=20
dsum = 1
product =1
for i in range(1,21):
    product*= i

while dsum > 0 and n < product:
    n+=20
    divisors = []
    for i in range(1,21):
        divisors.append(n%i)
    dsum = sum(divisors)
print(n)


