
n=100

sqr = (n*(n+1)/2)**2
m=0
for i in range(n+1):
    m+= i**2
    print(i**2)

print(sqr - m)