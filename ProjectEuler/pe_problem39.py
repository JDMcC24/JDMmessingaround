import time

starttime = time.time()

limit = 1000
maxcount = 0
for p in range(1,limit+1):

    count = 0
    for a in range(1,p+1):
        for b in range(1,a+1):
            c = p - a - b
            if a**2 + b**2 == c**2:
                count +=1
    if count > maxcount:
        maxcount = count
        pmax = p

print(maxcount, pmax, time.time() - starttime)

for a in range(pmax):
    for b in range(1,a):
        c = pmax - a - b
        if a**2 + b**2 == c**2:
            count +=1
            print([a,b,c])