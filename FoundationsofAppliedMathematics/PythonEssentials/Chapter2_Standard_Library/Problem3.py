import calculator as cal

def prob3(a,b):
    a = cal.product(a,a)
    b = cal.product(b,b)
    c = cal.sum(a,b)
    c = cal.sqrt(c)
    return c

print(prob3(3,4)) 
