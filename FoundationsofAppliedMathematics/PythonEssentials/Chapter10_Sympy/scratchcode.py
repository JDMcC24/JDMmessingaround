import sympy as sy

#Defining multiple variables simultaneously
x1,x2,x3,x4,x5 = sy.symbols(' x1:6')

expr = x1**2 + 2*x1*x2 + x2**2
print(sy.factor(expr))

print(sy.Rational(2,3)* sy.sin(x1))