import sympy as sy
from matplotlib import pyplot as plt

def prob1():
    x,y = sy.symbols(' x y')
    return sy.Rational(2,5) * sy.exp(x**2 - y) * sy.cosh(x+y) + sy.Rational(3,7)* sy.log(x*y + 1)

print( prob1())