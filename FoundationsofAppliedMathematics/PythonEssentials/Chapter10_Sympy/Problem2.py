import sympy as sy
from matplotlib import pyplot as plt

def prob2():
    x, j ,i  = sy.symbols(' x j i')
    expr = sy.product(sy.summation( j * ( sy.sin(x) + sy.cos(x)), (j,1,5)), (i,1,5))
    return sy.simplify(expr)

print(prob2())