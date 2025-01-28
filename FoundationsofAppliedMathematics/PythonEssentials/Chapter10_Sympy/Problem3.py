import sympy as sy
import numpy as np
from matplotlib import pyplot as plt



def Series(N):
    x, y = sy.symbols(" x  y")
    exp = sum( (x**n)/ sy.factorial(n) for n in range(N+1))
    exp = exp.subs( x, -y**2)
    f = sy.lambdify(y, exp)
    return f

dom = np.linspace(-2,2)

for N in range(1,6,1):
    fun = Series(N)
    plt.plot(dom, fun(dom), label = "Series for N = "+ str(N))

plt.plot(dom, np.exp(- dom**2), label ="Exact function")
plt.xlim((-2,2))
plt.ylim((-2,2))
plt.title("MacLauren Series convergence for exp(-y^2)")
plt.legend()
plt.show()


