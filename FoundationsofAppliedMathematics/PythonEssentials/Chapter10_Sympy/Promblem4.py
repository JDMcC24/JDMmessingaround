import sympy as sy
import numpy as np
import matplotlib.pyplot as plt

def prob4():
    x, y, t ,r = sy.symbols(' x, y t r')
    num = (x**2 + y**2)**(sy.Rational(7,2)) + 18* x**5 * y - 60* x**3 * y **3 + 18 * x* y**5
    denom = (x**2 + y**2)**3
    exp = 1-  num/denom
    exp = exp.subs({x: r *sy.cos(t), y: r * sy.sin(t)})
    exp = sy.simplify(exp)
    [r1, r2] = sy.solve(exp,r)
    dom = np.linspace(0, 2* np.pi,1000)
    r1 = sy.lambdify(t,r1)
    r2 = sy.lambdify(t,r2)

    plt.plot(r1(dom)*np.cos(dom), r1(dom)*np.sin(dom), 'b')
    plt.plot(r2(dom)*np.cos(dom), r2(dom)*np.sin(dom), 'b')
    plt.title("Rose Petal Curve")
    return plt.show()

prob4()