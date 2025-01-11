from matplotlib import pyplot as plt
import numpy as np

def prob4():
    dom = np.linspace(0, 2* np.pi)
    y1 = np.sin(dom)
    plt.subplot(221)
    plt.plot(dom,y1,  'g-', label = "Sin(x)")
    plt.axis([0, 2* np.pi, -2,2])
    plt.title("Sin(x)")
    plt.subplot(222)
    y2 = np.sin(2* dom)
    plt.plot(dom,y2,  'r--', label = "Sin(2x)")
    plt.title("Sin(2x)")
    plt.axis([0, 2* np.pi, -2,2])
    plt.subplot(223)
    y3 = 2* np.sin( dom)
    plt.axis([0, 2* np.pi, -2,2])
    plt.plot(dom,y3,  'b--')
    plt.title("2 Sin(x)")
    plt.subplot(224)
    y3 = 2* np.sin( 2* dom)
    plt.axis([0, 2* np.pi, -2,2])
    plt.plot(dom,y3,  'm--')
    plt.title("2 Sin(2 x)")
    plt.suptitle("Problem 4 plots")
    plt.show()


prob4()