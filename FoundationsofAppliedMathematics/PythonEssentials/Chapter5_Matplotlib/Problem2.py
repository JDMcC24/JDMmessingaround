import numpy as np
from matplotlib import pyplot as plt

def prob2():
    dom = np.linspace(-2*np.pi, 2 * np.pi)
    y1 = np.sin(dom)
    y2 = np.cos(dom)
    y3 = np.arctan(dom)
    plt.plot(dom, y1,'b', label = "Sine")
    plt.plot(dom, y2,'g', label = "Cosine")
    plt.plot(dom, y3,'r', label = "ArcTan")
    plt.legend( loc = "upper left")
    plt.title("Problem 2")
    plt.show()

prob2()


