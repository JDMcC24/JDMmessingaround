import numpy as np
from matplotlib import pyplot as plt


def prob3():
    dom1 = np.linspace(-2,1)
    dom2 = np.linspace(1,6)
    ran1 = 1 / (dom1 - 1)
    ran2 = 1 / (dom2 - 1)
    plt.plot(dom1,ran1, 'm--', linewidth = 4) 
    plt.plot(dom2,ran2, 'm--', linewidth = 4)
    plt.xlim((-2,6))
    plt.ylim((-6,6))
    plt.show()

prob3()