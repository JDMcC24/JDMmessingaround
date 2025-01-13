from matplotlib import pyplot as plt
import numpy as np

def prob6():
    x = np.linspace(-2* np.pi, 2*np.pi)
    y = x
    X, Y = np.meshgrid(x,y)
    g = np.sin(X) * np.sin(Y) / (X * Y)
    plt.subplot(121)
    plt.pcolormesh(X,Y,g, cmap = 'magma', shading = "auto")
    plt.colorbar()
    #plt.xlim(-2* np.pi, 2*np.pi)
    #plt.ylim(-2* np.pi, 2*np.pi)

    plt.subplot(122)
    plt.contour(X,Y,g, 20, cmap = 'coolwarm')
    plt.colorbar()


    plt.axis([-2* np.pi, 2* np.pi,-2* np.pi, 2* np.pi])
    return plt.show()

prob6()
