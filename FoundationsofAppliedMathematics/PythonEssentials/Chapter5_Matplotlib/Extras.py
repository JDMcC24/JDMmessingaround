from matplotlib import pyplot as plt
import numpy as np

#3D plotting
def extras():
    x = np.linspace(-2* np.pi, 2*np.pi)
    y = np.copy(x)
    X, Y = np.meshgrid(x,y)
    g = np.sin(X) * np.sin(Y) / (X * Y)
    ax = plt.figure().add_subplot(1,1,1, projection = '3d')
    ax.plot_surface(X,Y,g)
    plt.show()
extras()