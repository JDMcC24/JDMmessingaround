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

def prob61():
    """ Plot the function g(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of g, and one with a contour
            map of g. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Include a color scale bar for each subplot.
    """
    x = np.linspace(-2 * np.pi, 2 * np.pi)
    y = x
    X, Y = np.meshgrid(x,y)
    Z = np.sin(X) * np.sin(Y) / (X * Y)
    plt.subplot(1,2,1)
    plt.contour(X,Y,Z, 20, cmap = 'coolwarm' )
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.pcolormesh(X,Y,Z, cmap = 'coolwarm', shading = 'auto')
    plt.colorbar()

    plt.axis([-2* np.pi, 2* np.pi,-2* np.pi, 2* np.pi])
    return plt.show()

prob6()
