from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def generateSurface(height, width, dim=None, n_points=50):
    # Creating dataset
    if dim is None:
        dim = [-3, 3]

    x = np.outer(np.linspace(dim[0], dim[1], n_points), np.ones(n_points))
    y = x.copy().T  # transpose

    z = height * np.exp(-(x ** 2 + y ** 2) / (2 * width ** 2))

    return [x, y, z]


def plotSurface(x, y, z):
    ax3 = plt.axes(projection='3d')
    ax3.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax3.set_zlim([0, 3])

    plt.show()


def translationFunction(force):

    height = 0.2 + (force/5)**2 + 0.1*(force/5)**3
    width = 0.4 + 2*(force/5)**2 + 0.5*(force/5)**3

    return height, width


def generateSyntheticDataset(forces):
    # generate data with a few different forces and displacements accordingly
    surfaces = {}

    for ind, force in enumerate(forces):

        h, w = translationFunction(force)
        surfaces[ind] = generateSurface(h, w)

    return surfaces
