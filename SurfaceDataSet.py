from SyntheticSurface import generateSurface
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt


def fitPoly(x, y, z):

    x, y, z = localizeDisturbance(x, y, z)

    dim = len(x)

    features = {"x^0*y^0": np.matmul(x ** 0, y ** 0).flatten(), "x*y": np.matmul(x, y).flatten(),
                "x^0*y^2": np.matmul(x**0, y ** 2).flatten(), "x^2*y^0": np.matmul(x ** 2, y ** 0).flatten(),
                "x^2*y": np.matmul(x ** 2, y).flatten(), "x*y^2": np.matmul(x, y ** 2).flatten(),
                "x^3*y^0": np.matmul(x ** 3, y**0).flatten(), "x^0*y^3": np.matmul(x ** 0, y ** 3).flatten(),
                "sin(x)": np.matmul(np.sin(x), y**0).flatten(), "sin(y)": np.matmul(x ** 0, np.sin(y)).flatten()}

    dataset = pd.DataFrame(features)

    reg = LinearRegression()
    reg.fit(dataset.values, z.flatten())

    z_pred = reg.intercept_ + np.matmul(dataset.values, reg.coef_.reshape(-1, 1)).reshape(
        dim, dim
    )

    # visualize the results
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection="3d")
    # plot the fitted curve
    ax.plot_wireframe(x, y, z_pred, label="prediction")
    # plot the target values
    ax.scatter(x, y, z, c="r", label="datapoints")
    ax.view_init(25, 80)
    plt.legend()
    plt.show()

    print(reg.coef_)

def localizeDisturbance(x, y, z):
    #Function to find coordinate range where disturbance occurs to localize the region of best fit
    threshold = 0.001

    aboveThresh_mask = z > threshold

    # find max column and row size
    row_indices, col_indices = np.where(aboveThresh_mask)

    # Calculate the maximum row size and column size
    max_row_size = np.max(np.bincount(row_indices))
    max_col_size = np.max(np.bincount(col_indices))

    yStart = col_indices[0]
    xStart = row_indices[0]

    maxSize = max(max_row_size, max_col_size)
    # maxSizeY = max(yStart + maxSize, len(y))
    # maxSizeX = max(xStart + maxSize, len(x))

    if max_row_size >= max_col_size:
        maxSize = max_row_size
        startI = xStart
    else:
        maxSize = max_col_size
        startI = yStart

    # chop my matrices based on where the bumpy is

    endI = startI + maxSize
    x_new = x[startI:endI, startI:endI]
    y_new = y[startI:endI, startI:endI]

    z_new = z[startI:endI, startI:endI]

    return np.array(x_new, dtype='float32'), np.array(y_new, dtype='float32'), np.array(z_new, dtype='float32')
