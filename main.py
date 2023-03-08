from SyntheticSurface import generateSyntheticDataset
import numpy as np
from SurfaceDataSet import fitPoly


# generate a few surfaces and fit a "polynomial" on it
forces = np.arange(1, 6, 0.3)
dataset = generateSyntheticDataset(forces)


for surface in dataset:
    x, y, z = dataset[surface]

    fitPoly(x, y, z)


