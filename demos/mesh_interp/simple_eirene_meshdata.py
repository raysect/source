
import os
from scipy.io import loadmat
from scipy.interpolate import CloughTocher2DInterpolator
import numpy as np
import matplotlib.pylab as plt
from demos.mesh_interp.TriangleMeshData import TriangularDataMesh2D

vertices = np.array([[267.709991, -89.156998], [266.410004, -93.556999], [266.739990, -93.650002],
                     [268.000000, -89.250000], [267.850006, -93.960999], [268.980011, -89.514999],
                     [269.029999, -94.223000], [270.020020, -89.761002], [270.460022, -94.499001],
                     [271.279999, -90.058998]])

vertex_data = np.array([[1], [2], [3], [4], [25], [6], [7], [8], [9], [10]], dtype=np.float64)

triangles = np.array([[1, 2, 4], [2, 3, 4], [4, 3, 6], [3, 5, 6], [6, 5, 8], [5, 7, 8], [8, 7, 10], [7, 9, 10]],
                     dtype=np.int32) - 1

datamesh = TriangularDataMesh2D(vertices, vertex_data, triangles, ['test'], kdtree_search=True)

datamesh.plot_mesh()

# Begin testing
###############

linerad = datamesh.get_data_function('test')

# Sample our mesh for imshow test
samples = np.zeros((30, 20))
xrange = list(np.arange(265, 275, 10/20))
yrange = list(np.arange(-95, -85, 10/30))

for i, x in enumerate(xrange):
    for j, y in enumerate(yrange):
        samples[j, i] = linerad(x, y)

plt.ion()
plt.imshow(samples, extent=[265, 275, -95, -85], origin='lower')
plt.show()

