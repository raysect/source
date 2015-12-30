
import numpy as np
import matplotlib.pylab as plt
from raysect.core.math.interpolators import Interpolator2DMesh

vertices = np.array([[267.709991, -89.156998], [266.410004, -93.556999], [266.739990, -93.650002],
                     [268.000000, -89.250000], [267.850006, -93.960999], [268.980011, -89.514999],
                     [269.029999, -94.223000], [270.020020, -89.761002], [270.460022, -94.499001],
                     [271.279999, -90.058998]])

vertex_data = np.array([1, 2, 3, 4, 25, 6, 7, 8, 9, 10], dtype=np.float64)

triangles = np.array([[1, 2, 4], [2, 3, 4], [4, 3, 6], [3, 5, 6], [6, 5, 8], [5, 7, 8], [8, 7, 10], [7, 9, 10]]) - 1

linerad = Interpolator2DMesh(vertices, vertex_data, triangles, limit=False, default_value=0)
# linerad.plot_mesh()

# Sample mesh for imshow test
samples = np.zeros((300, 200))
xrange = list(np.arange(265, 275, 10/200))
yrange = list(np.arange(-95, -85, 10/300))

for i, x in enumerate(xrange):
    for j, y in enumerate(yrange):
        samples[j, i] = linerad(x, y)

plt.ioff()
plt.imshow(samples, extent=[265, 275, -95, -85], origin='lower')
plt.show()
