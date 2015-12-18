
import os
import pickle
import numpy as np
import matplotlib.pylab as plt
from demos.mesh_interp.TriangleMeshData import TriangularDataMesh2D


vertex_coords, vertex_data, triangles, data_names = pickle.load(open('demos/mesh_interp/eirene_mesh_data.pickle', 'rb'))

vertex_coords = vertex_coords/1000

datamesh = TriangularDataMesh2D(vertex_coords, vertex_data, triangles, data_names, kdtree_search=True)

plt.ion()
datamesh.plot_mesh()

# Begin testing
###############

electron_temp = datamesh.get_data_function('electron_temp')
electron_dens = datamesh.get_data_function('electron_dens')
atom_dens = datamesh.get_data_function('atom_density')
ion_dens = datamesh.get_data_function('ion_density')

# Sample our mesh for imshow test
te_samples = np.zeros((200, 200))
ne_samples = np.zeros((200, 200))
na_samples = np.zeros((200, 200))
ni_samples = np.zeros((200, 200))
xrange = np.linspace(0.15, 0.45, 200)
yrange = np.linspace(-0.2, 0.2, 200)
# xrange = list(np.arange(0.15, 0.45, 0.3/200))
# yrange = list(np.arange(-0.2, 0.2, 0.4/200))

for i, x in enumerate(xrange):
    for j, y in enumerate(yrange):
        ne_samples[j, i] = electron_dens(x, y)
        te_samples[j, i] = electron_temp(x, y)
        na_samples[j, i] = atom_dens(x, y)
        ni_samples[j, i] = ion_dens(x, y)

plt.figure()
plt.imshow(ne_samples, extent=[0.15, 0.45, -0.2, 0.2], origin='lower')
plt.colorbar()
plt.figure()
plt.imshow(te_samples, extent=[0.15, 0.45, -0.2, 0.2], origin='lower')
plt.colorbar()
plt.figure()
plt.imshow(na_samples, extent=[0.15, 0.45, -0.2, 0.2], origin='lower')
plt.colorbar()
plt.figure()
plt.imshow(ni_samples, extent=[0.15, 0.45, -0.2, 0.2], origin='lower')
plt.colorbar()
plt.show()

print(vertex_data[:, 0].min(), vertex_data[:, 0].mean(), vertex_data[:, 0].max())
