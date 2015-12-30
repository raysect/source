
import os
import pickle
import numpy as np
import matplotlib.pylab as plt
from raysect.core.math.interpolators import Interpolator2DMesh

vertex_coords, vertex_data, triangles, data_names = pickle.load(open('eirene_mesh_data.pickle', 'rb'))
# vertex_coords, vertex_data, triangles, data_names = pickle.load(open('demos/mesh_interp/eirene_mesh_data.pickle', 'rb'))

# Units correction, distances need to be in metres.
vertex_coords = vertex_coords / 1000

ion_dens_data = np.array(vertex_data[:, 0])
atom_dens_data = np.array(vertex_data[:, 1])
edens_data = np.array(vertex_data[:, 2])
etemp_data = np.array(vertex_data[:, 3])

ion_dens = Interpolator2DMesh(vertex_coords, ion_dens_data, triangles, limit=False, default_value=0)

atom_dens = Interpolator2DMesh(instance=ion_dens, vertex_data=atom_dens_data)
electron_dens = Interpolator2DMesh(instance=ion_dens, vertex_data=edens_data)
electron_temp = Interpolator2DMesh(instance=ion_dens, vertex_data=etemp_data)

# future api?
# atom_dens = Interpolator2DMesh.instance(ion_dens, vertex_data=atom_dens_data)
# electron_dens = Interpolator2DMesh.instance(ion_dens, vertex_data=edens_data)
# electron_temp = Interpolator2DMesh.instance(ion_dens, vertex_data=etemp_data)


plt.ioff()
# ion_dens.plot_mesh()

# Begin test output
###################

te_samples = np.zeros((100, 100))
ne_samples = np.zeros((100, 100))
na_samples = np.zeros((100, 100))
ni_samples = np.zeros((100, 100))
xrange = np.linspace(0.15, 0.45, 100)
yrange = np.linspace(-0.2, 0.2, 100)

for i, x in enumerate(xrange):
    for j, y in enumerate(yrange):
        ne_samples[j, i] = electron_dens(x, y)
        te_samples[j, i] = electron_temp(x, y)
        na_samples[j, i] = atom_dens(x, y)
        ni_samples[j, i] = ion_dens(x, y)
    print(i)

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

# print(vertex_data[:, 0].min(), vertex_data[:, 0].mean(), vertex_data[:, 0].max())
# print(data_names)
