
import os
from scipy.io import loadmat
from scipy.interpolate import CloughTocher2DInterpolator
import numpy as np
import matplotlib.pylab as plt
from demos.mesh_interp.TriangleMeshData import TriangularDataMesh2D

DATA_NAMES = ['linerad']
XRANGE = (0.2, 0.7)
YRANGE = (-1.4, -0.9)

# Loading SOLPS data
####################

solps_pth = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'solps_39625.mat')
solps_output = loadmat(solps_pth)

# temporary variable for all the r's and z's
r = solps_output['r']
z = solps_output['z']
cr = solps_output['cr']
cz = solps_output['cz']

# extract the number of poloidal and radial grid points
mesh_shape = solps_output['r'].shape[0:2]
minr = r.flatten().min()
maxr = r.flatten().max()
minz = z.flatten().min()
maxz = z.flatten().max()

centre_pts = []
tot_linerad = []

# Generate the data interpolators
for i in range(0, mesh_shape[0]):
    for j in range(0, mesh_shape[1]):
        # Get centre point of cell
        centre_pts.append([cr[i, j], cz[i, j]])

        # get linerad data for this cell
        tot_linerad.append((solps_output['linerad'][i, j, :].sum() + solps_output['brmrad'][i, j, :].sum()) /
                           solps_output['vol'][i, j])

# Generate interpolators for data
centre_pts = np.array(centre_pts)
tot_linerad = np.array(tot_linerad)
tot_linerad /= tot_linerad.max()
tot_linerad = CloughTocher2DInterpolator(centre_pts, tot_linerad, fill_value=0.0)


vertices_dict = {}
# vertex tuple is the key (guaranties unique vertices only), values is a list of the data
triangles_with_coords = np.zeros((mesh_shape[0]*mesh_shape[1]*2, 3, 2))
triangles_with_index = np.zeros((mesh_shape[0]*mesh_shape[1]*2, 3))
# triangles shape => [number of boxes * 2, 3 vertices, 2 values per vertex]

print("vertex shape - {}".format(r.shape))
print("triangle shape - {}".format(triangles_with_coords.shape))
print("mesh shape - {}".format(mesh_shape))
# Generate the triangle mesh
for i in range(mesh_shape[0]):
    for j in range(mesh_shape[1]):
        # Get vertices of cells
        u, v = r[i, j, 0], z[i, j, 0]
        vertices_dict[(u, v)] = [tot_linerad((u, v))]
        v1 = (u, v)

        u, v = r[i, j, 2], z[i, j, 2]
        vertices_dict[(u, v)] = [tot_linerad((u, v))]
        v2 = (u, v)

        u, v = r[i, j, 3], z[i, j, 3]
        vertices_dict[(u, v)] = [tot_linerad((u, v))]
        v3 = (u, v)

        u, v = r[i, j, 1], z[i, j, 1]
        vertices_dict[(u, v)] = [tot_linerad((u, v))]
        v4 = (u, v)

        # Create mesh triangles associated with these vertices.
        triangles_with_coords[i*mesh_shape[1]+j*2, :, :] = (v1, v2, v3)
        triangles_with_coords[i*mesh_shape[1]+j*2+1, :, :] = (v3, v4, v1)

vertices = np.zeros((len(triangles_with_coords), 2))
vertex_data = np.zeros((len(triangles_with_coords), len(DATA_NAMES)))
for i, key in enumerate(vertices_dict):
    vertices[i, :] = key
    vertex_data[i, :] = vertices_dict[key]
    vertices_dict[key] = i

for i, triangle in enumerate(triangles_with_coords):
    v1 = (triangle[0, 0], triangle[0, 1])
    v2 = (triangle[1, 0], triangle[1, 1])
    v3 = (triangle[2, 0], triangle[2, 1])

    print(v1, v2, v3)
    print((vertices_dict[v1], vertices_dict[v2], vertices_dict[v3]))
    triangles_with_index[i, :] = (vertices_dict[v1], vertices_dict[v2], vertices_dict[v3])

datamesh = TriangularDataMesh2D(vertices, vertex_data, triangles_with_index, DATA_NAMES, kdtree_search=False)


# # Begin testing
# ###############
#
# linerad = datamesh.get_data_function('linerad')
#
# # Sample our mesh for imshow test
# samples = np.zeros((100, 50))
# xrange = list(np.arange(0, 1, 1/50))
# yrange = list(np.arange(-2, 0, 2/100))
#
# for i, x in enumerate(xrange):
#     for j, y in enumerate(yrange):
#         samples[j, i] = linerad(x, y)
#
# plt.ion()
# # imshow(samples, extent=[minr, maxr, minz, maxz], origin='lower')
# plt.imshow(samples, extent=[0, 1, -2, 0], origin='lower')
# plt.show()
