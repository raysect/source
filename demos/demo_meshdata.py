
from scipy.interpolate import griddata
from scipy.io import loadmat


# Concept
# =======
#
# - Essentially, this is a mesh interpolator. Data is read in from a mesh-like data set. The data is parsed into a
# raysect mesh object. Data is stored at the mesh vertices. To get the data value at any 2D (x, y) or 3D (x, y, z)
# coordinate, the kdTree looks up the nearest triangular cells. Each triangular cell is tested for inclusion of the
# coordinate point. Once inside the point is known to be inside the cell, Barycentric interpolation is used to
# interpolate the value inside the mesh cell.
#
# - A single mesh may be used for multiple data values. For example, in the case of a plasma, the same input mesh may
# have plasma values of electron temperature, electron density, etc, as attributes.
#
# - The Polygon File Format (also known as Stanford Triangle Format) would be suitable for storing this data in files.
# It would allow us to write a list of triangle verticies with associated data values.


class DataMesh2D:
    pass


class DataMeshTriangle:

    def __init__(self, v1, v2, v3):

        self.verticies = [v1, v2, v3]

        for vertex in self.verticies:
            vertex.register_triangle(self)


class DataMeshVertex:
    """
    An individual vertex of the mesh.
    """

    _all_verticies = {}

    def __init__(self, u, v):

        self.u = u
        self.v = v

        self.triangles = []

        self.data = {}

        DataMeshVertex._all_verticies[(u, v)] = self

    def register_triangle(self, triangle):
        self.triangles.append(triangle)

    @classmethod
    def all_vertex_coords(cls):
        return [(vert.u, vert.v) for vert in cls._all_verticies.values()]

    # TODO - this function should be renamed. It doesn't quite do what its name implies.
    @classmethod
    def get_vertex(cls, coords):

        try:
            vertex = cls._all_verticies[coords]
        except IndexError:
            vertex = DataMeshVertex(*coords)

        return vertex


if __name__ == '__main__':
    solps_output = loadmat('./demos/resources/solps_39625.mat')

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

    # Generate the triangle mesh
    for i in range(0, mesh_shape[0]):
        for j in range(0, mesh_shape[1]):
            # Get vertices of cells
            cell_r = [r[i, j, 0], r[i, j, 2], r[i, j, 3], r[i, j, 1]]
            cell_z = [z[i, j, 0], z[i, j, 2], z[i, j, 3], z[i, j, 1]]

            # Create vertex objects
            v1 = DataMeshVertex.get_vertex((cell_r[0], cell_z[0]))
            v2 = DataMeshVertex.get_vertex((cell_r[1], cell_z[1]))
            v3 = DataMeshVertex.get_vertex((cell_r[2], cell_z[2]))
            v4 = DataMeshVertex.get_vertex((cell_r[3], cell_z[3]))

            # Create triangles associated with this vertices.
            triangle1 = DataMeshTriangle(v1, v2, v3)
            triangle2 = DataMeshTriangle(v3, v4, v1)

            # Extract centre point and associated data
            centre_pts.append((cr[i, j], cz[i, j]))
            tot_linerad.append((solps_output['linerad'][i, j, :].sum() + solps_output['brmrad'][i, j, :].sum()) /
                               solps_output['vol'][i, j])

    vertex_coords = DataMeshVertex.all_vertex_coords()


    interp = griddata(vertex_coords, tot_linerad, self.regular_grid_pts, method='cubic').reshape((Y_SIZE, X_SIZE))
