
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.io import loadmat
from numpy import array, zeros, arange
from matplotlib.pylab import imshow, show
import os


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
# It would allow us to write a list of triangle vertices with associated data values.


# to differentiate the api from the mesh primitive do the following:
# the init accepts a Nx3 list of vertices, and an Mx4 list of faces (3 vertices and a value)
# the interpolator uses triangle internally, it is not exposed to the user.
# this is to avoid there being multiple incompatible Triangle classes floating around the API.

# TODO - This name could be better.
# Interpolator2DMesh(Function2D)
class TriangularDataMesh:

    def __init__(self, *data_names):
        self._triangles = []
        self._vertices = {}

        if data_names is None or len(data_names) == 0:
            raise ValueError("Must be at least one data value for the mesh.")

        for name in data_names:
            if not isinstance(name, str):
                raise ValueError("All data names must be strings.")

        self.attached_data_names = data_names

        self._bounding_box = None

    def __call__(self, u, v, dataname):

        triangle = self.find_triangle_containing((u, v))

        if triangle:
            return triangle((u, v), dataname)
        else:
            return 0.0

    @property
    def number_of_triangles(self):
        return len(self._triangles)

    @property
    def number_of_vertices(self):
        return len(self._vertices)

    # TODO - better name for this function
    def add_vertex(self, u, v, data):
        """
        Make a new 2D vertex point with associated data.

        :param float u: U coordinate
        :param float v: V coordinate
        :param list data: List of data values for this vertex.
        :return: new_vertex or the existing vertex that was found
        """

        if not len(data) == len(self.attached_data_names):
            ValueError("Insufficient data given for this vertex. "
                       "Every vertex in the mesh must have the same number of data values.")

        try:
            # Try to load the vertex if it already exists
            vertex = self._vertices[(u, v)]

        except KeyError:            # Else, create the vertex and return it, add to list of mesh vertices.
            vertex = TriangleMeshVertex(u, v, self.attached_data_names, data)
            self._vertices[(u, v)] = vertex
        return vertex

    def add_triangle(self, v1, v2, v3):
        """
        Make a new triangle for three input vertices.

        :param v1: Vertex 1. Eventually Point 2D, currently Tuple.
        :param v2: Vertex 2. Eventually Point 2D, currently Tuple.
        :param v3: Vertex 3. Eventually Point 2D, currently Tuple.
        :param data: Tuple value for each data value in order.
        """

        triangle = TriangleMeshTriangle(v1, v2, v3)
        self._triangles.append(triangle)

        return triangle

    def find_triangle_containing(self, point):

        for triangle in self._triangles:
            result = triangle.contains(point)
            if result:
                return triangle
        return None


# Triangle
class TriangleMeshTriangle:

    def __init__(self, v1, v2, v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

    def __iter__(self):
        yield(self.v1)
        yield(self.v2)
        yield(self.v3)

    def __repr__(self):
        repr_str = "v1 => ({}, {})\n".format(self.v1.u, self.v1.v)
        repr_str += "v2 => ({}, {})\n".format(self.v2.u, self.v2.v)
        repr_str += "v3 => ({}, {})".format(self.v3.u, self.v3.v)
        return repr_str

    def __call__(self, point, data_name):

        px, py = point
        v1, v2, v3 = self

        try:
            alpha_data = getattr(v1, data_name)
            beta_data = getattr(v2, data_name)
            gamma_data = getattr(v3, data_name)

        except AttributeError:
            raise ValueError("The data value requested ({}) could not be found on these vertices.".format(data_name))

        alpha = ((v2.v - v3.v)*(px - v3.u) + (v3.u - v2.u)*(py - v3.v)) / \
                ((v2.v - v3.v)*(v1.u - v3.u) + (v3.u - v2.u)*(v1.v - v3.v))
        beta = ((v3.v - v1.v)*(px - v3.u) + (v1.u - v3.u)*(py - v3.v)) /\
               ((v2.v - v3.v)*(v1.u - v3.u) + (v3.u - v2.u)*(v1.v - v3.v))
        gamma = 1.0 - alpha - beta

        return alpha * alpha_data + beta * beta_data + gamma * gamma_data

    def contains(self, point):
        """
        Test if a 2D point lies inside this triangle.

        Covert

        :param Point2D point: The point of interest
        :return True or False.
        """

        px, py = point
        v1, v2, v3 = self

        # Compute barycentric coordinates
        alpha = ((v2.v - v3.v)*(px - v3.u) + (v3.u - v2.u)*(py - v3.v)) / \
                ((v2.v - v3.v)*(v1.u - v3.u) + (v3.u - v2.u)*(v1.v - v3.v))
        beta = ((v3.v - v1.v)*(px - v3.u) + (v1.u - v3.u)*(py - v3.v)) /\
               ((v2.v - v3.v)*(v1.u - v3.u) + (v3.u - v2.u)*(v1.v - v3.v))
        gamma = 1.0 - alpha - beta

        # Point is inside triangle if all coordinates between [0, 1]
        if alpha > 0 and beta > 0 and gamma > 0:
            return True
        else:
            return False

# this won't be necessary if the init of the interpolator takes lists/ndarrays of vertices + faces
class TriangleMeshVertex:
    """
    An individual vertex of the mesh.
    """

    _all_verticies = {}

    def __init__(self, u, v, data_names, data_values):

        self.u = u
        self.v = v

        # Set data values as attributes
        for k in range(len(data_names)):
            setattr(self, data_names[k], data_values[k])


if __name__ == '__main__':

    # Loading SOLPS data
    ####################

    solps_pth = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources/solps_39625.mat')
    # solps_output = loadmat('./demos/resources/solps_39625.mat')
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

    datamesh = TriangularDataMesh("linerad")

    # Generate the data interpolators
    for i in range(0, mesh_shape[0]):
        for j in range(0, mesh_shape[1]):
            # Get centre point of cell
            centre_pts.append([cr[i, j], cz[i, j]])

            # get linerad data for this cell
            tot_linerad.append((solps_output['linerad'][i, j, :].sum() + solps_output['brmrad'][i, j, :].sum()) /
                               solps_output['vol'][i, j])

    # Generate interpolators for data
    centre_pts = array(centre_pts)
    tot_linerad = array(tot_linerad)
    tot_linerad /= tot_linerad.max()
    tot_linerad = CloughTocher2DInterpolator(centre_pts, tot_linerad, fill_value=0.0)

    # Generate the triangle mesh
    for i in range(0, mesh_shape[0]):
        for j in range(0, mesh_shape[1]):

            # Get vertices of cells
            u, v = r[i, j, 0], z[i, j, 0]
            v1 = datamesh.add_vertex(u, v, [tot_linerad((u, v))])
            u, v = r[i, j, 2], z[i, j, 2]
            v2 = datamesh.add_vertex(u, v, [tot_linerad((u, v))])
            u, v = r[i, j, 3], z[i, j, 3]
            v3 = datamesh.add_vertex(u, v, [tot_linerad((u, v))])
            u, v = r[i, j, 1], z[i, j, 1]
            v4 = datamesh.add_vertex(u, v, [tot_linerad((u, v))])

            # Create mesh triangles associated with this vertices.
            triangle1 = datamesh.add_triangle(v1, v2, v3)
            triangle2 = datamesh.add_triangle(v3, v4, v1)

    # Begin testing
    ###############

    # Sample our mesh for imshow test
    samples = zeros((50, 50))
    xrange = list(arange(0, 2, 2/50))
    yrange = list(arange(-2, 0, 2/50))

    for x in xrange:
        print(x)
        for y in yrange:
            samples[y, x] = datamesh(x, y, "linerad")

    imshow(samples, extent=[minr, maxr, minz, maxz], origin='lower')
    show()
