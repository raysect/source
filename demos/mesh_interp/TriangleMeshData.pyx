
from scipy.spatial import KDTree
import numpy as np
cimport numpy as cnp


cdef class TriangularDataMesh2D:
    """
    An abstract data structure for interpolating data points lying on a triangular mesh.
    """

    def __init__(self, vertices, vertex_data, triangles, data_names):
        """
        :param vertices:
        :param vertex_data:
        :param triangles:
        :param data_names:
        :return:
        """

        self.vertices = np.zeros((vertices.shape[0]), dtype=object)
        for i, vertex in enumerate(vertices):
            self.vertices[i] = _TriangleMeshVertex(vertex[0], vertex[1], i)

        self.vertex_data = vertex_data.copy()

        self.triangles = np.zeros((triangles.shape[0]), dtype=object)
        for i, triangle in enumerate(triangles):
            try:
                v1 = _TriangleMeshVertex._all_vertices[tuple(triangle[0])]
                v2 = _TriangleMeshVertex._all_vertices[tuple(triangle[1])]
                v3 = _TriangleMeshVertex._all_vertices[tuple(triangle[2])]
            except IndexError:
                raise ValueError("vertex could not be found in vertex list")

            triangle = _TriangleMeshTriangle(v1, v2, v3)
            v1.triangles.append(triangle)
            v2.triangles.append(triangle)
            v3.triangles.append(triangle)

            self.triangles[i] = triangle

        self.data_names = {}
        for i, name in enumerate(data_names):
            self.data_names[name] = i

        unique_vertices = [(vertex.u, vertex.v) for vertex in _TriangleMeshVertex.all_verticies()]

        # TODO - implement KD-tree here
        # construct KD-tree from vertices
        self.kdtree = KDTree(unique_vertices)

    def get_data_function(self, name):
        data_axis = self.data_names[name]
        return InterpolatedMeshFunction(self, data_axis)


cdef class _TriangleMeshVertex:
    """
    An individual vertex of the mesh.
    """
    _all_vertices = {}

    def __init__(self, u, v, index):

        self.u = u
        self.v = v
        self.index = index
        self.triangles = []

        _TriangleMeshVertex._all_vertices[(u, v)] = self

    def __iter__(self):
        for tri in self.triangles:
            yield(tri)

    @classmethod
    def all_verticies(cls):
        return cls._all_vertices.values()


cdef class _TriangleMeshTriangle:

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

    def __call__(self, double x, double y, vertexdata):
        return self.evaluate(x, y, vertexdata)

    cdef double evaluate(self, double x, double y, cnp.float64_t[:] vertexdata):

        cdef:
            double alpha, beta, gamma, alpha_data, beta_data, gamma_data
            _TriangleMeshVertex v1, v2, v3

        v1 = self.v1
        v2 = self.v2
        v3 = self.v3

        alpha_data = vertexdata[v1.index]
        beta_data = vertexdata[v2.index]
        gamma_data = vertexdata[v3.index]

        alpha = ((v2.v - v3.v)*(x - v3.u) + (v3.u - v2.u)*(y - v3.v)) / \
                ((v2.v - v3.v)*(v1.u - v3.u) + (v3.u - v2.u)*(v1.v - v3.v))
        beta = ((v3.v - v1.v)*(x - v3.u) + (v1.u - v3.u)*(y - v3.v)) /\
               ((v2.v - v3.v)*(v1.u - v3.u) + (v3.u - v2.u)*(v1.v - v3.v))
        gamma = 1.0 - alpha - beta

        return alpha * alpha_data + beta * beta_data + gamma * gamma_data


    cdef bint contains(self, double px, double py):
        """
        Test if a 2D point lies inside this triangle.

        Covert

        :param Point2D point: The point of interest
        :return True or False.
        """
        cdef:
            double v1, v2, v3, alpha, beta, gamma

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


# Construct an interpolated mesh function on the fly for specific data type attached to this mesh.
cdef class InterpolatedMeshFunction(Function2D):

    def __init__(self, mesh, data_axis):
        self.mesh = mesh
        self.axis = data_axis


    def __call__(self, double x, double y):
        return self.evaluate(x, y)

    cdef double evaluate(self, double x, double y) except *:
        cdef:
            int i_closest
            double dist
            _TriangleMeshVertex closest_vertex
            _TriangleMeshTriangle triangle

        # Find closest vertex through KD-tree
        dist, i_closest = self.mesh.kdtree.query((x, y))
        closest_vertex = self.mesh.vertices[i_closest]

        # cycle through all triangles connected to this vertex
        for triangle in closest_vertex.triangles:

            if triangle.contains(x, y):

                print("##############################")
                print("THIS BIT RUNS")
                print("##############################")

                # get memory view of vertex data for this parameter and evaluate
                vertexdata = self.mesh.vertex_data[self.axis, :]
                print(triangle.evaluate(x, y, vertexdata))
                return triangle.evaluate(x, y, vertexdata)

        return 0.0
