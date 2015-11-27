
cimport numpy as cnp
from raysect.core.math.function.function2d cimport Function2D


cdef class TriangularDataMesh2D:

    cdef:
        public object[:] vertices, triangles
        public cnp.float64_t[:,:] vertex_data
        dict data_names
        public object kdtree


cdef class _TriangleMeshVertex:

    cdef:
        public double u, v
        public int index
        public list triangles


cdef class _TriangleMeshTriangle:

    cdef:
        public _TriangleMeshVertex v1, v2, v3

    cdef double evaluate(self, double x, double y, cnp.float64_t[:] vertexdata)

    cdef bint contains(self, double px, double py)


cdef class InterpolatedMeshFunction(Function2D):

    cdef:
        public TriangularDataMesh2D mesh
        public int axis

    cdef double evaluate(self, double x, double y) except *

