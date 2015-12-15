
import numpy as np
cimport numpy as cnp
from raysect.core.math.function.function2d cimport Function2D


cdef class TriangularDataMesh2D:

    cdef:
        public object[:] vertices, triangles
        public double [:,:] vertex_data
        dict data_names
        public object kdtree
        public bint kdtree_search

    cpdef _TriangleMeshTriangle find_triangle_containing(self, double u, double v)

    cdef _TriangleMeshTriangle brute_force_method(self, double u, double v)

    cdef _TriangleMeshTriangle kdtree_method(self, double u, double v)


cdef class _TriangleMeshVertex:

    cdef:
        public double u, v
        public int index
        public list triangles


cdef class _TriangleMeshTriangle:

    cdef:
        public _TriangleMeshVertex v1, v2, v3

    cdef double evaluate(self, double x, double y, TriangularDataMesh2D mesh, int data_axis)

    cdef bint contains(self, double px, double py)


cdef class InterpolatedMeshFunction(Function2D):

    cdef:
        public TriangularDataMesh2D mesh
        public int data_axis

    cdef double evaluate(self, double x, double y) except *
