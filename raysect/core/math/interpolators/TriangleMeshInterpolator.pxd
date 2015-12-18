
import numpy as np
cimport numpy as cnp
from raysect.core.math.function.function2d cimport Function2D


cdef class TriangleMeshInterpolator(Function2D):

    cdef:
        public object[:] vertices, triangles
        public double [:] vertex_data
        public object kdtree
        public bint kdtree_search

    cdef double evaluate(self, double x, double y) except *

    cpdef _TriangleMeshTriangle find_triangle_containing(self, double u, double v)

    cdef _TriangleMeshTriangle brute_force_method(self, double u, double v)

    cdef _TriangleMeshTriangle kdtree_method(self, double u, double v)

    cdef double get_vertex_data(self, int vertex_index)


cdef class _TriangleMeshVertex:

    cdef:
        public double u, v
        public int index
        public list triangles


cdef class _TriangleMeshTriangle:

    cdef:
        public _TriangleMeshVertex v1, v2, v3

    cdef double evaluate(self, double x, double y, TriangleMeshInterpolator mesh)

    cdef bint contains(self, double px, double py)
