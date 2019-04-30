
from raysect.core cimport Point3D, Vector3D


cdef class Triangle:

    cdef:
        readonly Point3D v1, v2, v3, vc
        readonly Vector3D normal
        readonly double area


cpdef tuple triangle3d_intersects_triangle3d(Triangle triangle_1, Triangle triangle_2, double tolerance=*)
