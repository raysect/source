
from raysect.core.math.affinematrix cimport AffineMatrix3D as AffineMatrix3D_t


cdef class PointGenerator:
    cdef public AffineMatrix3D_t transform
    cpdef list sample(self, int n)


cdef class SinglePointGenerator(PointGenerator):
    pass


cdef class CircularPointGenerator(PointGenerator):
    cdef public double radius


cdef class RectangularPointGenerator(PointGenerator):
    cdef public double width, height

