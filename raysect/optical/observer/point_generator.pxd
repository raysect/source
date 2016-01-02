

cdef class PointGenerator:
    cdef list sample(self, int n)


cdef class SinglePointGenerator(PointGenerator):
    pass


cdef class CircularPointGenerator(PointGenerator):
    pass


cdef class RectangularPointGenerator(PointGenerator):
    pass
