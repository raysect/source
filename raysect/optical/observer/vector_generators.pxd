

cdef class VectorGenerator:
    cpdef list sample(self, int n)


cdef class SingleRay(VectorGenerator):
    pass


cdef class Cone(VectorGenerator):
    cdef public double acceptance_angle


cdef class Hemisphere(VectorGenerator):
    pass


cdef class CosineHemisphere(VectorGenerator):
    pass


cdef class CosineHemisphereWithForwardBias(VectorGenerator):
    cpdef double forward_bias
