

cdef class VectorGenerator:
    cdef list sample(self, int n)


cdef class SingleRayVectorGenerator(VectorGenerator):
    pass


cdef class LightConeVectorGenerator(VectorGenerator):
    cdef public double acceptance_angle


cdef class HemisphereVectorGenerator(VectorGenerator):
    pass


cdef class CosineHemisphereVectorGenerator(VectorGenerator):
    pass
