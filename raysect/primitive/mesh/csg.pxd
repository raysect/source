
from raysect.primitive.mesh.mesh cimport Mesh


cdef class CSG_Operator:

    cdef bint _m1(self, double signed_distance)

    cdef bint _m2(self, double signed_distance)


cdef class Union(CSG_Operator):
    pass


cdef class Intersect(CSG_Operator):
    pass


cdef class Subtract(CSG_Operator):
    pass


cpdef Mesh perform_mesh_csg(Mesh mesh_1, Mesh mesh_2, CSG_Operator operator)
