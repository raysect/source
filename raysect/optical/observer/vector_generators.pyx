
# Classes for generating vectors and points which sample over a pixel's acceptance cone. These classes are split into
# two categories based on the way they sample areas of the pixel surface, and solid angles.

from raysect.core.math.vector import Vector3D
from raysect.core.math.random import vector_hemisphere_uniform, vector_hemisphere_cosine


cdef class VectorGenerator:
    """
    Base class for defining the solid angle (acceptance cone) for rays launched by an observer. Observers use the
    VectorGenerator and PointGenerator classes to build N rays for sampling.
    """
    pass

    def __call__(self, n):
        """
        :param int n: Generate n vectors that sample this observers acceptance solid angle.
        """
        return self.sample(n)

    cpdef list sample(self, int n):
        """
        :param int n: Generate n vectors that sample this observers acceptance solid angle.
        """
        raise NotImplemented("The method sample(n) for this vector generator needs to be implemented.")


cdef class SingleRayVectorGenerator(VectorGenerator):
    """
    Fires a single ray along the observer axis N times. Effectively a delta function acceptance cone.
    """

    cpdef list sample(self, int n):
        cdef list results
        cdef int i

        results = []
        for i in range(n):
            results.append(Vector3D(0, 0, 1))
        return results


cdef class LightConeVectorGenerator(VectorGenerator):
    """
    A conical ray acceptance volume. An example would be the light cone accepted by an optical fibre.
    """

    def __init__(self, double acceptance_angle):
        """
        :param double acceptance_angle: The angle defining a cone for this observers acceptance solid angle.
        """
        self.acceptance_angle = acceptance_angle


cdef class HemisphereVectorGenerator(VectorGenerator):
    """
    Samples rays over hemisphere in direction of surface normal.
    """
    cpdef list sample(self, int n):
        cdef list results
        cdef int i

        results = []
        for i in range(n):
            results.append(vector_hemisphere_uniform())
        return results


cdef class CosineHemisphereVectorGenerator(VectorGenerator):
    """
    Samples rays over a cosine-weighted hemisphere in direction of surface normal.
    """
    cpdef list sample(self, int n):
        cdef list results
        cdef int i

        results = []
        for i in range(n):
            results.append(vector_hemisphere_cosine())
        return results
