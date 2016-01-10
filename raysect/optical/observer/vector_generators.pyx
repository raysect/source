
# Classes for generating vectors and points which sample over a pixel's acceptance cone. These classes are split into
# two categories based on the way they sample areas of the pixel surface, and solid angles.

from numpy import pi as PI

from raysect.core.math.vector import Vector3D
from raysect.core.math.random import vector_hemisphere_uniform, vector_hemisphere_cosine, vector_cone


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


cdef class SingleRay(VectorGenerator):
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


cdef class Cone(VectorGenerator):
    """
    A conical ray acceptance volume. An example would be the light cone accepted by an optical fibre.
    """

    def __init__(self, double acceptance_angle=PI/8):
        """
        :param double acceptance_angle: The angle defining a cone for this observers acceptance solid angle.
        """
        if not 0 <= acceptance_angle <= PI/4:
            raise RuntimeError("Acceptance angle {} for Cone VectorGenerator must be between 0 and pi/4."
                               "".format(acceptance_angle))
        self.acceptance_angle = acceptance_angle

    cpdef list sample(self, int n):
        cdef list results
        cdef int i

        results = []
        for i in range(n):
            results.append(vector_cone(self.acceptance_angle))
        return results


cdef class Hemisphere(VectorGenerator):
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


cdef class CosineHemisphere(VectorGenerator):
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


cdef class CosineHemisphereWithForwardBias(VectorGenerator):
    """
    Samples rays over a cosine-weighted hemisphere in direction of surface normal, with an optional forward bias.
    """
    def __init__(self, forward_bias=0.0):
        self.forward_bias = forward_bias

    cpdef list sample(self, int n):
        cdef list results
        cdef int i

        results = []
        for i in range(n):
            results.append((vector_hemisphere_cosine() + self.forward_bias * Vector3D(0, 0, 1)).normalise())
        return results
