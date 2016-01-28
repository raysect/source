# cython: language_level=3

# Copyright (c) 2014, Dr Alex Meakins, Raysect Project
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the Raysect Project nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


# Classes for generating vectors and points which sample over a pixel's acceptance cone. These classes are split into
# two categories based on the way they sample areas of the pixel surface, and solid angles.

from libc.math cimport M_PI as PI
from raysect.core.math.vector import Vector3D
from raysect.core.math.random import vector_hemisphere_uniform, vector_hemisphere_cosine, vector_cone, vector_sphere


cdef class VectorGenerator:
    """
    Base class for an object that generates a list of Vector3D objects.
    """

    def __call__(self, samples):
        """
        :param int samples: Number of vectors to generate.
        """
        return self.sample(samples)

    cpdef list sample(self, int samples):
        """
        :param int samples: Number of vectors to generate.
        """
        raise NotImplemented("The method sample() is not implemented for this vector generator.")


cdef class SingleRay(VectorGenerator):
    """
    Fires a single ray along the observer axis N times. Effectively a delta function acceptance cone.
    """

    cpdef list sample(self, int samples):
        cdef list results
        cdef int i

        results = []
        for i in range(samples):
            results.append(Vector3D(0, 0, 1))
        return results


cdef class ConeUniform(VectorGenerator):
    """
    Generates a list of random unit Vector3D objects inside a cone.

    The cone is aligned along the z-axis.

    :param angle: Angle of the cone in degrees.
    """

    def __init__(self, double angle=45):

        super().__init__()
        if not 0 <= angle <= 90:
            raise RuntimeError("The cone angle must be between 0 and 90 degrees.")
        self.angle = angle

    cpdef list sample(self, int samples):
        cdef list results
        cdef int i

        results = []
        for i in range(samples):
            results.append(vector_cone(self.angle))
        return results


cdef class SphereUniform(VectorGenerator):
    """
    Generates a random vector on a unit sphere.
    """
    cpdef list sample(self, int samples):
        cdef list results
        cdef int i

        results = []
        for i in range(samples):
            results.append(vector_sphere())
        return results


cdef class HemisphereUniform(VectorGenerator):
    """
    Generates a random vector on a unit hemisphere.

    The hemisphere is aligned along the z-axis - the plane that forms the
    hemisphere base lies in the x-y plane.
    """
    cpdef list sample(self, int samples):
        cdef list results
        cdef int i

        results = []
        for i in range(samples):
            results.append(vector_hemisphere_uniform())
        return results


cdef class HemisphereCosine(VectorGenerator):
    """
    Generates a cosine-weighted random vector on a unit hemisphere.

    The hemisphere is aligned along the z-axis - the plane that forms the
    hemisphere base lies in the x-y plane.
    """
    cpdef list sample(self, int samples):
        cdef list results
        cdef int i

        results = []
        for i in range(samples):
            results.append(vector_hemisphere_cosine())
        return results
