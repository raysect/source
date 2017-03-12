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

# TODO: these should really also return the probability of the given sample (pdf)

from libc.math cimport M_PI as PI
from raysect.core.math import Vector3D
from raysect.core.math cimport Point2D, new_point3d
from raysect.core.math.random cimport vector_hemisphere_uniform, vector_hemisphere_cosine, vector_cone, vector_sphere, point_disk, uniform


cdef class PointSampler:
    """
    Base class for an object that generates a list of Point3D objects.
    """

    def __call__(self, samples):
        """
        :param int samples: Number of points to generate.
        :rtype: list
        """

        return self.sample(samples)

    cpdef list sample(self, int samples):
        """
        :param int samples: Number of points to generate.
        :rtype: list
        """
        raise NotImplemented("The method sample() is not implemented for this point generator.")


cdef class VectorSampler:
    """
    Base class for an object that generates a list of Vector3D objects.
    """

    def __call__(self, samples):
        """
        :param int samples: Number of vectors to generate.
        :rtype: list
        """
        return self.sample(samples)

    cpdef list sample(self, int samples):
        """
        :param int samples: Number of vectors to generate.
        :rtype: list
        """
        raise NotImplemented("The method sample() is not implemented for this vector generator.")


cdef class DiskSampler(PointSampler):
    """
    Generates a random Point3D on a disk.

    :param double radius: The radius of the disk.
    """

    def __init__(self, radius=1):
        super().__init__()
        self.radius = radius

    cpdef list sample(self, int samples):

        cdef list results
        cdef int i
        cdef Point2D random_point

        results = []
        for i in range(samples):
            random_point = point_disk()
            results.append(new_point3d(random_point.x * self.radius, random_point.y * self.radius, 0))

        return results


cdef class RectangleSampler(PointSampler):
    """
    Generates a random Point3D on a rectangle.

    :param double width: The width of the rectangular sampling area of this observer.
    :param double height: The height of the rectangular sampling area of this observer.
    """

    def __init__(self, width=1, height=1):

        super().__init__()
        self.width = width
        self.height = height

    cpdef list sample(self, int samples):

        cdef list results
        cdef int i
        cdef double u, v
        cdef double width_offset = 0.5 * self.width
        cdef double height_offset = 0.5 * self.height

        results = []
        for i in range(samples):
            u = uniform() * self.width - width_offset
            v = uniform() * self.height - height_offset
            results.append(new_point3d(u, v, 0))

        return results


cdef class ConeSampler(VectorSampler):
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


cdef class SphereSampler(VectorSampler):
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


cdef class HemisphereUniformSampler(VectorSampler):
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


cdef class HemisphereCosineSampler(VectorSampler):
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
