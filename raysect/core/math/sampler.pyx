# cython: language_level=3

# Copyright (c) 2014-17, Dr Alex Meakins, Raysect Project
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
from raysect.core.math cimport Point2D, new_point2d, Point3D, new_point3d, Vector3D, new_vector3d
from raysect.core.math.random cimport vector_hemisphere_uniform, vector_hemisphere_cosine, vector_cone_uniform, \
    vector_sphere, point_disk, uniform, vector_cone_cosine, point_square
from raysect.core.math.cython cimport barycentric_coords


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


cdef class ConeUniformSampler(VectorSampler):
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
            results.append(vector_cone_uniform(self.angle))
        return results


cdef class ConeCosineSampler(VectorSampler):
    """
    Generates a list of random unit Vector3D objects inside a cone with cosine weighting.

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
            results.append(vector_cone_cosine(self.angle))
        return results


cdef class QuadVectorSampler(VectorSampler):
    """
    Generates a list of random unit Vector3D objects sampled on a quadrangle.

    Useful for sub-sampling pixels on non-physical cameras where only the central pixel
    vectors are available. The vectors at each corner of the quad are supplied. The sampler
    generates a random sample point on the quad, linear vector interpolation is used
    between the corners.

    .. Warning::
        For best results, the vectors at each corner should be close in angle. Results will
        be not be sensible for cases where vectors have large angle separation
        (i.e. > 90 degrees).

    :param Vector3D v1: Vector in lower left corner.
    :param Vector3D v2: Vector in upper left corner.
    :param Vector3D v3: Vector in upper right corner.
    :param Vector3D v4: Vector in lower right corner.
    """

    def __init__(self, Vector3D v1, Vector3D v2, Vector3D v3, Vector3D v4):

        super().__init__()

        self.v1 = v1.normalise()
        self.v2 = v2.normalise()
        self.v3 = v3.normalise()
        self.v4 = v4.normalise()

    cpdef list sample(self, int samples):
        cdef:
            list results
            int i
            Point2D sample_point
            double alpha, beta, gamma

        results = []
        for i in range(samples):

            # Generate new sample point in unit square
            sample_point = point_square()

            # Test if point is in upper triangle
            if sample_point.y > sample_point.x:
                # coordinates are p1 (0, 0), p2 (0, 1), p3 (1, 1)
                barycentric_coords(0, 0, 0, 1, 1, 1, sample_point.x, sample_point.y, &alpha, &beta, &gamma)
                sample_vector = self.v1.mul(alpha) + self.v2.mul(beta) + self.v3.mul(gamma)
                results.append(sample_vector.normalise())

            # Point must be in lower triangle
            else:
                # coordinates are p3 (1, 1), p4 (1, 0), p1 (0, 0)
                barycentric_coords(1, 1, 1, 0, 0, 0, sample_point.x, sample_point.y, &alpha, &beta, &gamma)
                sample_vector = self.v3.mul(alpha) + self.v4.mul(beta) + self.v1.mul(gamma)
                results.append(sample_vector.normalise())

        return results

