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

from libc.math cimport M_PI as PI, sqrt
from raysect.core.math cimport Point2D, new_point2d, Point3D, new_point3d, Vector3D, new_vector3d
from raysect.core.math.random cimport vector_hemisphere_uniform, vector_hemisphere_cosine, vector_cone_uniform, \
    vector_sphere, point_disk, uniform, vector_cone_cosine, point_square
from raysect.core.math.cython cimport barycentric_coords, barycentric_interpolation


cdef class SamplerPoint3D:
    """
    Base class for an object that generates a list of Point3D objects.
    """

    def __call__(self, object samples=None, bint pdf=False):
        """
        If samples is not provided, returns a single Point3D sample from
        the distribution. If samples is set to a value then a number of
        samples equal to the value specified is returned in a list.

        If pdf is set to True the Point3D sample is returned inside a tuple
        with its associated pdf value as the second element.

        :param int samples: Number of points to generate (default=None).
        :param bool pdf: Toggle for returning associated sample pdfs (default=False).
        :return: A Point3D or list of Point3D objects.
        """

        if samples:
            samples = int(samples)
            if samples <= 0:
                raise ValueError("Number of samples must be greater than 0.")
            if pdf:
                return self.samples_with_pdfs(samples)
            return self.samples(samples)
        else:
            if pdf:
                return self.sample_with_pdf()
            return self.sample()

    cpdef double pdf(self, Point3D sample):
        """
        Generates a pdf for a given sample value.

        :param Point3D sample: The sample point at which to get the pdf.
        :rtype: float
        """
        raise NotImplemented("The method pdf() is not implemented for this sampler.")

    cdef Point3D sample(self):
        """
        Generate a single sample.

        If the pdf is required please see sample_with_pdf().

        :rtype: Point3D
        """
        raise NotImplemented("The method sample() is not implemented for this sampler.")

    cdef tuple sample_with_pdf(self):
        """
        Generates a single sample with its associated pdf.

        Returns a tuple with the sample point as the first element and pdf value as
        the second element.

        Obtaining a sample with its pdf is generally more efficient than requesting the sample and then
        its pdf in a subsequent call since some of the calculation is common between the two steps.

        :rtype: tuple
        """
        raise NotImplemented("The method pdf() is not implemented for this sampler.")

    cdef list samples(self, int samples):
        """
        Generates a list of samples.

        If pdfs are required please see samples_with_pdfs().

        :param int samples: Number of points to generate.
        :rtype: list
        """
        raise NotImplemented("The method samples() is not implemented for this sampler.")

    cdef list samples_with_pdfs(self, int samples):
        """
        Generates a list of tuples containing samples and their associated pdfs.

        Each sample is a tuple with the sample point as the first element and pdf value as
        the second element.

        Obtaining samples with pdfs is generally more efficient than requesting samples and then
        the pdf in a subsequent call since some of the calculation is common between the two steps.

        :param int samples: Number of points to generate.
        :rtype: list
        """
        raise NotImplemented("The method samples_with_pdfs() is not implemented for this sampler.")


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


cdef class TriangleSampler(PointSampler):
    """
    Generates a random Point3D on a triangle.

    :param Point3D v1: Triangle vertex 1.
    :param Point3D v2: Triangle vertex 2.
    :param Point3D v3: Triangle vertex 3.
    """

    def __init__(self, Point3D v1, Point3D v2, Point3D v3):
        super().__init__()
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

    cpdef list sample(self, int samples):

        cdef list results
        cdef int i
        cdef double temp, alpha, beta, gamma

        results = []
        for i in range(samples):

            # generate barycentric coordinate
            temp = sqrt(uniform())
            alpha = 1 - temp
            beta = uniform() * temp
            gamma = 1 - alpha - beta

            # interpolate vertex coordinates to generate sample point coordinate
            results.append(
                new_point3d(
                    barycentric_interpolation(alpha, beta, gamma, self.v1.x, self.v2.x, self.v3.x),
                    barycentric_interpolation(alpha, beta, gamma, self.v1.y, self.v2.y, self.v3.y),
                    barycentric_interpolation(alpha, beta, gamma, self.v1.z, self.v2.z, self.v3.z)
                )
            )

        return results

    # TODO - implement me (1 / area of triangle)
    cpdef double pdf(self):
        raise NotImplementedError()


# cpdef Point2D point_disk():
#     """
#     Returns a random point on a disk of unit radius.
#
#     :rtype: Point2D
#     """
#
#     cdef double r = sqrt(uniform())
#     cdef double theta = 2.0 * PI * uniform()
#     return new_point2d(r * cos(theta), r * sin(theta))
#
#
# cpdef Point2D point_square():
#     """
#     Returns a random point on a square of unit radius.
#
#     :rtype: Point2D
#     """
#
#     return new_point2d(uniform(), uniform())

