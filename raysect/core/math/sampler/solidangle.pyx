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

from libc.math cimport M_PI, M_1_PI, sqrt, sin, cos
from raysect.core.math cimport Point2D, new_point2d, Point3D, new_point3d, Vector3D, new_vector3d
from raysect.core.math.random cimport uniform
from raysect.core.math.cython cimport barycentric_coords, barycentric_interpolation

DEF R_2_PI = 0.15915494309189535  # 1 / (2 * pi)
DEF R_4_PI = 0.07957747154594767  # 1 / (4 * pi)


cdef class SamplerSolidAngle:
    """
    Base class for an object that generates samples over a solid angle.
    """

    def __call__(self, object samples=None, bint pdf=False):
        """
        If samples is not provided, returns a single Vector3D sample from
        the distribution. If samples is set to a value then a number of
        samples equal to the value specified is returned in a list.

        If pdf is set to True the Vector3D sample is returned inside a tuple
        with its associated pdf value as the second element.

        :param int samples: Number of points to generate (default=None).
        :param bool pdf: Toggle for returning associated sample pdfs (default=False).
        :return: A Vector3D, tuple or list of Vector3D objects.
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

    cpdef double pdf(self, Vector3D sample):
        """
        Generates a pdf for a given sample value.
        
        Vectors *must* be normalised.

        :param Vector3D sample: The sample point at which to get the pdf.
        :rtype: float
        """
        raise NotImplemented("The method pdf() is not implemented for this sampler.")

    cdef Vector3D sample(self):
        """
        Generate a single sample.

        If the pdf is required please see sample_with_pdf().

        :rtype: Vector3D
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

        cdef list results
        cdef int i

        results = []
        for i in range(samples):
            results.append(self.sample())
        return results

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

        cdef list results
        cdef int i

        results = []
        for i in range(samples):
            results.append(self.sample_with_pdf())
        return results


cdef class SphereSampler(SamplerSolidAngle):
    """
    Generates a random vector on a unit sphere.
    """

    cpdef double pdf(self, Vector3D sample):
        return R_4_PI

    cdef Vector3D sample(self):
        cdef double z = 1.0 - 2.0 * uniform()
        cdef double r = sqrt(max(0, 1.0 - z*z))
        cdef double phi = 2.0 * M_PI * uniform()
        cdef double x = r * cos(phi)
        cdef double y = r * sin(phi)
        return new_vector3d(x, y, z)

    cdef tuple sample_with_pdf(self):
        return self.sample(), R_4_PI


cdef class HemisphereUniformSampler(SamplerSolidAngle):
    """
    Generates a random vector on a unit hemisphere.

    The hemisphere is aligned along the z-axis - the plane that forms the
    hemisphere base lies in the x-y plane.
    """

    cpdef double pdf(self, Vector3D sample):
        if sample.z >= 0.0:
            return R_2_PI
        return 0.0

    cdef Vector3D sample(self):
        cdef double z = uniform()
        cdef double r = sqrt(max(0, 1.0 - z*z))
        cdef double phi = 2.0 * M_PI * uniform()
        cdef double x = r * cos(phi)
        cdef double y = r * sin(phi)
        return new_vector3d(x, y, z)

    cdef tuple sample_with_pdf(self):
        return self.sample(), R_2_PI


cdef class HemisphereCosineSampler(SamplerSolidAngle):
    """
    Generates a cosine-weighted random vector on a unit hemisphere.

    The hemisphere is aligned along the z-axis - the plane that forms the
    hemisphere base lies in the x-y plane.
    """

    cpdef double pdf(self, Vector3D sample):
        if sample.z >= 0.0:
            return  M_1_PI * sample.z
        return 0.0

    cdef Vector3D sample(self):
        cdef double r = sqrt(uniform())
        cdef double phi = 2.0 * M_PI * uniform()
        cdef double x = r * cos(phi)
        cdef double y = r * sin(phi)
        return new_vector3d(x, y, sqrt(max(0, 1.0 - x*x - y*y)))

    cdef tuple sample_with_pdf(self):
        cdef Vector3D sample = self.sample()
        return sample, M_1_PI * sample.z


# cdef class ConeUniformSampler(VectorSampler):
#     """
#     Generates a list of random unit Vector3D objects inside a cone.
#
#     The cone is aligned along the z-axis.
#
#     :param angle: Angle of the cone in degrees.
#     """
#
#     def __init__(self, double angle=45):
#
#         super().__init__()
#         if not 0 <= angle <= 90:
#             raise RuntimeError("The cone angle must be between 0 and 90 degrees.")
#         self.angle = angle
#
#     cpdef list sample(self, int samples):
#         cdef list results
#         cdef int i
#
#         results = []
#         for i in range(samples):
#             results.append(vector_cone_uniform(self.angle))
#         return results
#
#
# cdef class ConeCosineSampler(SamplerSolidAngle):
#     """
#     Generates a list of random unit Vector3D objects inside a cone with cosine weighting.
#
#     The cone is aligned along the z-axis.
#
#     :param angle: Angle of the cone in degrees.
#     """
#
#     def __init__(self, double angle=45):
#
#         super().__init__()
#         if not 0 <= angle <= 90:
#             raise RuntimeError("The cone angle must be between 0 and 90 degrees.")
#         self.angle = angle
#
#     cpdef list sample(self, int samples):
#         cdef list results
#         cdef int i
#
#         results = []
#         for i in range(samples):
#             results.append(vector_cone_cosine(self.angle))
#         return results
#
#
# cdef class QuadVectorSampler(SamplerSolidAngle):
#     """
#     Generates a list of random unit Vector3D objects sampled on a quadrangle.
#
#     Useful for sub-sampling pixels on non-physical cameras where only the central pixel
#     vectors are available. The vectors at each corner of the quad are supplied. The sampler
#     generates a random sample point on the quad, linear vector interpolation is used
#     between the corners.
#
#     .. Warning::
#         For best results, the vectors at each corner should be close in angle. Results will
#         be not be sensible for cases where vectors have large angle separation
#         (i.e. > 90 degrees).
#
#     :param Vector3D v1: Vector in lower left corner.
#     :param Vector3D v2: Vector in upper left corner.
#     :param Vector3D v3: Vector in upper right corner.
#     :param Vector3D v4: Vector in lower right corner.
#     """
#
#     def __init__(self, Vector3D v1, Vector3D v2, Vector3D v3, Vector3D v4):
#
#         super().__init__()
#
#         self.v1 = v1.normalise()
#         self.v2 = v2.normalise()
#         self.v3 = v3.normalise()
#         self.v4 = v4.normalise()
#
#     cpdef list sample(self, int samples):
#         cdef:
#             list results
#             int i
#             Point2D sample_point
#             double alpha, beta, gamma
#
#         results = []
#         for i in range(samples):
#
#             # Generate new sample point in unit square
#             sample_point = point_square()
#
#             # Test if point is in upper triangle
#             if sample_point.y > sample_point.x:
#                 # coordinates are p1 (0, 0), p2 (0, 1), p3 (1, 1)
#                 barycentric_coords(0, 0, 0, 1, 1, 1, sample_point.x, sample_point.y, &alpha, &beta, &gamma)
#                 sample_vector = self.v1.mul(alpha) + self.v2.mul(beta) + self.v3.mul(gamma)
#                 results.append(sample_vector.normalise())
#
#             # Point must be in lower triangle
#             else:
#                 # coordinates are p3 (1, 1), p4 (1, 0), p1 (0, 0)
#                 barycentric_coords(1, 1, 1, 0, 0, 0, sample_point.x, sample_point.y, &alpha, &beta, &gamma)
#                 sample_vector = self.v3.mul(alpha) + self.v4.mul(beta) + self.v1.mul(gamma)
#                 results.append(sample_vector.normalise())
#
#         return results




# cpdef Vector3D vector_cone_uniform(double theta):
#     """
#     Generates a random vector in a cone along the z-axis.
#
#     The angle of the cone is specified with the theta parameter. For speed, no
#     checks are performs on the theta parameter, it is up to user to ensure the
#     angle is sensible.
#
#     :param float theta: An angle between 0 and 90 degrees.
#     :returns: A random Vector3D in the cone defined by theta.
#     :rtype: Vector3D
#     """
#
#     theta *= 0.017453292519943295 # PI / 180
#     cdef double phi = 2.0 * PI * uniform()
#     cdef double cos_theta = cos(theta)
#     cdef double z = uniform()*(1 - cos_theta) + cos_theta
#     cdef double r = sqrt(max(0, 1.0 - z*z))
#     cdef double x = r * cos(phi)
#     cdef double y = r * sin(phi)
#     return new_vector3d(x, y, z)
#
#
# cpdef Vector3D vector_cone_cosine(double theta):
#     """
#     Generates a cosine-weighted random vector on a cone along the z-axis.
#
#     The angle of the cone is specified with the theta parameter. For speed, no
#     checks are performs on the theta parameter, it is up to user to ensure the
#     angle is sensible.
#
#     :param float theta: An angle between 0 and 90 degrees.
#     :returns: A random Vector3D in the cone defined by theta.
#     :rtype: Vector3D
#     """
#
#     theta *= 0.017453292519943295 # PI / 180
#     cdef double r_max_scaled = asin(theta)
#     cdef double r = sqrt(uniform()) * r_max_scaled
#     cdef double phi = 2.0 * PI * uniform()
#     cdef double x = r * cos(phi)
#     cdef double y = r * sin(phi)
#     return new_vector3d(x, y, sqrt(max(0, 1.0 - x*x - y*y)))



# cdef class SphereSampler(SamplerSolidAngle):
#     """
#     Generates a random vector on a unit sphere.
#     """
#
#     cpdef double pdf(self, Vector3D sample):
#         return 0.07957747154594767  # 1 / (4 * pi)
#
#     cdef Vector3D sample(self):
#         pass
#
#     cdef tuple sample_with_pdf(self):
#         pass
#
#     cdef list samples(self, int samples):
#         pass
#
#     cdef list samples_with_pdfs(self, int samples):
#         pass