# cython: language_level=3

# Copyright (c) 2014-2023, Dr Alex Meakins, Raysect Project
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

from libc.math cimport M_PI, M_1_PI, sqrt, sin, cos, asin
from raysect.core.math cimport Vector3D, new_vector3d
from raysect.core.math.random cimport uniform

# TODO: add tests - idea: solve the lighting equation with a uniform emitting surface with each sampler and check the mean radiance is unity

DEF R_2_PI = 0.15915494309189535  # 1 / (2 * pi)
DEF R_4_PI = 0.07957747154594767  # 1 / (4 * pi)


cdef class SolidAngleSampler:
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


cdef class SphereSampler(SolidAngleSampler):
    """
    Generates a random vector on a unit sphere.

        >>> from raysect.core.math import SphereSampler
        >>>
        >>> sphere_sampler = SphereSampler()
        >>> sphere_sampler(2)
        [Vector3D(-0.03659868898144491, 0.24230159277890417, 0.9695104301149347),
         Vector3D(-0.6983609515217772, -0.6547708308112921, -0.28907981684698814)]
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


cdef class HemisphereUniformSampler(SolidAngleSampler):
    """
    Generates a random vector on a unit hemisphere.

    The hemisphere is aligned along the z-axis - the plane that forms the
    hemisphere base lies in the x-y plane.

        >>> from raysect.core.math import HemisphereUniformSampler
        >>>
        >>> sampler = HemisphereUniformSampler()
        >>> sampler(2)
        [Vector3D(-0.5555921819133177, -0.41159192618517343, 0.7224329821485018),
         Vector3D(0.03447410534618117, 0.33544044138689, 0.9414304256517041)]
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


cdef class HemisphereCosineSampler(SolidAngleSampler):
    """
    Generates a cosine-weighted random vector on a unit hemisphere.

    The hemisphere is aligned along the z-axis - the plane that forms the
    hemisphere base lies in the x-y plane.

        >>> from raysect.core.math import HemisphereCosineSampler
        >>>
        >>> sampler = HemisphereCosineSampler()
        >>> sampler(2)
        [Vector3D(0.18950017731212562, 0.4920026797683874, 0.8497193924463526),
         Vector3D(0.21900782218503353, 0.918767789013818, 0.32848336897387853)]
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


cdef class ConeUniformSampler(SolidAngleSampler):
    """
    Generates a uniform weighted random vector from a cone.

    The cone is aligned along the z-axis.
    
    :param angle: Angle of the cone in degrees (default=45).

    .. code-block:: pycon

        >>> from raysect.core.math import ConeUniformSampler
        >>> sampler = ConeUniformSampler(5)
        >>> sampler(2)
        [Vector3D(-0.032984782761108486, 0.02339453130328099, 0.9991820154562943),
         Vector3D(0.0246657314750599, 0.08269560820438482, 0.9962695609494988)]
    """

    def __init__(self, double angle=45):

        super().__init__()
        if not 0 < angle <= 90:
            raise ValueError("The cone angle must be between 0 and 90 degrees.")
        self.angle = angle
        self._angle_radians = angle / 180 * M_PI
        self._angle_cosine = cos(self._angle_radians)
        self._solid_angle = 2 * M_PI * (1 - self._angle_cosine)
        self._solid_angle_inv = 1 / self._solid_angle

    cpdef double pdf(self, Vector3D sample):
        if sample.z >= self._angle_cosine:
            return self._solid_angle_inv
        return 0.0

    cdef Vector3D sample(self):
        cdef double phi = 2.0 * M_PI * uniform()
        cdef double cos_theta = cos(self._angle_radians)
        cdef double z = uniform()*(1 - cos_theta) + cos_theta
        cdef double r = sqrt(max(0, 1.0 - z*z))
        cdef double x = r * cos(phi)
        cdef double y = r * sin(phi)
        return new_vector3d(x, y, z)

    cdef tuple sample_with_pdf(self):
        return self.sample(), self._solid_angle_inv
