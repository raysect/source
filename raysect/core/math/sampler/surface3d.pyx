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

from libc.math cimport M_PI as PI, sqrt, sin, cos

from raysect.core.math cimport Point3D, new_point3d, Vector3D
from raysect.core.math.random cimport uniform
from raysect.core.math.cython cimport barycentric_interpolation


cdef class SurfaceSampler3D:
    """
    Base class for an object that generates samples from a surface in 3D.
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
        :return: A Point3D, tuple or list.
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


# TODO - implement stratified sampling for samples and samples_with_pdfs
cdef class DiskSampler3D(SurfaceSampler3D):
    """
    Generates Point3D samples from a disk centred in the x-y plane.

    :param double radius: The radius of the disk in metres (default=1).

    .. code-block:: pycon

        >>> from raysect.core.math import DiskSampler3D
        >>>
        >>> disk_sampler = DiskSampler3D()
        >>> disk_sampler(2)
        [Point3D(-0.8755314944066419, -0.36748751614554004, 0.0),
         Point3D(-0.7515341075950953, 0.15368157833817775, 0.0)]
    """

    # TODO - validation
    def __init__(self, double radius=1.0):
        super().__init__()
        self.radius = radius
        self.area = PI * self.radius * self.radius
        self._area_inv = 1 / self.area

    cdef Point3D sample(self):
        cdef double r = sqrt(uniform()) * self.radius
        cdef double theta = 2.0 * PI * uniform()
        return new_point3d(r * cos(theta), r * sin(theta), 0)

    cdef tuple sample_with_pdf(self):
        return self.sample(), self._area_inv


# TODO - implement stratified sampling for samples and samples_with_pdfs
cdef class RectangleSampler3D(SurfaceSampler3D):
    """
    Generates Point3D samples from a rectangle centred in the x-y plane.

    :param double width: The width of the rectangle.
    :param double height: The height of the rectangle.

    .. code-block:: pycon

        >>> from raysect.core.math import RectangleSampler3D
        >>>
        >>> rectangle_sampler = RectangleSampler3D(width=3, height=3)
        >>> rectangle_sampler(2)
        [Point3D(0.8755185034767394, -1.4596971179451579, 0.0),
         Point3D(1.3514601271010727, 0.9710083493215418, 0.0)]
    """

    # TODO - validation
    def __init__(self, double width=1, double height=1):

        super().__init__()
        self.width = width
        self.height = height
        self.area = width * height
        self._area_inv = 1 / self.area
        self._width_offset = 0.5 * width
        self._height_offset = 0.5 * height

    cdef Point3D sample(self):
        return new_point3d(uniform() * self.width - self._width_offset, uniform() * self.height - self._height_offset, 0)

    cdef tuple sample_with_pdf(self):
        return self.sample(), self._area_inv


# TODO - implement stratified sampling for samples and samples_with_pdfs
cdef class TriangleSampler3D(SurfaceSampler3D):
    """
    Generates Point3D samples from a triangle in 3D space.

    :param Point3D v1: Triangle vertex 1.
    :param Point3D v2: Triangle vertex 2.
    :param Point3D v3: Triangle vertex 3.

    .. code-block:: pycon

        >>> from raysect.core.math import TriangleSampler3D
        >>>
        >>> tri_sampler = TriangleSampler3D(Point3D(0,0,0),
                                            Point3D(1,0,0),
                                            Point3D(1,1,0))
        >>> tri_sampler(2)
        [Point3D(0.9033819087428726, 0.053382913976399715, 0.0),
         Point3D(0.857350441035813, 0.4243360393025779, 0.0)]

    """

    # TODO - add validation
    def __init__(self, Point3D v1, Point3D v2, Point3D v3):
        super().__init__()
        self._v1 = v1
        self._v2 = v2
        self._v3 = v3

        self.area = self._calculate_area(v1, v2, v3)
        self._area_inv = 1 / self.area

    # TODO - please test me
    cdef double _calculate_area(self, Point3D v1, Point3D v2, Point3D v3):
        cdef Vector3D e1 = v1.vector_to(v2)
        cdef Vector3D e2 = v1.vector_to(v3)
        return 0.5 * e1.cross(e2).get_length()

    cdef Point3D sample(self):

        cdef double temp, alpha, beta, gamma

        # generate barycentric coordinate
        temp = sqrt(uniform())
        alpha = 1 - temp
        beta = uniform() * temp
        gamma = 1 - alpha - beta

        # interpolate vertex coordinates to generate sample point coordinate
        return new_point3d(
            barycentric_interpolation(alpha, beta, gamma, self._v1.x, self._v2.x, self._v3.x),
            barycentric_interpolation(alpha, beta, gamma, self._v1.y, self._v2.y, self._v3.y),
            barycentric_interpolation(alpha, beta, gamma, self._v1.z, self._v2.z, self._v3.z)
        )

    cdef tuple sample_with_pdf(self):
        return self.sample(), self._area_inv
