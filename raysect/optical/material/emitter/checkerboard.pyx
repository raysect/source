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

from raysect.optical.colour import d65_white

from raysect.optical cimport World, Primitive, Ray, Spectrum, Point3D, AffineMatrix3D, Normal3D, Intersection
from libc.math cimport round, fabs
cimport cython


cdef class Checkerboard(NullVolume):
    """
    Isotropic checkerboard surface emitter

    Defines a plane of alternating squares of emission forming a checkerboard
    pattern. Useful in debugging and as a light source in test scenes.

    :param float width: The width of the squares in metres.
    :param SpectralFunction emission_spectrum1: Emission spectrum for square one.
    :param SpectralFunction emission_spectrum2: Emission spectrum for square two.
    :param float scale1: Intensity of square one emission.
    :param float scale2: Intensity of square two emission.

    .. code-block:: pycon

        >>> from raysect.primitive import Box
        >>> from raysect.optical import World, rotate, Point3D, d65_white
        >>> from raysect.optical.material import Checkerboard
        >>>
        >>> world = World()
        >>>
        >>> # checker board wall that acts as emitter
        >>> emitter = Box(lower=Point3D(-10, -10, 10), upper=Point3D(10, 10, 10.1), parent=world,
                          transform=rotate(45, 0, 0))
        >>> emitter.material=Checkerboard(4, d65_white, d65_white, 0.1, 2.0)
    """

    def __init__(self, double width=1.0, SpectralFunction emission_spectrum1=d65_white,
                 SpectralFunction emission_spectrum2=d65_white, double scale1=0.25, double scale2=0.5):

        super().__init__()
        self._width = width
        self._rwidth = 1.0 / width
        self.emission_spectrum1 = emission_spectrum1
        self.emission_spectrum2 = emission_spectrum2
        self.scale1 = scale1
        self.scale2 = scale2
        self.importance = 1.0

    @property
    def width(self):
        """
        The width of the squares in metres.

        :rtype: float
        """
        return self._width

    @width.setter
    @cython.cdivision(True)
    def width(self, double v):
        self._width = v
        self._rwidth = 1.0 / v

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef Spectrum evaluate_surface(self, World world, Ray ray, Primitive primitive, Point3D hit_point,
                                    bint exiting, Point3D inside_point, Point3D outside_point,
                                    Normal3D normal, AffineMatrix3D world_to_primitive, AffineMatrix3D primitive_to_world,
                                    Intersection intersection):

        cdef:
            Spectrum spectrum
            double[::1] emission
            bint v
            int index
            double scale

        v = False

        # generate check pattern
        v = self._flip(v, hit_point.x)
        v = self._flip(v, hit_point.y)
        v = self._flip(v, hit_point.z)

        # select emission
        spectrum = ray.new_spectrum()

        if v:
            emission = self.emission_spectrum1.sample_mv(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.bins)
            scale = self.scale1
        else:
            emission = self.emission_spectrum2.sample_mv(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.bins)
            scale = self.scale2

        for index in range(spectrum.bins):
            spectrum.samples_mv[index] = emission[index] * scale

        return spectrum

    @cython.cdivision(True)
    cdef bint _flip(self, bint v, double p) nogil:

        # round to avoid numerical precision issues (rounds to nearest nanometer)
        p = round(p * 1e9) / 1e9

        # generates check pattern from [0, inf]
        if fabs(self._rwidth * p) % 2 >= 1.0:
            v = not v

        # invert pattern for negative
        if p < 0:
            v = not v

        return v
