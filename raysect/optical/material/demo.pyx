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

cimport cython
from raysect.core.math.affinematrix cimport AffineMatrix
from raysect.core.scenegraph.primitive cimport Primitive
from raysect.core.scenegraph.world cimport World
from raysect.optical.ray cimport Ray
from raysect.core.math.point cimport Point
from raysect.core.math.vector cimport Vector, new_vector
from raysect.optical.spectrum cimport Spectrum
from libc.math cimport exp, floor, fabs

cdef class GaussianBeam(VolumeEmitterInhomogeneous):

    def __init__(self, double power = 1, double sigma = 0.2, double step = 0.05):

        super().__init__(step)

        self.power = power
        self.sigma = sigma

    property sigma:

        def __get__(self):

            return self._sigma

        @cython.cdivision(True)
        def __set__(self, double sigma):

            self._sigma = sigma
            self._denominator = 0.5 / (sigma * sigma)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef Spectrum emission_function(self, Point point, Vector direction, Spectrum spectrum,
                                     World world, Ray ray, Primitive primitive,
                                     AffineMatrix to_local, AffineMatrix to_world):

        cdef:
            int index, count
            double scale
            double[::1] samples_view

        samples_view = spectrum.samples

        # gaussian beam uniform spectal power density
        scale = exp(-self._denominator * (point.x * point.x + point.y * point.y)) ** 4

        for index in range(spectrum.samples):

            samples_view[index] = self.power * scale / spectrum.samples

        index = int(floor(fabs(point.z * spectrum.samples)))
        if index < 0:

            index = 0

        if index >= spectrum.samples:

            index = spectrum.samples - 1

        samples_view[index] += self.power * scale

        return spectrum

