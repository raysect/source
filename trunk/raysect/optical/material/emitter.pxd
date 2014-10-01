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

from raysect.optical.material.material cimport NullSurface, NullVolume
from raysect.core.math.vector cimport Vector
from raysect.core.math.point cimport Point
from raysect.optical.spectrum cimport Spectrum
from raysect.optical.spectralfunction cimport SpectralFunction

cdef class VolumeEmitterHomogeneous(NullSurface):

    cpdef Spectrum emission_function(self, Vector direction, Spectrum spectrum)


cdef class VolumeEmitterInhomogeneous(NullSurface):

    cdef double _step

    cpdef Spectrum emission_function(self, Point point, Vector direction, Spectrum spectrum)


cdef class UniformSurfaceEmitter(NullVolume):

    cdef:
        public SpectralFunction emission_spectrum
        public double scale


cdef class UniformVolumeEmitter(VolumeEmitterHomogeneous):

    cdef:
        public SpectralFunction emission_spectrum
        public double scale



cdef class Checkerboard(NullVolume):

    cdef:
        double _width
        double _rwidth
        public SpectralFunction emission_spectrum1
        public SpectralFunction emission_spectrum2
        public double scale1
        public double scale2

    cdef inline bint _flip(self, bint v, double p)