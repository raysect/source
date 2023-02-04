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

from raysect.optical cimport SpectralFunction, NumericallyIntegratedSF, Spectrum, Vector3D, Normal3D
from raysect.optical.material cimport Material

cdef class Sellmeier(NumericallyIntegratedSF):

    cdef:
        double b1, b2, b3
        double c1, c2, c3


cdef class Dielectric(Material):

    cdef:
        public SpectralFunction index
        public SpectralFunction external_index
        public SpectralFunction transmission

    cdef (double, double, double, double, double) _fresnel(self, double ci, double ct, double ni, double nt) nogil

    cdef void _apply_mueller_reflection_tir(self, Spectrum spectrum, double ci, double gamma)

    cdef void _apply_mueller_reflection_non_tir(self, Spectrum spectrum, double p, double s)

    cdef void _apply_mueller_transmission(self, Spectrum spectrum, double p, double s)

    cdef void _apply_stokes_rotation(self, Spectrum spectrum, double theta)

    cdef double _polarisation_frame_angle(self, Vector3D direction, Vector3D ray_orientation, Vector3D interface_orientation)