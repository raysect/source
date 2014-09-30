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
from raysect.core.math.utility cimport integrate

# Plank's constant * speed of light in a vacuum
DEF CONSTANT_HC = 1.9864456832693028e-25

# required by numpy c-api
import_array()

cdef class Spectrum(SampledSF):
    """
    radiance units: W/m^2/str/nm

    """

    cpdef bint is_black(self):

        cdef:
            int index
            double[::1] s_view

        # sanity check as users can modify the sample array
        if self.samples is None:

            raise ValueError("Cannot generate samples as the sample array is None.")

        if self.samples.shape[0] != self.num_samples:

            raise ValueError("Sample array length is inconsistent with num_samples.")

        s_view = self.samples
        for index in range(self.num_samples):

            if s_view[index] != 0.0:

                return False

        return True

    cpdef double total(self):

        # sanity check as users can modify the sample array
        if self.samples is None:

            raise ValueError("Cannot generate samples as the sample array is None.")

        if self.samples.shape[0] != self.num_samples:

            raise ValueError("Sample array length is inconsistent with num_samples.")

        # this calculation requires the wavelength array
        self._populate_wavelengths()

        return integrate(self._wavelengths, self.samples, self.min_wavelength, self.max_wavelength)


cdef Spectrum new_spectrum(double min_wavelength, double max_wavelength, int samples):

    cdef Spectrum v

    v = Spectrum.__new__(Spectrum)
    v._construct(min_wavelength, max_wavelength, samples, False)

    return v


cpdef double photon_energy(double wavelength):
    """
    Returns the energy of a photon with the specified wavelength.

    Arguements:
        wavelength: photon wavelength in nanometers

    Returns:
        photon energy in Joules
    """

    with cython.cdivision:

        # h * c / lambda
        return CONSTANT_HC / (wavelength * 1e-9)
