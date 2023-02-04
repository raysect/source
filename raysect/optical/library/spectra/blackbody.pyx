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

from raysect.optical cimport NumericallyIntegratedSF, SpectralFunction, ConstantSF
from raysect.core.math.cython cimport clamp
from libc.math cimport exp
cimport cython


cdef class BlackBody(NumericallyIntegratedSF):
    """
    Generates a black body radiation spectrum.

    Implements Planck's Law to generate a black body spectrum for the given
    body temperature. The temperature must be supplied in Kelvin.

    An optional emissivity spectral function may be supplied. This function
    should return a value in the range [0, 1]. Values outside this range will
    be clamped.

    Averages and integrals are calculated using numerical integration, the step
    size of this integration can be controlled by the user. The default step
    size in 1 nm.

    :param temperature: The temperature in Kelvin.
    :param emissivity: Emissivity function (default=ConstantSF(1.0)).
    :param scale: Scales the spectra (default=1.0).
    :param sample_resolution: Numerical integration step size (default=1nm).
    """

    cdef readonly double temperature, scale
    cdef readonly SpectralFunction emissivity
    cdef double c1, c2

    def __init__(self, double temperature, SpectralFunction emissivity=None, double scale=1.0, double sample_resolution=1):

        super().__init__(sample_resolution)

        if temperature <= 0:
            raise ValueError("Temperature must be greater than zero.")

        if scale <= 0:
            raise ValueError("Scale factor must be greater than zero.")

        self.emissivity = emissivity or ConstantSF(1.0)
        self.temperature = temperature
        self.scale = scale

        self.c1 = self.scale * 1.19104295e20
        self.c2 = 1.43877735e7 / self.temperature

    def __getstate__(self):
        return self.temperature, self.scale, self.c1, self.c2, super().__getstate__()

    def __setstate__(self, state):
        self.temperature, self.scale, self.c1, self.c2, super_state = state
        super().__setstate__(super_state)

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @cython.cdivision(True)
    cpdef double function(self, double wavelength):
        """
        Planck's Law.
         
        :param wavelength: Wavelength in nm. 
        :return: Spectral radiance (W/m^2/str/nm).
        """

        # Planck's Law equation (wavelength in nm)
        cdef double emissivity = clamp(self.emissivity.evaluate(wavelength), 0, 1)
        return emissivity * self.c1 / (wavelength**5 * (exp(self.c2 / wavelength) - 1))












