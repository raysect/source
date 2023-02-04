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

cimport numpy as np

from raysect.optical.observer.base cimport PixelProcessor, Pipeline0D, Pipeline1D, Pipeline2D
from raysect.core.math cimport StatsArray1D, StatsArray2D, StatsArray3D


cdef class SpectralPowerPipeline0D(Pipeline0D):

    cdef:
        public str name
        public bint accumulate
        readonly StatsArray1D samples
        list _spectral_slices
        readonly int bins
        readonly double min_wavelength, max_wavelength, delta_wavelength
        readonly np.ndarray wavelengths
        public bint display_progress
        object _display_figure
        bint _quiet


cdef class SpectralPowerPipeline1D(Pipeline1D):

    cdef:
        public str name
        public bint accumulate
        readonly StatsArray2D frame
        int _pixels
        int _samples
        list _spectral_slices
        readonly int bins
        readonly double min_wavelength, max_wavelength, delta_wavelength
        readonly np.ndarray wavelengths


cdef class SpectralPowerPipeline2D(Pipeline2D):

    cdef:
        public str name
        public bint accumulate
        readonly StatsArray3D frame
        tuple _pixels
        int _samples
        list _spectral_slices
        readonly int bins
        readonly double min_wavelength, max_wavelength, delta_wavelength
        readonly np.ndarray wavelengths


cdef class SpectralPowerPixelProcessor(PixelProcessor):

    cdef StatsArray1D bins
