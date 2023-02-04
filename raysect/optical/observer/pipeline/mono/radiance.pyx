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

from raysect.optical.spectralfunction cimport SpectralFunction
from raysect.optical.spectrum cimport Spectrum
cimport cython


_DEFAULT_PIPELINE_NAME = "Radiance Pipeline"
_DISPLAY_DPI = 100
_DISPLAY_SIZE = (512 / _DISPLAY_DPI, 512 / _DISPLAY_DPI)


cdef class RadiancePipeline0D(PowerPipeline0D):
    """
    A radiance pipeline for 0D observers.

    The raw spectrum collected by the observer is multiplied by a spectra filter
    and integrated to give to total radiance collected (W/str/m^2).

    The measured value and error are accessed at self.value.mean and self.value.error
    respectively.

    :param SpectralFunction filter: A filter function to be multiplied with the
     measured spectrum.
    :param bool accumulate:
    :param str name: User friendly name for this pipeline.
    """

    def __init__(self, SpectralFunction filter=None, bint accumulate=True, str name=None):
        name = name or _DEFAULT_PIPELINE_NAME
        super().__init__(filter=filter, accumulate=accumulate, name=name)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef PixelProcessor pixel_processor(self, int slice_id):
        return RadiancePixelProcessor(self._resampled_filter[slice_id])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef object finalise(self):

        cdef:
            double mean, variance
            int slice_id, samples

        mean = 0
        variance = 0
        samples = self._working_buffer.samples_mv[0]

        # combine sampled distributions for each spectral slice into the total
        for slice_id in range(self._working_buffer.length):
            mean += self._working_buffer.mean_mv[slice_id]
            variance += self._working_buffer.variance_mv[slice_id]
            if self._working_buffer.samples_mv[slice_id] != samples:
                raise ValueError("Samples for each spectral slice are inconsistent.")

        self.value.combine_samples(mean, variance, samples)

        if not self._quiet:
            print("{} - incident radiance: {:.4G} +/- {:.4G} W/str/m^2"
                  "".format(self.name, self.value.mean, self.value.error()))


cdef class RadiancePipeline1D(PowerPipeline1D):
    """
    A radiance pipeline for 1D observers.

    The raw spectrum collected at each pixel by the observer is multiplied by
    a spectral filter and integrated to give to total radiance collected at that
    pixel (W/str/m^2).

    The measured value and error for each pixel are accessed at self.frame.mean
    and self.frame.error respectively.

    :param SpectralFunction filter: A filter function to be multiplied with the
     measured spectrum.
    :param bool accumulate: Whether to accumulate samples with subsequent calls
      to observe() (default=True).
    :param str name: User friendly name for this pipeline.
    """

    def __init__(self, SpectralFunction filter=None, bint accumulate=True, str name=None):

        name = name or _DEFAULT_PIPELINE_NAME
        super().__init__(filter=filter, accumulate=accumulate, name=name)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef PixelProcessor pixel_processor(self, int pixel, int slice_id):
        return RadiancePixelProcessor(self._resampled_filter[slice_id])


cdef class RadiancePipeline2D(PowerPipeline2D):
    """
    A radiance pipeline for 2D observers.

    The raw spectrum collected at each pixel by the observer is multiplied by
    a spectral filter and integrated to give to total radiance collected at that
    pixel (W/str/m^2).

    The measured value and error for each pixel are accessed at self.frame.mean and self.frame.error
    respectively.

    :param SpectralFunction filter: A filter function to be multiplied with the
     measured spectrum.
    :param bool display_progress: Toggles the display of live render progress
      (default=True).
    :param float display_update_time: Time in seconds between preview display
      updates (default=15 seconds).
    :param bool accumulate: Whether to accumulate samples with subsequent calls
      to observe() (default=True).
    :param bool display_auto_exposure: Toggles the use of automatic exposure of
      final images (default=True).
    :param float display_black_point:
    :param float display_white_point:
    :param float display_unsaturated_fraction: Fraction of pixels that must not
      be saturated. Display values will be scaled to satisfy this value
      (default=1.0).
    :param float display_gamma:
    :param str name: User friendly name for this pipeline.
    """

    def __init__(self, SpectralFunction filter=None, bint display_progress=True,
                 double display_update_time=15, bint accumulate=True,
                 bint display_auto_exposure=True, double display_black_point=0.0, double display_white_point=1.0,
                 double display_unsaturated_fraction=1.0, display_gamma=2.2, str name=None):

        name = name or _DEFAULT_PIPELINE_NAME
        super().__init__(filter=filter, display_progress=display_progress, display_update_time=display_update_time,
                         accumulate=accumulate, display_auto_exposure=display_auto_exposure,
                         display_black_point=display_black_point, display_white_point=display_white_point,
                         display_unsaturated_fraction=display_unsaturated_fraction, display_gamma=display_gamma,
                         name=name)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef PixelProcessor pixel_processor(self, int x, int y, int slice_id):
        return RadiancePixelProcessor(self._resampled_filter[slice_id])


cdef class RadiancePixelProcessor(PixelProcessor):
    """
    PixelProcessor that converts each pixel's spectrum into total radiance by
    integrating over the spectrum.
    """

    def __init__(self, double[::1] filter):
        self.bin = StatsBin()
        self.filter = filter

    cpdef object reset(self):
        self.bin.clear()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object add_sample(self, Spectrum spectrum, double sensitivity):

        cdef:
            int index
            double total = 0
            Spectrum filtered

        # apply filter curve and integrate
        for index in range(spectrum.bins):
            total += spectrum.samples_mv[index] * self.filter[index] * spectrum.delta_wavelength

        self.bin.add_sample(total)

    cpdef tuple pack_results(self):
        return self.bin.mean, self.bin.variance

