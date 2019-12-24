# Copyright (c) 2014-2018, Dr Alex Meakins, Raysect Project
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

import numpy as np
from random import shuffle

from raysect.optical.observer.base cimport FrameSampler1D
from raysect.optical.observer.pipeline cimport RadiancePipeline1D, PowerPipeline1D, SpectralRadiancePipeline1D, SpectralPowerPipeline1D
from raysect.core.math cimport StatsArray1D, StatsArray2D
cimport numpy as np
cimport cython


cdef class FullFrameSampler1D(FrameSampler1D):

    cpdef list generate_tasks(self, int pixels):

        cdef:
            list tasks
            int pixel

        tasks = []
        for pixel in range(pixels):
            tasks.append((pixel, ))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks


cdef class MonoAdaptiveSampler1D(FrameSampler1D):
    """
    FrameSampler that dynamically adjusts a camera's pixel samples based on the noise
    level in each pixel's power value.

    Pixels that have high noise levels will receive extra samples until the desired
    noise threshold is achieve across the whole image.

    :param PowerPipeline1D pipeline: The specific power pipeline to use for feedback control.
    :param float fraction: The fraction of frame pixels to receive extra sampling
      (default=0.2).
    :param float ratio: The maximum allowable ratio between the maximum and minimum number of
      samples obtained for the pixels of the same observer (default=10).
      The sampler will generate additional tasks for pixels with the least number of samples
      in order to keep this ratio below a given value.
    :param int min_samples: Minimum number of pixel samples across the image before
      turning on adaptive sampling (default=1000).
    :param double cutoff: Normalised noise threshold at which extra sampling will be aborted and
      rendering will complete (default=0.0). The standard error is normalised to 1 so that a
      cutoff of 0.01 corresponds to 1% standard error.
    """

    cdef:
        PowerPipeline1D _pipeline
        double _fraction, _ratio, _cutoff
        int _min_samples

    def __init__(self, object pipeline, double fraction=0.2, double ratio=10.0, int min_samples=1000, double cutoff=0.0):

        self.pipeline = pipeline
        self.fraction = fraction
        self.ratio = ratio
        self.min_samples = min_samples
        self.cutoff = cutoff

    @property
    def pipeline(self):
        return self._pipeline

    @pipeline.setter
    def pipeline(self, value):
        if not isinstance(value, (PowerPipeline1D, RadiancePipeline1D)):
            raise TypeError('Sampler only compatible with PowerPipeLine1D or RadiancePipeline1D pipelines.')
        self._pipeline = value

    @property
    def fraction(self):
        return self._fraction

    @fraction.setter
    def fraction(self, value):
        if value <= 0 or value > 1.:
            raise ValueError("fraction must be in the range (0, 1]")
        self._fraction = value

    @property
    def ratio(self):
        return self._ratio

    @ratio.setter
    def ratio(self, value):
        if value < 1.:
            raise ValueError("ratio must be >= 1")
        self._ratio = value

    @property
    def min_samples(self):
        return self._min_samples

    @min_samples.setter
    def min_samples(self, value):
        if value < 1:
            raise ValueError("min_samples must be >= 1")
        self._min_samples = value

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value):
        if value < 0 or value > 1.:
            raise ValueError("cutoff must be in the range [0, 1]")
        self._cutoff = value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef list generate_tasks(self, int pixels):

        cdef:
            StatsArray1D frame
            int pixel, min_samples
            np.ndarray normalised
            double[::1] error, normalised_mv
            double percentile_error
            list tasks

        frame = self.pipeline.frame
        if frame is None:
            # no frame data available, generate tasks for the full frame
            return self._full_frame(pixels)

        # sanity check
        if pixels != frame.length:
            raise ValueError('The pixel geometry passed to the frame sampler is inconsistent with the pipeline frame size.')

        min_samples = max(self.min_samples, <int>(frame.samples.max() / self.ratio))
        error = frame.errors()
        normalised = np.zeros(frame.length)
        normalised_mv = normalised

        # calculated normalised standard error
        for pixel in range(frame.length):
            if frame.mean_mv[pixel] <= 0:
                normalised_mv[pixel] = 0
            else:
                normalised_mv[pixel] = error[pixel] / frame.mean_mv[pixel]

        # locate error value corresponding to fraction of frame to process
        percentile_error = np.percentile(normalised, (1 - self.fraction) * 100)

        # build tasks
        tasks = []
        for pixel in range(frame.length):
            if frame.samples_mv[pixel] < min_samples or normalised_mv[pixel] > max(self.cutoff, percentile_error):
                tasks.append((pixel, ))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks

    cpdef list _full_frame(self, int pixels):

        cdef:
            list tasks
            int pixel

        tasks = []
        for pixel in range(pixels):
            tasks.append((pixel, ))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks


cdef class SpectralAdaptiveSampler1D(FrameSampler1D):
    """
    FrameSampler that dynamically adjusts a camera's pixel samples based on the noise
    level in each pixel's power value.

    Pixels that have high noise levels will receive extra samples until the desired
    noise threshold is achieve across the whole image.

    :param SpectralPowerPipeline1D pipeline: The specific spectral power pipeline to use
      for feedback control.
    :param float fraction: The fraction of frame pixels to receive extra sampling
      (default=0.2).
    :param float ratio: The maximum allowable ratio between the maximum and minimum number of
      samples obtained for the pixels of the same observer (default=10).
      The sampler will generate additional tasks for pixels with the least number of samples
      in order to keep this ratio below a given value.
    :param int min_samples: Minimum number of pixel samples across the image before
      turning on adaptive sampling (default=1000).
    :param double cutoff: Normalised noise threshold at which extra sampling will be aborted and
      rendering will complete (default=0.0). The standard error is normalised to 1 so that a
      cutoff of 0.01 corresponds to 1% standard error.
    :param str reduction_method: A method for obtaining spectral-average value of normalised
      error of a pixel from spectral array of errors (default='percentile').
       - `reduction_method='weighted'`: the error of a pixel is calculated as power-weighted
         average of the spectral errors,
       - `reduction_method='mean'`: the error of a pixel is calculated as a mean
         of the spectral errors excluding spectral bins with zero power,
       - `reduction_method='percentile'`: the error of a pixel is calculated as a user-defined
         percentile of the spectral errors excluding spectral bins with zero power.
    :param double percentile: If `reduction_method='percentile'`, defines the percentile of
      statistical errors of spectral bins with non-zero power, which is used to calculate
      normalised error of a pixel (default=100). If `percentile=x`, extra sampling will be aborted
      if `x`% of spectral bins of each pixel have normalised error lower than `cutoff`.
    """

    cdef:
        SpectralPowerPipeline1D _pipeline
        str _reduction_method
        double _fraction, _ratio, _cutoff, _percentile
        int _min_samples

    def __init__(self, object pipeline, double fraction=0.2, double ratio=10.0, int min_samples=1000, double cutoff=0.0,
                 str reduction_method='percentile', double percentile=100.):

        self.pipeline = pipeline
        self.fraction = fraction
        self.ratio = ratio
        self.min_samples = min_samples
        self.cutoff = cutoff
        self.reduction_method = reduction_method
        self.percentile = percentile

    @property
    def pipeline(self):
        return self._pipeline

    @pipeline.setter
    def pipeline(self, value):
        if not isinstance(value, (SpectralPowerPipeline1D, SpectralRadiancePipeline1D)):
            raise TypeError('Sampler only compatible with SpectralPowerPipeLine1D or SpectralRadiancePipeline1D pipelines.')
        self._pipeline = value

    @property
    def fraction(self):
        return self._fraction

    @fraction.setter
    def fraction(self, value):
        if value <= 0 or value > 1.:
            raise ValueError("fraction must be in the range (0, 1]")
        self._fraction = value

    @property
    def ratio(self):
        return self._ratio

    @ratio.setter
    def ratio(self, value):
        if value < 1.:
            raise ValueError("ratio must be >= 1")
        self._ratio = value

    @property
    def min_samples(self):
        return self._min_samples

    @min_samples.setter
    def min_samples(self, value):
        if value < 1:
            raise ValueError("min_samples must be >= 1")
        self._min_samples = value

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value):
        if value < 0 or value > 1.:
            raise ValueError("cutoff must be in the range [0, 1]")
        self._cutoff = value

    @property
    def reduction_method(self):
        return self._reduction_method

    @reduction_method.setter
    def reduction_method(self, value):
        if value not in {'weighted', 'mean', 'percentile'}:
            raise ValueError("reduction_method must be 'weighted', 'mean' or 'percentile'")
        self._reduction_method = value

    @property
    def percentile(self):
        return self._percentile

    @percentile.setter
    def percentile(self, value):
        if value < 0 or value > 100.:
            raise ValueError("Percentiles must be in the range [0, 100]")
        self._percentile = value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef list generate_tasks(self, int pixels):

        cdef:
            StatsArray2D frame
            int pixel, sbin, count, min_samples
            np.ndarray normalised, spectral_normalised
            double[::1] normalised_mv, spectral_normalised_mv
            double[:, ::1] error
            double percentile_error, pixel_power
            list tasks

        frame = self.pipeline.frame
        if frame is None:
            # no frame data available, generate tasks for the full frame
            return self._full_frame(pixels)

        # sanity check
        if pixels != frame.nx:
            raise ValueError('The pixel geometry passed to the frame sampler is inconsistent with the pipeline frame size.')

        min_samples = max(self.min_samples, <int>(frame.samples.max() / self.ratio))
        error = frame.errors()
        normalised = np.zeros(frame.nx)
        normalised_mv = normalised

        # calculated normalised standard error
        if self.reduction_method == 'weighted':
            for pixel in range(frame.nx):
                pixel_power = 0
                for sbin in range(frame.ny):
                    if frame.mean_mv[pixel, sbin] > 0:
                        normalised_mv[pixel] += error[pixel, sbin]
                        pixel_power += frame.mean_mv[pixel, sbin]
                if pixel_power:
                    normalised_mv[pixel] /= pixel_power

        elif self.reduction_method == 'mean':
            for pixel in range(frame.nx):
                count = 0
                for sbin in range(frame.ny):
                    if frame.mean_mv[pixel, sbin] > 0:
                        normalised_mv[pixel] += error[pixel, sbin] / frame.mean_mv[pixel, sbin]
                        count += 1
                if count:
                    normalised_mv[pixel] /= count
        else:
            spectral_normalised = np.zeros(frame.ny)
            spectral_normalised_mv = spectral_normalised
            for pixel in range(frame.nx):
                count = 0
                for sbin in range(frame.ny):
                    if frame.mean_mv[pixel, sbin] > 0:
                        spectral_normalised_mv[count] = error[pixel, sbin] / frame.mean_mv[pixel, sbin]
                        count += 1
                if count:
                    normalised_mv[pixel] = np.percentile(spectral_normalised[:count], self.percentile)

        # locate error value corresponding to fraction of frame to process
        percentile_error = np.percentile(normalised, (1 - self.fraction) * 100)

        # build tasks
        tasks = []
        for pixel in range(frame.nx):
            if frame.samples[pixel].min() < min_samples or normalised_mv[pixel] > max(self.cutoff, percentile_error):
                tasks.append((pixel, ))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks

    cpdef list _full_frame(self, int pixels):

        cdef:
            list tasks
            int pixel

        tasks = []
        for pixel in range(pixels):
            tasks.append((pixel, ))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks
