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

from raysect.optical.observer.base cimport FrameSampler2D
from raysect.optical.observer.pipeline cimport RGBPipeline2D, RadiancePipeline2D, PowerPipeline2D, SpectralRadiancePipeline2D, SpectralPowerPipeline2D
from raysect.core.math cimport StatsArray1D, StatsArray2D, StatsArray3D
from timeit import default_timer as timer
cimport numpy as np
cimport cython


cdef class FullFrameSampler2D(FrameSampler2D):
    """ Evenly samples the full 2D frame. """

    cpdef list generate_tasks(self, tuple pixels):

        cdef:
            list tasks
            int nx, ny

        tasks = []
        nx, ny = pixels
        for iy in range(ny):
            for ix in range(nx):
                tasks.append((ix, iy))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks


cdef class MaskedFrameSampler2D(FrameSampler2D):
    """
    Evenly samples the masked 2D frame.

    :param np.ndarray mask: The image mask array.
    """

    cdef:
        np.ndarray _mask

    def __init__(self, np.ndarray mask):

        self.mask = mask

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, np.ndarray value):
        if value.ndim != 2:
            raise ValueError("Mask must be a 2D array")
        if value.dtype not in (np.int, np.int32, np.int64, np.bool):
            raise TypeError("Mask must be an array of booleans or integers")
        self._mask = value

    cpdef list generate_tasks(self, tuple pixels):

        cdef:
            list tasks
            int nx, ny

        if pixels != (self.mask.shape[0], self.mask.shape[1]):
            raise ValueError('The pixel geometry passed to the frame sampler is inconsistent with the mask frame size.')

        tasks = []
        nx, ny = pixels
        for iy in range(ny):
            for ix in range(nx):
                if self.mask[ix, iy]:
                    tasks.append((ix, iy))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks


cdef class MonoAdaptiveSampler2D(FrameSampler2D):
    """
    FrameSampler that dynamically adjusts a camera's pixel samples based on the noise
    level in each pixel's power value.

    Pixels that have high noise levels will receive extra samples until the desired
    noise threshold is achieve across the whole image.

    :param PowerPipeline2D pipeline: The specific power pipeline to use for feedback control.
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
        PowerPipeline2D _pipeline
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
        if not isinstance(value, (PowerPipeline2D, RadiancePipeline2D)):
            raise TypeError('Sampler only compatible with PowerPipeLine2D or RadiancePipeline2D pipelines.')
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
    cpdef list generate_tasks(self, tuple pixels):

        cdef:
            StatsArray2D frame
            int x, y, min_samples
            np.ndarray normalised
            double[:, ::1] error, normalised_mv
            double percentile_error
            list tasks

        frame = self.pipeline.frame
        if frame is None:
            # no frame data available, generate tasks for the full frame
            return self._full_frame(pixels)

        # sanity check
        if pixels != frame.shape:
            raise ValueError('The pixel geometry passed to the frame sampler is inconsistent with the pipeline frame size.')

        min_samples = max(self.min_samples, <int>(frame.samples.max() / self.ratio))
        error = frame.errors()
        normalised = np.zeros((frame.nx, frame.ny))
        normalised_mv = normalised

        # calculated normalised standard error
        for x in range(frame.nx):
            for y in range(frame.ny):
                if frame.mean_mv[x, y] <= 0:
                    normalised_mv[x, y] = 0
                else:
                    normalised_mv[x, y] = error[x, y] / frame.mean_mv[x, y]

        # locate error value corresponding to fraction of frame to process
        percentile_error = np.percentile(normalised, (1 - self.fraction) * 100)

        # build tasks
        tasks = []
        for x in range(frame.nx):
            for y in range(frame.ny):
                if frame.samples_mv[x, y] < min_samples or normalised_mv[x, y] > max(self.cutoff, percentile_error):
                    tasks.append((x, y))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks

    cpdef list _full_frame(self, tuple pixels):

        cdef:
            list tasks
            int nx, ny, x, y

        tasks = []
        nx, ny = pixels
        for x in range(nx):
            for y in range(ny):
                tasks.append((x, y))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks


cdef class MaskedMonoAdaptiveSampler2D(FrameSampler2D):
    """
    A masked FrameSampler that dynamically adjusts a camera's pixel samples based on the noise
    level in each pixel's power value.

    Pixels that have high noise levels will receive extra samples until the desired
    noise threshold is achieve across the masked image.

    :param PowerPipeline2D pipeline: The specific power pipeline to use for feedback control.
    :param np.ndarray mask: The image mask array.
    :param int min_samples: Minimum number of pixel samples across the image before
      turning on adaptive sampling (default=1000).
    :param double cutoff: Normalised noise threshold at which extra sampling will be aborted and
      rendering will complete (default=0.0). The standard error is normalised to 1 so that a
      cutoff of 0.01 corresponds to 1% standard error.
    """

    cdef:
        PowerPipeline2D _pipeline
        double _cutoff
        int _min_samples
        np.ndarray _mask

    def __init__(self, object pipeline, np.ndarray mask, int min_samples=1000, double cutoff=0.0):

        self.pipeline = pipeline
        self.min_samples = min_samples
        self.cutoff = cutoff
        self.mask = mask

    @property
    def pipeline(self):
        return self._pipeline

    @pipeline.setter
    def pipeline(self, value):
        if not isinstance(value, (PowerPipeline2D, RadiancePipeline2D)):
            raise TypeError('Sampler only compatible with PowerPipeLine2D or RadiancePipeline2D pipelines.')
        self._pipeline = value

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, np.ndarray value):
        if value.ndim != 2:
            raise ValueError("Mask must be a 2D array")
        if value.dtype not in (np.int, np.int32, np.int64, np.bool):
            raise TypeError("Mask must be an array of booleans or integers")
        self._mask = value

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
    cpdef list generate_tasks(self, tuple pixels):

        cdef:
            StatsArray2D frame
            int x, y
            np.ndarray normalised
            double[:,::1] error, normalised_mv
            list tasks

        if pixels != (self.mask.shape[0], self.mask.shape[1]):
            raise ValueError('The pixel geometry passed to the frame sampler is inconsistent with the mask frame size.')

        frame = self.pipeline.frame
        if frame is None:
            # no frame data available, generate tasks for the full frame
            return self._full_frame(pixels)

        # sanity check
        if pixels != frame.shape:
            raise ValueError('The pixel geometry passed to the frame sampler is inconsistent with the pipeline frame size.')

        error = frame.errors()
        normalised = np.zeros((frame.nx, frame.ny))
        normalised_mv = normalised

        # calculated normalised standard error
        for x in range(frame.nx):
            for y in range(frame.ny):
                if frame.mean_mv[x, y] <= 0:
                    normalised_mv[x, y] = 0
                else:
                    normalised_mv[x, y] = error[x, y] / frame.mean_mv[x, y]

        # build tasks
        tasks = []
        for x in range(frame.nx):
            for y in range(frame.ny):
                if self.mask[x, y] and (frame.samples_mv[x, y] < self.min_samples or normalised_mv[x, y] > self.cutoff):
                    tasks.append((x, y))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks

    cpdef list _full_frame(self, tuple pixels):

        cdef:
            list tasks
            int nx, ny, x, y

        tasks = []
        nx, ny = pixels
        for x in range(nx):
            for y in range(ny):
                if self.mask[x, y]:
                    tasks.append((x, y))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks


cdef class SpectralAdaptiveSampler2D(FrameSampler2D):
    """
    FrameSampler that dynamically adjusts a camera's pixel samples based on the noise
    level in each pixel's power value.

    Pixels that have high noise levels will receive extra samples until the desired
    noise threshold is achieve across the whole image.

    :param SpectralPowerPipeline2D pipeline: The specific power pipeline to use for feedback control.
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
        SpectralPowerPipeline2D _pipeline
        double _fraction, _ratio, _cutoff, _percentile
        str _reduction_method
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
        if not isinstance(value, (SpectralPowerPipeline2D, SpectralRadiancePipeline2D)):
            raise TypeError('Sampler only compatible with SpectralPowerPipeLine2D or SpectralRadiancePipeline2D pipelines.')
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
    cpdef list generate_tasks(self, tuple pixels):

        cdef:
            StatsArray3D frame
            int x, y, z, count, min_samples
            np.ndarray normalised, spectral_normalised
            double[::1] spectral_normalised_mv
            double[:, ::1] normalised_mv
            double[:, :, ::1] error
            double percentile_error, pixel_power
            list tasks

        frame = self.pipeline.frame
        if frame is None:
            # no frame data available, generate tasks for the full frame
            return self._full_frame(pixels)

        # sanity check
        if pixels != (frame.nx, frame.ny):
            raise ValueError('The pixel geometry passed to the frame sampler is inconsistent with the pipeline frame size.')

        min_samples = max(self.min_samples, <int>(frame.samples.max() / self.ratio))
        error = frame.errors()
        normalised = np.zeros((frame.nx, frame.ny))
        normalised_mv = normalised

        # calculated normalised standard error
        if self.reduction_method == 'weighted':
            for x in range(frame.nx):
                for y in range(frame.ny):
                    pixel_power = 0
                    for z in range(frame.nz):
                        if frame.mean_mv[x, y, z] > 0:
                            normalised_mv[x, y] += error[x, y, z]
                            pixel_power += frame.mean_mv[x, y, z]
                    if pixel_power:
                        normalised_mv[x, y] /= pixel_power

        elif self.reduction_method == 'mean':
            for x in range(frame.nx):
                for y in range(frame.ny):
                    count = 0
                    for z in range(frame.nz):
                        if frame.mean_mv[x, y, z] > 0:
                            normalised_mv[x, y] += error[x, y, z] / frame.mean_mv[x, y, z]
                            count += 1
                    if count:
                        normalised_mv[x, y] /= count
        else:
            spectral_normalised = np.zeros(frame.nz)
            spectral_normalised_mv = spectral_normalised
            for x in range(frame.nx):
                for y in range(frame.ny):
                    count = 0
                    for z in range(frame.nz):
                        if frame.mean_mv[x, y, z] > 0:
                            spectral_normalised_mv[count] = error[x, y, z] / frame.mean_mv[x, y, z]
                            count += 1
                    if count:
                        normalised_mv[x, y] = np.percentile(spectral_normalised[:count], self.percentile)

        # locate error value corresponding to fraction of frame to process
        percentile_error = np.percentile(normalised, (1 - self.fraction) * 100)

        # build tasks
        tasks = []
        for x in range(frame.nx):
            for y in range(frame.ny):
                if frame.samples[x, y].min() < min_samples or normalised_mv[x, y] > max(self.cutoff, percentile_error):
                    tasks.append((x, y))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks

    cpdef list _full_frame(self, tuple pixels):

        cdef:
            list tasks
            int nx, ny, x, y

        tasks = []
        nx, ny = pixels
        for x in range(nx):
            for y in range(ny):
                tasks.append((x, y))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks


cdef class MaskedSpectralAdaptiveSampler2D(FrameSampler2D):
    """
    A masked FrameSampler that dynamically adjusts a camera's pixel samples based on the noise
    level in each pixel's power value.

    Pixels that have high noise levels will receive extra samples until the desired
    noise threshold is achieve across the masked image.

    :param SpectralPowerPipeline2D pipeline: The specific power pipeline to use for feedback control.
    :param np.ndarray mask: The image mask array.
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
        SpectralPowerPipeline2D _pipeline
        double _cutoff, _percentile
        str _reduction_method
        int _min_samples
        np.ndarray _mask

    def __init__(self, object pipeline, np.ndarray mask, int min_samples=1000, double cutoff=0.0,
                 str reduction_method='percentile', double percentile=100.):

        self.pipeline = pipeline
        self.mask = mask
        self.min_samples = min_samples
        self.cutoff = cutoff
        self.reduction_method = reduction_method
        self.percentile = percentile

    @property
    def pipeline(self):
        return self._pipeline

    @pipeline.setter
    def pipeline(self, value):
        if not isinstance(value, (SpectralPowerPipeline2D, SpectralRadiancePipeline2D)):
            raise TypeError('Sampler only compatible with SpectralPowerPipeLine2D or SpectralRadiancePipeline2D pipelines.')
        self._pipeline = value

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, np.ndarray value):
        if value.ndim != 2:
            raise ValueError("Mask must be a 2D array")
        if value.dtype not in (np.int, np.int32, np.int64, np.bool):
            raise TypeError("Mask must be an array of booleans or integers")
        self._mask = value

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
    cpdef list generate_tasks(self, tuple pixels):

        cdef:
            StatsArray3D frame
            int x, y, z, count
            np.ndarray normalised, spectral_normalised
            double[::1] spectral_normalised_mv
            double[:, ::1] normalised_mv
            double[:, :, ::1] error
            double pixel_power
            list tasks

        if pixels != (self.mask.shape[0], self.mask.shape[1]):
            raise ValueError('The pixel geometry passed to the frame sampler is inconsistent with the mask frame size.')

        frame = self.pipeline.frame
        if frame is None:
            # no frame data available, generate tasks for the full frame
            return self._full_frame(pixels)

        # sanity check
        if pixels != (frame.nx, frame.ny):
            raise ValueError('The pixel geometry passed to the frame sampler is inconsistent with the pipeline frame size.')

        error = frame.errors()
        normalised = np.zeros((frame.nx, frame.ny))
        normalised_mv = normalised

        # calculated normalised standard error
        if self.reduction_method == 'weighted':
            for x in range(frame.nx):
                for y in range(frame.ny):
                    pixel_power = 0
                    for z in range(frame.nz):
                        if frame.mean_mv[x, y, z] > 0:
                            normalised_mv[x, y] += error[x, y, z]
                            pixel_power += frame.mean_mv[x, y, z]
                    if pixel_power:
                        normalised_mv[x, y] /= pixel_power

        elif self.reduction_method == 'mean':
            for x in range(frame.nx):
                for y in range(frame.ny):
                    count = 0
                    for z in range(frame.nz):
                        if frame.mean_mv[x, y, z] > 0:
                            normalised_mv[x, y] += error[x, y, z] / frame.mean_mv[x, y, z]
                            count += 1
                    if count:
                        normalised_mv[x, y] /= count
        else:
            spectral_normalised = np.zeros(frame.nz)
            spectral_normalised_mv = spectral_normalised
            for x in range(frame.nx):
                for y in range(frame.ny):
                    count = 0
                    for z in range(frame.nz):
                        if frame.mean_mv[x, y, z] > 0:
                            spectral_normalised_mv[count] = error[x, y, z] / frame.mean_mv[x, y, z]
                            count += 1
                    if count:
                        normalised_mv[x, y] = np.percentile(spectral_normalised[:count], self.percentile)

        # build tasks
        tasks = []
        for x in range(frame.nx):
            for y in range(frame.ny):
                if self.mask[x, y] and (frame.samples[x, y].min() < self.min_samples or normalised_mv[x, y] > self.cutoff):
                    tasks.append((x, y))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks

    cpdef list _full_frame(self, tuple pixels):

        cdef:
            list tasks
            int nx, ny, x, y

        tasks = []
        nx, ny = pixels
        for x in range(nx):
            for y in range(ny):
                if self.mask[x, y]:
                    tasks.append((x, y))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks


cdef class RGBAdaptiveSampler2D(FrameSampler2D):
    """
    FrameSampler that dynamically adjusts a camera's pixel samples based on the noise
    level in each RGB pixel value.

    Pixels that have high noise levels will receive extra samples until the desired
    noise threshold is achieve across the whole image.

    :param RGBPipeline2D pipeline: The specific RGB pipeline to use for feedback control.
    :param float fraction: The fraction of frame pixels to receive extra sampling
      (default=0.2).
    :param float ratio: The maximum allowable ratio between the maximum and minimum number of
      samples obtained for the pixels of the same observer (default=10).
      The sampler will generate additional tasks for pixels with the least number of samples
      in order to keep this ratio below a given value.
    :param int min_samples: Minimum number of pixel samples across the image before
      turning on adaptive sampling (default=1000).
    :param double cutoff: Noise threshold at which extra sampling will be aborted and
      rendering will complete (default=0.0).
    """

    cdef:
        RGBPipeline2D _pipeline
        double _fraction, _ratio, _cutoff
        int _min_samples

    def __init__(self, RGBPipeline2D pipeline, double fraction=0.2, double ratio=10.0, int min_samples=1000, double cutoff=0.0):

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
        if not isinstance(value, RGBPipeline2D):
            raise TypeError('Sampler only compatible with RGBPipeline2D pipeline.')
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
    @cython.cdivision(True)
    cpdef list generate_tasks(self, tuple pixels):

        cdef:
            StatsArray3D frame
            int x, y, c, min_samples, samples
            np.ndarray normalised
            double[:,:,::1] error
            double[:,::1] normalised_mv
            double percentile_error
            list tasks
            double[3] pixel_normalised

        frame = self.pipeline.xyz_frame
        if frame is None:
            # no frame data available, generate tasks for the full frame
            return self._full_frame(pixels)

        # sanity check
        if (pixels[0], pixels[1], 3) != frame.shape:
            raise ValueError('The number of pixels passed to the frame sampler are inconsistent with the pipeline frame size.')

        min_samples = max(self.min_samples, <int>(frame.samples.max() / self.ratio))
        error = frame.errors()
        normalised = np.zeros((frame.nx, frame.ny))
        normalised_mv = normalised

        # calculated normalised standard error
        for x in range(frame.nx):
            for y in range(frame.ny):
                for c in range(3):
                    if frame.mean_mv[x, y, c] <= 0:
                        pixel_normalised[c] = 0
                    else:
                        pixel_normalised[c] = error[x, y, c] / frame.mean_mv[x, y, c]
                normalised_mv[x, y] = max(pixel_normalised[0], pixel_normalised[1], pixel_normalised[2])

        # locate error value corresponding to fraction of frame to process
        percentile_error = np.percentile(normalised, (1 - self.fraction) * 100)

        # build tasks
        tasks = []
        for x in range(frame.nx):
            for y in range(frame.ny):
                samples = min(frame.samples_mv[x, y, 0], frame.samples_mv[x, y, 1], frame.samples_mv[x, y, 2])
                if samples < min_samples or normalised_mv[x, y] > max(self.cutoff, percentile_error):
                    tasks.append((x, y))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks

    cpdef list _full_frame(self, tuple pixels):

        cdef:
            list tasks
            int nx, ny, x, y

        tasks = []
        nx, ny = pixels
        for x in range(nx):
            for y in range(ny):
                tasks.append((x, y))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks


cdef class MaskedRGBAdaptiveSampler2D(FrameSampler2D):
    """
    A masked FrameSampler that dynamically adjusts a camera's pixel samples based on the noise
    level in each RGB pixel value.

    Pixels that have high noise levels will receive extra samples until the desired
    noise threshold is achieve across the whole image.

    :param RGBPipeline2D pipeline: The specific RGB pipeline to use for feedback control.
    :param np.ndarray mask: The image mask array.
    :param int min_samples: Minimum number of pixel samples across the image before
      turning on adaptive sampling (default=1000).
    :param double cutoff: Noise threshold at which extra sampling will be aborted and
      rendering will complete (default=0.0).
    """

    cdef:
        RGBPipeline2D _pipeline
        double _fraction, _cutoff
        int _min_samples
        np.ndarray _mask

    def __init__(self, RGBPipeline2D pipeline, np.ndarray mask, int min_samples=1000, double cutoff=0.0):

        self.pipeline = pipeline
        self.min_samples = min_samples
        self.cutoff = cutoff
        self.mask = mask

    @property
    def pipeline(self):
        return self._pipeline

    @pipeline.setter
    def pipeline(self, value):
        if not isinstance(value, RGBPipeline2D):
            raise TypeError('Sampler only compatible with RGBPipeline2D pipeline.')
        self._pipeline = value

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, np.ndarray value):
        if value.ndim != 2:
            raise ValueError("Mask must be a 2D array")
        if value.dtype not in (np.int, np.int32, np.int64, np.bool):
            raise TypeError("Mask must be an array of booleans or integers")
        self._mask = value

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
    @cython.cdivision(True)
    cpdef list generate_tasks(self, tuple pixels):

        cdef:
            StatsArray3D frame
            int x, y, c, samples
            np.ndarray normalised
            double[:,:,::1] error
            double[:,::1] normalised_mv
            double percentile_error
            list tasks
            double[3] pixel_normalised

        if pixels != (self.mask.shape[0], self.mask.shape[1]):
            raise ValueError('The pixel geometry passed to the frame sampler is inconsistent with the mask frame size.')

        frame = self.pipeline.xyz_frame
        if frame is None:
            # no frame data available, generate tasks for the full frame
            return self._full_frame(pixels)

        # sanity check
        if (pixels[0], pixels[1], 3) != frame.shape:
            raise ValueError('The number of pixels passed to the frame sampler are inconsistent with the pipeline frame size.')

        error = frame.errors()
        normalised = np.zeros((frame.nx, frame.ny))
        normalised_mv = normalised

        # calculated normalised standard error
        for x in range(frame.nx):
            for y in range(frame.ny):
                for c in range(3):
                    if frame.mean_mv[x, y, c] <= 0:
                        pixel_normalised[c] = 0
                    else:
                        pixel_normalised[c] = error[x, y, c] / frame.mean_mv[x, y, c]
                normalised_mv[x, y] = max(pixel_normalised[0], pixel_normalised[1], pixel_normalised[2])

        # build tasks
        tasks = []
        for x in range(frame.nx):
            for y in range(frame.ny):
                if self.mask[x, y]:
                    samples = min(frame.samples_mv[x, y, 0], frame.samples_mv[x, y, 1], frame.samples_mv[x, y, 2])
                    if samples < self.min_samples or normalised_mv[x, y] > self.cutoff:
                        tasks.append((x, y))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks

    cpdef list _full_frame(self, tuple pixels):

        cdef:
            list tasks
            int nx, ny, x, y

        tasks = []
        nx, ny = pixels
        for x in range(nx):
            for y in range(ny):
                if self.mask[x, y]:
                    tasks.append((x, y))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks
