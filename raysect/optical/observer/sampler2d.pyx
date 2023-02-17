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

import warnings
import numpy as np
from random import shuffle

from raysect.optical.observer.base cimport FrameSampler2D
from raysect.optical.observer.pipeline cimport RGBPipeline2D, RadiancePipeline2D, PowerPipeline2D, SpectralRadiancePipeline2D, SpectralPowerPipeline2D
from raysect.core.math cimport StatsArray2D, StatsArray3D
cimport numpy as np
cimport cython
ctypedef np.uint8_t uint8  # numpy boolean arrays are stored as 8-bit values


cdef class FullFrameSampler2D(FrameSampler2D):
    """
    Evenly samples the full 2D frame or its masked fragment.

    :param np.ndarray mask: The image mask array (default=None). A 2D boolean array with
      the same shape as the frame. The tasks are generated only for those pixels for which
      the mask is True.
    """
    cdef:
        np.ndarray _mask
        uint8[:, ::1] _mask_mv

    def __init__(self, np.ndarray mask=None):

        self.mask = mask

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, np.ndarray value):
        if value is None:
            self._mask = None
        else:
            if value.ndim != 2:
                raise ValueError("Mask must be a 2D array.")
            self._mask = value.astype(bool)
            self._mask_mv = np.frombuffer(self._mask, dtype=np.uint8).reshape(self.mask.shape)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef list generate_tasks(self, tuple pixels):

        cdef:
            list tasks
            int nx, ny

        # The all-true mask is created during the first call of generate_tasks if no mask was provided
        if self.mask is None:
            self.mask = np.ones(pixels, dtype=bool)

        if pixels != (self._mask.shape[0], self._mask.shape[1]):
            if np.all(self._mask):
                # In case of all-true mask, generate a new one that matches the pixel geometry.
                self.mask = np.ones(pixels, dtype=bool)
            else:
                raise ValueError('The pixel geometry passed to the frame sampler is inconsistent with the mask shape.')

        tasks = []
        nx, ny = pixels
        for iy in range(ny):
            for ix in range(nx):
                if self._mask_mv[ix, iy]:
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
    :param float fraction: The fraction of frame (or its masked fragment) pixels to receive
      extra sampling (default=0.2).
    :param float ratio: The maximum allowable ratio between the maximum and minimum number of
      samples obtained for the pixels of the same observer (default=10).
      The sampler will generate additional tasks for pixels with the least number of samples
      in order to keep this ratio below a given value.
    :param int min_samples: Minimum number of pixel samples across the image
      (or its masked fragment) before turning on adaptive sampling (default=1000).
    :param double cutoff: Normalised noise threshold at which extra sampling will be aborted and
      rendering will complete (default=0.0). The standard error is normalised to 1 so that a
      cutoff of 0.01 corresponds to 1% standard error.
    :param np.ndarray mask: The image mask array (default=None). A 2D boolean array with
      the same shape as the frame. The tasks are generated only for those pixels for which
      the mask is True. If not provided, the all-true mask will be created during the first call
      of generate_tasks().
    """

    cdef:
        PowerPipeline2D _pipeline
        double _fraction, _ratio, _cutoff
        int _min_samples
        np.ndarray _mask
        uint8[:, ::1] _mask_mv

    def __init__(self, object pipeline, double fraction=0.2, double ratio=10.0, int min_samples=1000, double cutoff=0.0, mask=None):

        self.pipeline = pipeline
        self.fraction = fraction
        self.ratio = ratio
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
    def fraction(self):
        return self._fraction

    @fraction.setter
    def fraction(self, value):
        if value <= 0 or value > 1.:
            raise ValueError("Attribute 'fraction' must be in the range (0, 1].")
        self._fraction = value

    @property
    def ratio(self):
        return self._ratio

    @ratio.setter
    def ratio(self, value):
        if value < 1.:
            raise ValueError("Attribute 'ratio' must be >= 1.")
        self._ratio = value

    @property
    def min_samples(self):
        return self._min_samples

    @min_samples.setter
    def min_samples(self, value):
        if value < 1:
            raise ValueError("Attribute 'min_samples' must be >= 1.")
        self._min_samples = value

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value):
        if value < 0 or value > 1.:
            raise ValueError("Attribute 'cutoff' must be in the range [0, 1].")
        self._cutoff = value

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, np.ndarray value):
        if value is None:
            self._mask = None
        else:
            if value.ndim != 2:
                raise ValueError("Mask must be a 2D array.")
            self._mask = value.astype(bool)
            self._mask_mv = np.frombuffer(self._mask, dtype=np.uint8).reshape(self.mask.shape)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef list generate_tasks(self, tuple pixels):

        cdef:
            StatsArray2D frame
            int x, y, min_samples
            np.ndarray normalised
            double[:, ::1] error, normalised_mv
            double percentile_error, cutoff
            list tasks

        # The all-true mask is created during the first call of generate_tasks if no mask was provided
        if self.mask is None:
            self.mask = np.ones(pixels, dtype=bool)

        if pixels != (self._mask.shape[0], self._mask.shape[1]):
            if np.all(self._mask):
                # In case of all-true mask, generate a new one that matches the pixel geometry.
                self.mask = np.ones(pixels, dtype=bool)
            else:
                raise ValueError('The pixel geometry passed to the frame sampler is inconsistent with the mask shape.')

        frame = self._pipeline.frame
        if frame is None:
            # no frame data available, generate tasks for the full frame
            return self._full_frame(pixels)

        # sanity check
        if pixels != frame.shape:
            raise ValueError('The pixel geometry passed to the frame sampler is inconsistent with the pipeline frame size.')

        min_samples = max(self._min_samples, <int>(frame.samples[self._mask].max() / self._ratio))

        error = frame.errors()
        normalised = np.zeros((frame.nx, frame.ny))
        normalised_mv = normalised

        # calculated normalised standard error
        for x in range(frame.nx):
            for y in range(frame.ny):
                if self._mask_mv[x, y] and frame.mean_mv[x, y] > 0:
                    normalised_mv[x, y] = error[x, y] / frame.mean_mv[x, y]

        # locate error value corresponding to fraction of frame to process
        percentile_error = np.percentile(normalised[self._mask], (1 - self._fraction) * 100)
        cutoff = max(self._cutoff, percentile_error)

        # build tasks
        tasks = []
        for x in range(frame.nx):
            for y in range(frame.ny):
                if self._mask_mv[x, y] and (frame.samples_mv[x, y] < min_samples or normalised_mv[x, y] > cutoff):
                    tasks.append((x, y))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef list _full_frame(self, tuple pixels):

        cdef:
            list tasks
            int nx, ny, x, y

        if self.mask is None:  # just in case if _full_frame() is called before generate_tasks()
            self.mask = np.ones(pixels, dtype=bool)

        tasks = []
        nx, ny = pixels
        for x in range(nx):
            for y in range(ny):
                if self._mask_mv[x, y]:
                    tasks.append((x, y))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks


cdef class MaskedMonoAdaptiveSampler2D(MonoAdaptiveSampler2D):
    """
    A masked FrameSampler that dynamically adjusts a camera's pixel samples based on the noise
    level in each pixel's power value. Deprecated in version 0.7, instead use
    MonoAdaptiveSampler2D with `mask` attribute.

    Pixels that have high noise levels will receive extra samples until the desired
    noise threshold is achieve across the masked image.

    :param PowerPipeline2D pipeline: The specific power pipeline to use for feedback control.
    :param np.ndarray mask: The image mask array. A 2D boolean array with
      the same shape as the frame. The tasks are generated only for those pixels for which
      the mask is True.
    :param int min_samples: Minimum number of pixel samples across the image before
      turning on adaptive sampling (default=1000).
    :param double cutoff: Normalised noise threshold at which extra sampling will be aborted and
      rendering will complete (default=0.0). The standard error is normalised to 1 so that a
      cutoff of 0.01 corresponds to 1% standard error.
    """

    def __init__(self, object pipeline, np.ndarray mask, int min_samples=1000, double cutoff=0.0):

        warnings.warn("MaskedMonoAdaptiveSampler2D is deprecated and will be removed in a future version. " +
                      "Use MonoAdaptiveSampler2D with 'mask' attribute.", FutureWarning)
        super().__init__(pipeline, fraction=1.0, ratio=100000.0, min_samples=min_samples, cutoff=cutoff, mask=mask)


cdef class SpectralAdaptiveSampler2D(FrameSampler2D):
    """
    FrameSampler that dynamically adjusts a camera's pixel samples based on the noise
    level in each pixel's power value.

    Pixels that have high noise levels will receive extra samples until the desired
    noise threshold is achieve across the whole image.

    :param SpectralPowerPipeline2D pipeline: The specific power pipeline to use for feedback control.
    :param float fraction: The fraction of frame (or its masked fragment) pixels to receive
      extra sampling (default=0.2).
    :param float ratio: The maximum allowable ratio between the maximum and minimum number of
      samples obtained for the pixels of the same observer (default=10).
      The sampler will generate additional tasks for pixels with the least number of samples
      in order to keep this ratio below a given value.
    :param int min_samples: Minimum number of pixel samples across the image
      (or its masked fragment) before turning on adaptive sampling (default=1000).
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
       - `reduction_method='power_percentile'`: the error of a pixel is calculated as the highest
         spectral error among a given percentage of spectral bins with the highest spectral power.
    :param double percentile: Used only if `reduction_method='percentile'` or
      `reduction_method='power_percentile'` (default=100).
       - `reduction_method='percentile'`: If `percentile=x`, extra sampling will be aborted
         if x% of spectral bins of each pixel have normalised errors lower than `cutoff`.
       - `reduction_method='power_percentile'`: If `percentile=x`, extra sampling will be aborted
         if x% of spectral bins with the highest spectral power all have normalised errors lower
         than `cutoff`.
    """

    cdef:
        SpectralPowerPipeline2D _pipeline
        double _fraction, _ratio, _cutoff, _percentile
        str _reduction_method
        int _min_samples
        np.ndarray _mask
        uint8[:, ::1] _mask_mv

    def __init__(self, object pipeline, double fraction=0.2, double ratio=10.0, int min_samples=1000, double cutoff=0.0,
                 str reduction_method='percentile', double percentile=100., np.ndarray mask=None):

        self.pipeline = pipeline
        self.fraction = fraction
        self.ratio = ratio
        self.min_samples = min_samples
        self.cutoff = cutoff
        self.reduction_method = reduction_method
        self.percentile = percentile
        self.mask = mask

    @property
    def pipeline(self):
        return self._pipeline

    @pipeline.setter
    def pipeline(self, value):
        if not isinstance(value, (SpectralPowerPipeline2D, SpectralRadiancePipeline2D)):
            raise TypeError('Sampler only compatible with SpectralPowerPipeline2D or SpectralRadiancePipeline2D pipelines.')
        self._pipeline = value

    @property
    def fraction(self):
        return self._fraction

    @fraction.setter
    def fraction(self, value):
        if value <= 0 or value > 1.:
            raise ValueError("Attribute 'fraction' must be in the range (0, 1].")
        self._fraction = value

    @property
    def ratio(self):
        return self._ratio

    @ratio.setter
    def ratio(self, value):
        if value < 1.:
            raise ValueError("Attribute 'ratio' must be >= 1.")
        self._ratio = value

    @property
    def min_samples(self):
        return self._min_samples

    @min_samples.setter
    def min_samples(self, value):
        if value < 1:
            raise ValueError("Attribute 'min_samples' must be >= 1.")
        self._min_samples = value

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value):
        if value < 0 or value > 1.:
            raise ValueError("Attribute 'cutoff' must be in the range [0, 1].")
        self._cutoff = value

    @property
    def reduction_method(self):
        return self._reduction_method

    @reduction_method.setter
    def reduction_method(self, value):
        if value not in {'weighted', 'mean', 'percentile', 'power_percentile'}:
            raise ValueError("Attribute 'reduction_method' must be 'weighted', 'mean', 'percentile' or 'power_percentile'.")
        self._reduction_method = value

    @property
    def percentile(self):
        return self._percentile

    @percentile.setter
    def percentile(self, value):
        if value < 0 or value > 100.:
            raise ValueError("Percentiles must be in the range [0, 100].")
        self._percentile = value

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, np.ndarray value):
        if value is None:
            self._mask = None
        else:
            if value.ndim != 2:
                raise ValueError("Mask must be a 2D array.")
            self._mask = value.astype(bool)
            self._mask_mv = np.frombuffer(self._mask, dtype=np.uint8).reshape(self.mask.shape)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef list generate_tasks(self, tuple pixels):

        cdef:
            StatsArray3D frame
            int x, y, min_samples
            np.ndarray normalised
            double[:, ::1] normalised_mv
            int[:, ::1] frame_min_samples
            double percentile_error, cutoff
            list tasks

        # The all-true mask is created during the first call of generate_tasks if no mask was provided
        if self.mask is None:
            self.mask = np.ones(pixels, dtype=bool)

        if pixels != (self._mask.shape[0], self._mask.shape[1]):
            if np.all(self._mask):
                # In case of all-true mask, generate a new one that matches the pixel geometry.
                self.mask = np.ones(pixels, dtype=bool)
            else:
                raise ValueError('The pixel geometry passed to the frame sampler is inconsistent with the mask shape.')

        frame = self._pipeline.frame
        if frame is None:
            # no frame data available, generate tasks for the full frame
            return self._full_frame(pixels)

        # sanity check
        if pixels != (frame.nx, frame.ny):
            raise ValueError('The pixel geometry passed to the frame sampler is inconsistent with the pipeline frame size.')

        min_samples = max(self._min_samples, <int>(frame.samples.max(2)[self._mask].max() / self._ratio))

        # calculated normalised standard error
        if self._reduction_method == 'weighted':
            normalised = self._reduce_weighted()

        elif self._reduction_method == 'mean':
            normalised = self._reduce_mean()

        elif self._reduction_method == 'percentile':
            normalised = self._reduce_percentile()

        elif self._reduction_method == 'power_percentile':
            normalised = self._reduce_power_percentile()

        else:
            raise ValueError("Attribute 'reduction_method' has wrong value: %s. " % self._reduction_method +
                             "Must be 'weighted', 'mean', 'percentile' or 'power_percentile'.")

        normalised_mv = normalised

        # locate error value corresponding to fraction of frame to process
        percentile_error = np.percentile(normalised[self._mask], (1 - self._fraction) * 100)
        cutoff = max(self._cutoff, percentile_error)

        # build tasks
        tasks = []
        frame_min_samples = frame.samples.min(2)
        for x in range(frame.nx):
            for y in range(frame.ny):
                if self._mask_mv[x, y] and (frame_min_samples[x, y] < min_samples or normalised_mv[x, y] > cutoff):
                    tasks.append((x, y))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef np.ndarray _reduce_weighted(self):

        cdef:
            StatsArray3D frame
            int x, y, z
            np.ndarray normalised
            double[:, :, ::1] error
            double[:, ::1] normalised_mv
            double pixel_power

        frame = self._pipeline.frame
        error = frame.errors()
        normalised = np.zeros((frame.nx, frame.ny))
        normalised_mv = normalised
        for x in range(frame.nx):
            for y in range(frame.ny):
                if self._mask_mv[x, y]:
                    pixel_power = 0
                    for z in range(frame.nz):
                        if frame.mean_mv[x, y, z] > 0:
                            normalised_mv[x, y] += error[x, y, z]
                            pixel_power += frame.mean_mv[x, y, z]
                    if pixel_power:
                        normalised_mv[x, y] /= pixel_power

        return normalised

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef np.ndarray _reduce_mean(self):

        cdef:
            StatsArray3D frame
            int x, y, z, count
            np.ndarray normalised
            double[:, :, ::1] error
            double[:, ::1] normalised_mv

        frame = self._pipeline.frame
        error = frame.errors()
        normalised = np.zeros((frame.nx, frame.ny))
        normalised_mv = normalised
        for x in range(frame.nx):
            for y in range(frame.ny):
                if self._mask_mv[x, y]:
                    count = 0
                    for z in range(frame.nz):
                        if frame.mean_mv[x, y, z] > 0:
                            normalised_mv[x, y] += error[x, y, z] / frame.mean_mv[x, y, z]
                            count += 1
                    if count:
                        normalised_mv[x, y] /= count

        return normalised

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef np.ndarray _reduce_percentile(self):

        cdef:
            StatsArray3D frame
            int x, y, z, count
            np.ndarray spectral_normalised, normalised
            double[:, :, ::1] error
            double[:, ::1] normalised_mv
            double[:] spectral_normalised_mv

        frame = self._pipeline.frame
        error = frame.errors()
        normalised = np.zeros((frame.nx, frame.ny))
        normalised_mv = normalised
        spectral_normalised = np.zeros(frame.nz)
        spectral_normalised_mv = spectral_normalised
        for x in range(frame.nx):
            for y in range(frame.ny):
                if self._mask_mv[x, y]:
                    count = 0
                    for z in range(frame.nz):
                        if frame.mean_mv[x, y, z] > 0:
                            spectral_normalised_mv[count] = error[x, y, z] / frame.mean_mv[x, y, z]
                            count += 1
                    if count:
                        normalised_mv[x, y] = np.percentile(spectral_normalised[:count], self._percentile)

        return normalised

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef np.ndarray _reduce_power_percentile(self):

        cdef:
            StatsArray3D frame
            int x, y, z, count
            np.ndarray normalised, spectral_power
            double[:, :, ::1] error
            double[:, ::1] normalised_mv
            double[:] spectral_power_mv
            double power_threshold

        frame = self._pipeline.frame
        error = frame.errors()
        normalised = np.zeros((frame.nx, frame.ny))
        normalised_mv = normalised
        spectral_power = np.zeros(frame.nz)
        spectral_power_mv = spectral_power
        for x in range(frame.nx):
            for y in range(frame.ny):
                if self._mask_mv[x, y]:
                    count = 0
                    for z in range(frame.nz):
                        if frame.mean_mv[x, y, z] > 0:
                            spectral_power_mv[count] = frame.mean_mv[x, y, z]
                            count += 1
                    if count:
                        power_threshold = np.percentile(spectral_power[:count], (100. - self._percentile))
                        for z in range(frame.nz):
                            if frame.mean_mv[x, y, z] >= power_threshold:
                                normalised_mv[x, y] = max(normalised_mv[x, y], error[x, y, z] / frame.mean_mv[x, y, z])

        return normalised

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef list _full_frame(self, tuple pixels):

        cdef:
            list tasks
            int nx, ny, x, y

        if self.mask is None:  # just in case if _full_frame() is called before generate_tasks()
            self.mask = np.ones(pixels, dtype=bool)

        tasks = []
        nx, ny = pixels
        for x in range(nx):
            for y in range(ny):
                if self._mask_mv[x, y]:
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
    :param float fraction: The fraction of frame (or its masked fragment) pixels to receive
      extra sampling (default=0.2).
    :param float ratio: The maximum allowable ratio between the maximum and minimum number of
      samples obtained for the pixels of the same observer (default=10).
      The sampler will generate additional tasks for pixels with the least number of samples
      in order to keep this ratio below a given value.
    :param int min_samples: Minimum number of pixel samples across the image
      (or its masked fragment) before turning on adaptive sampling (default=1000).
    :param double cutoff: Noise threshold at which extra sampling will be aborted and
      rendering will complete (default=0.0).
    :param np.ndarray mask: The image mask array (default=None). A 2D boolean array with
      the same shape as the frame. The tasks are generated only for those pixels for which
      the mask is True. If not provided, the all-true mask will be created during the first call
      of generate_tasks().
    """

    cdef:
        RGBPipeline2D _pipeline
        double _fraction, _ratio, _cutoff
        int _min_samples
        np.ndarray _mask
        uint8[:, ::1] _mask_mv

    def __init__(self, RGBPipeline2D pipeline, double fraction=0.2, double ratio=10.0, int min_samples=1000, double cutoff=0.0, np.ndarray mask=None):

        self.pipeline = pipeline
        self.fraction = fraction
        self.ratio = ratio
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
    def fraction(self):
        return self._fraction

    @fraction.setter
    def fraction(self, value):
        if value <= 0 or value > 1.:
            raise ValueError("Attribute 'fraction' must be in the range (0, 1].")
        self._fraction = value

    @property
    def ratio(self):
        return self._ratio

    @ratio.setter
    def ratio(self, value):
        if value < 1.:
            raise ValueError("Attribute 'ratio' must be >= 1.")
        self._ratio = value

    @property
    def min_samples(self):
        return self._min_samples

    @min_samples.setter
    def min_samples(self, value):
        if value < 1:
            raise ValueError("Attribute 'min_samples' must be >= 1.")
        self._min_samples = value

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value):
        if value < 0 or value > 1.:
            raise ValueError("Attribute 'cutoff' must be in the range [0, 1].")
        self._cutoff = value

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, np.ndarray value):
        if value is None:
            self._mask = None
        else:
            if value.ndim != 2:
                raise ValueError("Mask must be a 2D array.")
            self._mask = value.astype(bool)
            self._mask_mv = np.frombuffer(self._mask, dtype=np.uint8).reshape(self.mask.shape)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cpdef list generate_tasks(self, tuple pixels):

        cdef:
            StatsArray3D frame
            int x, y, c, min_samples, min_pixel_samples
            np.ndarray normalised
            double[:, :, ::1] error
            double[:, ::1] normalised_mv
            double percentile_error, cutoff
            list tasks
            double[3] pixel_normalised

        # The all-true mask is created during the first call of generate_tasks if no mask was provided
        if self.mask is None:
            self.mask = np.ones(pixels, dtype=bool)

        if pixels != (self._mask.shape[0], self._mask.shape[1]):
            if np.all(self._mask):
                # In case of all-true mask, generate a new one that matches the pixel geometry.
                self.mask = np.ones(pixels, dtype=bool)
            else:
                raise ValueError('The pixel geometry passed to the frame sampler is inconsistent with the mask shape.')

        frame = self._pipeline.xyz_frame
        if frame is None:
            # no frame data available, generate tasks for the full frame
            return self._full_frame(pixels)

        # sanity check
        if (pixels[0], pixels[1], 3) != frame.shape:
            raise ValueError('The number of pixels passed to the frame sampler are inconsistent with the pipeline frame size.')

        min_samples = max(self._min_samples, <int>(frame.samples[self._mask].max() / self._ratio))
        error = frame.errors()
        normalised = np.zeros((frame.nx, frame.ny))
        normalised_mv = normalised

        # calculated normalised standard error
        for x in range(frame.nx):
            for y in range(frame.ny):
                if self._mask_mv[x, y]:
                    for c in range(3):
                        if frame.mean_mv[x, y, c] > 0:
                            pixel_normalised[c] = error[x, y, c] / frame.mean_mv[x, y, c]
                    normalised_mv[x, y] = max(pixel_normalised[0], pixel_normalised[1], pixel_normalised[2])

        # locate error value corresponding to fraction of frame to process
        percentile_error = np.percentile(normalised[self._mask], (1 - self._fraction) * 100)
        cutoff = max(self._cutoff, percentile_error)

        # build tasks
        tasks = []
        for x in range(frame.nx):
            for y in range(frame.ny):
                if self._mask_mv[x, y]:
                    min_pixel_samples = min(frame.samples_mv[x, y, 0], frame.samples_mv[x, y, 1], frame.samples_mv[x, y, 2])
                    if min_pixel_samples < min_samples or normalised_mv[x, y] > cutoff:
                        tasks.append((x, y))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef list _full_frame(self, tuple pixels):

        cdef:
            list tasks
            int nx, ny, x, y

        if self.mask is None:  # just in case if _full_frame() is called before generate_tasks()
            self.mask = np.ones(pixels, dtype=bool)

        tasks = []
        nx, ny = pixels
        for x in range(nx):
            for y in range(ny):
                if self._mask_mv[x, y]:
                    tasks.append((x, y))

        # perform tasks in random order so that image is assembled randomly rather than sequentially
        shuffle(tasks)

        return tasks


cdef class MaskedRGBAdaptiveSampler2D(RGBAdaptiveSampler2D):
    """
    A masked FrameSampler that dynamically adjusts a camera's pixel samples based on the noise
    level in each RGB pixel value. Deprecated in version 0.7, instead use MonoAdaptiveSampler2D
    with `mask` attribute.

    Pixels that have high noise levels will receive extra samples until the desired
    noise threshold is achieve across the whole image.

    :param RGBPipeline2D pipeline: The specific RGB pipeline to use for feedback control.
    :param np.ndarray mask: The image mask array.
    :param int min_samples: Minimum number of pixel samples across the image before
      turning on adaptive sampling (default=1000).
    :param double cutoff: Noise threshold at which extra sampling will be aborted and
      rendering will complete (default=0.0).
    """

    def __init__(self, RGBPipeline2D pipeline, np.ndarray mask, int min_samples=1000, double cutoff=0.0):

        warnings.warn("MaskedRGBAdaptiveSampler2D is deprecated and will be removed in a future version. " +
                      "Use RGBAdaptiveSampler2D with 'mask' attribute.", FutureWarning)
        super().__init__(pipeline, fraction=1.0, ratio=100000.0, min_samples=min_samples, cutoff=cutoff, mask=mask)
