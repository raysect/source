# Copyright (c) 2016, Dr Alex Meakins, Raysect Project
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
from raysect.optical.observer.pipeline cimport RGBPipeline2D, RadiancePipeline2D, PowerPipeline2D
from raysect.core.math cimport StatsArray1D, StatsArray2D, StatsArray3D
cimport numpy as np
cimport cython


cdef class FullFrameSampler2D(FrameSampler2D):

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


cdef class MonoAdaptiveSampler2D(FrameSampler2D):
    """
    FrameSampler that dynamically adjusts a camera's pixel samples based on the noise
    level in each pixel's power value.

    Pixels that have high noise levels will receive extra samples until the desired
    noise threshold is achieve across the whole image.

    :param PowerPipeline2D pipeline: The specific power pipeline to use for feedback control.
    :param float fraction: The fraction of frame pixels to receive extra sampling
      (default=0.2).
    :param float ratio:
    :param int min_samples: Minimum number of pixel samples across the image before
      turning on adaptive sampling (default=1000).
    :param double cutoff: Normalised noise threshold at which extra sampling will be aborted and
      rendering will complete (default=0.0). The standard error is normalised to 1 so that a
      cutoff of 0.01 corresponds to 1% standard error.
    """

    cdef:
        PowerPipeline2D pipeline
        double fraction, ratio, cutoff
        int min_samples

    def __init__(self, object pipeline, double fraction=0.2, double ratio=10.0, int min_samples=1000, double cutoff=0.0):

        if not isinstance(pipeline, (PowerPipeline2D, RadiancePipeline2D)):
            raise TypeError('Sampler only compatible with PowerPipeLine2D or RadiancePipeline2D pipelines.')

        # todo: validation
        self.pipeline = pipeline
        self.fraction = fraction
        self.ratio = ratio
        self.min_samples = min_samples
        self.cutoff = cutoff

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef list generate_tasks(self, tuple pixels):

        cdef:
            StatsArray2D frame
            int x, y
            np.ndarray normalised
            double[:,::1] error, normalised_mv
            double percentile_error
            list tasks

        frame = self.pipeline.frame
        if frame is None:
            # no frame data available, generate tasks for the full frame
            return self._full_frame(pixels)

        # sanity check
        if pixels != frame.shape:
            raise ValueError('The pixel geometry passed to the frame sampler is inconsistent with the pipeline frame size.')

        min_samples = max(self.min_samples, frame.samples.max() / self.ratio)
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
        PowerPipeline2D pipeline
        double fraction, ratio, cutoff
        int min_samples
        np.ndarray mask

    def __init__(self, object pipeline, np.ndarray mask, int min_samples=1000, double cutoff=0.0):

        if not isinstance(pipeline, (PowerPipeline2D, RadiancePipeline2D)):
            raise TypeError('Sampler only compatible with PowerPipeLine2D or RadiancePipeline2D pipelines.')

        # todo: validation
        self.pipeline = pipeline
        self.min_samples = min_samples
        self.cutoff = cutoff
        self.mask = mask

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef list generate_tasks(self, tuple pixels):

        cdef:
            StatsArray2D frame
            int x, y
            np.ndarray normalised
            double[:,::1] error, normalised_mv
            double percentile_error
            list tasks

        frame = self.pipeline.frame
        if frame is None:
            # no frame data available, generate tasks for the full frame
            return self._full_frame(pixels)

        # sanity check
        if pixels != frame.shape:
            raise ValueError('The pixel geometry passed to the frame sampler is inconsistent with the pipeline frame size.')

        if pixels != (self.mask.shape[0], self.mask.shape[1]):
            raise ValueError('The pixel geometry passed to the frame sampler is inconsistent with the mask frame size.')

        min_samples = self.min_samples
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
                if self.mask[x, y] and (frame.samples_mv[x, y] < min_samples or normalised_mv[x, y] > self.cutoff):
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
    :param float ratio:
    :param int min_samples: Minimum number of pixel samples across the image before
      turning on adaptive sampling (default=1000).
    :param double cutoff: Noise threshold at which extra sampling will be aborted and
      rendering will complete (default=0.0).
    """

    cdef:
        RGBPipeline2D pipeline
        double fraction, ratio, cutoff
        int min_samples

    def __init__(self, RGBPipeline2D pipeline, double fraction=0.2, double ratio=10.0, int min_samples=1000, double cutoff=0.0):

        # todo: validation
        self.pipeline = pipeline
        self.fraction = fraction
        self.ratio = ratio
        self.min_samples = min_samples
        self.cutoff = cutoff

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
        RGBPipeline2D pipeline
        double fraction, ratio, cutoff
        int min_samples
        np.ndarray mask

    def __init__(self, RGBPipeline2D pipeline, np.ndarray mask, int min_samples=1000, double cutoff=0.0):

        # todo: validation
        self.pipeline = pipeline
        self.min_samples = min_samples
        self.cutoff = cutoff
        self.mask = mask

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

        if pixels != (self.mask.shape[0], self.mask.shape[1]):
            raise ValueError('The pixel geometry passed to the frame sampler is inconsistent with the mask frame size.')

        min_samples = self.min_samples
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
                    if samples < min_samples or normalised_mv[x, y] > self.cutoff:
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
