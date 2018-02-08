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
from raysect.optical.observer.pipeline cimport RadiancePipeline1D, PowerPipeline1D
from raysect.core.math cimport StatsArray1D
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
    :param float ratio:
    :param int min_samples: Minimum number of pixel samples across the image before
      turning on adaptive sampling (default=1000).
    :param double cutoff: Normalised noise threshold at which extra sampling will be aborted and
      rendering will complete (default=0.0). The standard error is normalised to 1 so that a
      cutoff of 0.01 corresponds to 1% standard error.
    """

    cdef:
        PowerPipeline1D pipeline
        double fraction, ratio, cutoff
        int min_samples

    def __init__(self, object pipeline, double fraction=0.2, double ratio=10.0, int min_samples=1000, double cutoff=0.0):

        if not isinstance(pipeline, (PowerPipeline1D, RadiancePipeline1D)):
            raise TypeError('Sampler only compatible with PowerPipeLine1D or RadiancePipeline1D pipelines.')

        # todo: validation
        self.pipeline = pipeline
        self.fraction = fraction
        self.ratio = ratio
        self.min_samples = min_samples
        self.cutoff = cutoff

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef list generate_tasks(self, int pixels):

        cdef:
            StatsArray1D frame
            int pixel
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

        min_samples = max(self.min_samples, frame.samples.max() / self.ratio)
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

