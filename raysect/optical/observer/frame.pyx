# cython: language_level=3

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

from numpy import zeros, full, float64, int32, inf
from numpy cimport ndarray


cdef class Frame2D:

    cdef:
        readonly tuple pixels
        readonly int channels
        readonly ndarray value
        readonly ndarray variance
        readonly ndarray samples

    def __init__(self, pixels, channels):

        nx, ny = pixels
        if nx < 1 or ny < 1:
            raise ValueError("Pixels must be a tuple of x and y dimensions, both of which must be >= 1.")

        if channels <= 0:
            raise ValueError("There must be at least one channel.")

        self.pixels = pixels
        self.channels = channels

        # generate frame buffers
        self._new_buffers()

    cpdef object add_sample(self, int x, int y, int channel, double value):
        pass

    cpdef object combine_samples(self, int x, int y, int channel, double mean, double variance, int sample_count):

        cdef:
            int nx, ny, nt
            double mx, my, mt
            double vx, vy, vt
            int[:,:,::1] samples_mv
            double[:,:,::1] value_mv, variance_mv

        if sample_count < 1:
            raise ValueError('Number of samples cannot be less than one.')

        if sample_count == 1:
            self.add_sample(x, y, channel, mean)
            return

        # acquire memory-views
        value_mv = self.value
        variance_mv = self.variance
        samples_mv = self.samples

        # accumulate samples
        nx = samples_mv[x, y, channel]
        ny = sample_count
        nt = nx + ny

        # calculate new mean
        mx = value_mv[x, y, channel]
        my = mean
        mt = (nx*mx + ny*my) / <double> nt

        # convert unbiased variance to biased variance
        vx = (nx - 1) * variance_mv[x, y, channel] / <double> nx
        vy = (ny - 1) * variance / <double> ny

        # calculate new variance
        vt = (nx * (mx*mx + vx) + ny * (my*my + vy)) / <double> nt - mt*mt

        # convert biased variance to unbiased variance
        vt = nt * vt / <double> (nt - 1)

        # update frame values
        value_mv[x, y, channel] = mt
        variance_mv[x, y, channel] = vt
        samples_mv[x, y, channel] = nt

    cpdef object clear(self):
        self._new_buffers()

    cdef inline void _new_buffers(self):
        nx, ny = self.pixels
        self.value = zeros((nx, ny, self.channels), dtype=float64)
        self.variance = full((nx, ny, self.channels), fill_value=inf, dtype=float64)
        self.samples = zeros((nx, ny, self.channels), dtype=int32)
