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

from numpy import zeros, float64, int32
from numpy cimport ndarray
cimport cython


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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object add_sample(self, int x, int y, int channel, double sample):
        cdef:
            int nx, ny, n
            double m, v
            int[:,:,::1] samples_mv
            double[:,:,::1] value_mv, variance_mv

        self._bounds_check(x, y, channel)

        # acquire memory-views
        value_mv = self.value
        variance_mv = self.variance
        samples_mv = self.samples

        # initial values
        m = value_mv[x, y, channel]
        v = variance_mv[x, y, channel]
        n = samples_mv[x, y, channel]

        # calculate statistics
        _add_sample(sample, &m, &v, &n)

        # update frame values
        value_mv[x, y, channel] = m
        variance_mv[x, y, channel] = v
        samples_mv[x, y, channel] = n

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object combine_samples(self, int x, int y, int channel, double mean, double variance, int sample_count):

        cdef:
            int nx, ny, nt = 0
            double mx, my, mt = 0
            double vx, vy, vt = 0
            int[:,:,::1] samples_mv
            double[:,:,::1] value_mv, variance_mv

        if sample_count < 1:
            raise ValueError('Number of samples cannot be less than one.')

        if sample_count == 1:
            self.add_sample(x, y, channel, mean)
            return

        self._bounds_check(x, y, channel)

        # acquire memory-views
        value_mv = self.value
        variance_mv = self.variance
        samples_mv = self.samples

        # set 1 sample count, mean and variance
        mx = value_mv[x, y, channel]
        vx = variance_mv[x, y, channel]
        nx = samples_mv[x, y, channel]

        # set 2 sample count, mean and variance
        my = mean
        vy = variance
        ny = sample_count

        # calculate statistics
        _combine_samples(mx, vx, nx, my, vy, ny, &mt, &vt, &nt)

        # update frame values
        value_mv[x, y, channel] = mt
        variance_mv[x, y, channel] = vt
        samples_mv[x, y, channel] = nt

    cpdef object clear(self):
        self._new_buffers()

    cdef inline void _new_buffers(self):
        nx, ny = self.pixels
        self.value = zeros((nx, ny, self.channels), dtype=float64)
        self.variance = zeros((nx, ny, self.channels), dtype=float64)
        self.samples = zeros((nx, ny, self.channels), dtype=int32)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline object _bounds_check(self, int x, int y, int channel):

        cdef int nx, ny

        if channel < 0 or channel >= self.channels:
            raise ValueError("Channel index is out of range.")

        nx, ny = self.pixels

        if x < 0 or x >= nx:
            raise ValueError("Pixel x index is out of range.")

        if y < 0 or y >= ny:
            raise ValueError("Pixel y index is out of range.")


@cython.cdivision(True)
cdef inline void _add_sample(double sample, double *m, double *v, int *n):
    """
    Updates the mean, variance and sample count with the supplied sample value.

    :param sample: Sample value.
    :param m: Mean to update.
    :param v: Variance to update.
    :param n: Sample count to update.
    """

    cdef:
        double prev_m, prev_v
        int prev_n

    if n[0] == 0:

        # initial sample
        n[0] = 1
        m[0] = sample
        v[0] = 0

    else:

        # cache previous values for computation
        # prev_n is bump up for a single sample to avoid a divide by zero
        # in this scenario prev_v will always be 0 so this has no effect on the result
        prev_m = m[0]
        prev_v = v[0]
        prev_n = n[0] if n[0] > 1 else 2

        # update statistics
        n[0] += 1
        m[0] = prev_m + (sample - prev_m) / n[0]
        v[0] = (prev_v * (prev_n - 1) + (sample - prev_m)*(sample - m[0])) / (n[0] - 1)


@cython.cdivision(True)
cdef inline void _combine_samples(double mx, double vx, int nx, double my, double vy, int ny, double *mt, double *vt, int *nt):
    """
    Computes the combined statistics of two sets of samples specified by mean, variance and sample count.

    :param mx: Mean of set x.
    :param vx: Variance of set x.
    :param nx: Sample count of set x.
    :param my: Mean of set y.
    :param vy: Variance of set y.
    :param ny: Sample count of set y.
    :param mt: Combined mean.
    :param vt: Combined variance.
    :param nt: Combined sample count.
    """

    # accumulate samples
    nt[0] = nx + ny

    # calculate new mean
    mt[0] = (nx*mx + ny*my) / <double> nt[0]

    # convert unbiased variance to biased variance
    vx = (nx - 1) * vx / <double> nx
    vy = (ny - 1) * vy / <double> ny

    # calculate new variance
    vt[0] = (nx * (mx*mx + vx) + ny * (my*my + vy)) / <double> nt[0] - mt[0]*mt[0]

    # convert biased variance to unbiased variance
    vt[0] = nt[0] * vt[0] / <double> (nt[0] - 1)
