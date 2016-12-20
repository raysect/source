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

from libc.math cimport sqrt
cimport cython

# todo: add recalculate() method to all below to force error recalculation ... or better yet return std_error on demand as a function()

cdef class Pixel:

    def __init__(self, channels):

        if channels <= 0:
            raise ValueError("There must be at least one c.")

        self.channels = channels

        # generate pixel buffers
        self._new_buffers()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object add_sample(self, int channel, double sample):
        cdef:
            int n
            double m, v

        self._bounds_check(channel)

        # initial values
        m = self._mean_mv[channel]
        v = self._variance_mv[channel]
        n = self._samples_mv[channel]

        # calculate statistics
        _add_sample(sample, &m, &v, &n)

        # update frame values
        self._mean_mv[channel] = m
        self._variance_mv[channel] = v
        self._samples_mv[channel] = n

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object combine_samples(self, int channel, double mean, double variance, int sample_count):

        cdef:
            int nx, ny, nt = 0
            double mx, my, mt = 0
            double vx, vy, vt = 0

        self._bounds_check(channel)

        # validate
        if sample_count < 1:
            raise ValueError('Number of samples must not be less than 1.')

        # clamp variance to zero
        # occasionally numerical accuracy limits can result in values < 0
        if variance < 0:
            variance = 0

        # stored sample count, mean and variance
        mx = self._mean_mv[channel]
        vx = self._variance_mv[channel]
        nx = self._samples_mv[channel]

        # external sample count, mean and variance
        my = mean
        vy = variance
        ny = sample_count

        # calculate statistics
        _combine_samples(mx, vx, nx, my, vy, ny, &mt, &vt, &nt)

        # update frame values
        self._mean_mv[channel] = mt
        self._variance_mv[channel] = vt
        self._samples_mv[channel] = nt

    cpdef double error(self, int channel):
        return _std_error(self._variance_mv[channel], self._samples_mv[channel])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ndarray errors(self):

        cdef:
            int c
            ndarray errors
            double[::1] errors_mv

        errors = zeros((self.channels,), dtype=float64)

        errors_mv = errors
        for c in range(self.channels):
            errors_mv[c] = _std_error(self._variance_mv[c], self._samples_mv[c])

        return errors

    cpdef object clear(self):
        self._new_buffers()

    cdef inline void _new_buffers(self):
        self.mean = zeros((self.channels,), dtype=float64)
        self.variance = zeros((self.channels, ), dtype=float64)
        self.samples = zeros((self.channels, ), dtype=int32)
        self._mean_mv = self.mean
        self._variance_mv = self.variance
        self._samples_mv = self.samples

    cdef inline object _bounds_check(self, int channel):

        if channel < 0 or channel >= self.channels:
            raise ValueError("Channel index is out of range.")


cdef class Frame1D:

    def __init__(self, pixels, channels):

        if pixels < 1:
            raise ValueError("Number of pixels must be >= 1.")

        if channels <= 0:
            raise ValueError("There must be at least one c.")

        self.pixels = pixels
        self.channels = channels

        # generate frame buffers
        self._new_buffers()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object add_sample(self, int i, int channel, double sample):
        cdef:
            int n
            double m, v

        self._bounds_check(i, channel)

        # initial values
        m = self._mean_mv[i, channel]
        v = self._variance_mv[i, channel]
        n = self._samples_mv[i, channel]

        # calculate statistics
        _add_sample(sample, &m, &v, &n)

        # update frame values
        self._mean_mv[i, channel] = m
        self._variance_mv[i, channel] = v
        self._samples_mv[i, channel] = n

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object combine_samples(self, int i, int channel, double mean, double variance, int sample_count):

        cdef:
            int nx, ny, nt = 0
            double mx, my, mt = 0
            double vx, vy, vt = 0

        self._bounds_check(i, channel)

        # validate
        if sample_count < 1:
            raise ValueError('Number of samples must not be less than 1.')

        # clamp variance to zero
        # occasionally numerical accuracy limits can result in values < 0
        if variance < 0:
            variance = 0

        # stored sample count, mean and variance
        mx = self._mean_mv[i, channel]
        vx = self._variance_mv[i, channel]
        nx = self._samples_mv[i, channel]

        # external sample count, mean and variance
        my = mean
        vy = variance
        ny = sample_count

        # calculate statistics
        _combine_samples(mx, vx, nx, my, vy, ny, &mt, &vt, &nt)

        # update frame values
        self._mean_mv[i, channel] = mt
        self._variance_mv[i, channel] = vt
        self._samples_mv[i, channel] = nt

    cpdef double error(self, int i, int channel):
        return _std_error(self._variance_mv[i, channel], self._samples_mv[i, channel])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ndarray errors(self):

        cdef:
            int i, c
            ndarray errors
            double[:,::1] errors_mv

        errors = zeros((self.pixels, self.channels), dtype=float64)

        errors_mv = errors
        for i in range(self.pixels):
            for c in range(self.channels):
                errors_mv[i, c] = _std_error(self._variance_mv[i, c], self._samples_mv[i, c])

        return errors

    cpdef object clear(self):
        self._new_buffers()

    cdef inline void _new_buffers(self):
        self.mean = zeros((self.pixels, self.channels), dtype=float64)
        self.variance = zeros((self.pixels, self.channels), dtype=float64)
        self.samples = zeros((self.pixels, self.channels), dtype=int32)
        self._mean_mv = self.mean
        self._variance_mv = self.variance
        self._samples_mv = self.samples

    cdef inline object _bounds_check(self, int i, int channel):

        if channel < 0 or channel >= self.channels:
            raise ValueError("Channel index is out of range.")

        if i < 0 or i >= self.pixels:
            raise ValueError("Pixel index is out of range.")


cdef class Frame2D:

    def __init__(self, pixels, channels):

        nx, ny = pixels
        if nx < 1 or ny < 1:
            raise ValueError("Bins must be a tuple of x and y dimensions, both of which must be >= 1.")

        if channels <= 0:
            raise ValueError("There must be at least one c.")

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

        self._bounds_check(x, y, channel)

        # initial values
        m = self._mean_mv[x, y, channel]
        v = self._variance_mv[x, y, channel]
        n = self._samples_mv[x, y, channel]

        # calculate statistics
        _add_sample(sample, &m, &v, &n)

        # update frame values
        self._mean_mv[x, y, channel] = m
        self._variance_mv[x, y, channel] = v
        self._samples_mv[x, y, channel] = n

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object combine_samples(self, int x, int y, int channel, double mean, double variance, int sample_count):

        cdef:
            int nx, ny, nt = 0
            double mx, my, mt = 0
            double vx, vy, vt = 0

        self._bounds_check(x, y, channel)

        # validate
        if sample_count < 1:
            raise ValueError('Number of samples must not be less than 1.')

        # clamp variance to zero
        # occasionally numerical accuracy limits can result in values < 0
        if variance < 0:
            variance = 0

        # stored sample count, mean and variance
        mx = self._mean_mv[x, y, channel]
        vx = self._variance_mv[x, y, channel]
        nx = self._samples_mv[x, y, channel]

        # external sample count, mean and variance
        my = mean
        vy = variance
        ny = sample_count

        # calculate statistics
        _combine_samples(mx, vx, nx, my, vy, ny, &mt, &vt, &nt)

        # update frame values
        self._mean_mv[x, y, channel] = mt
        self._variance_mv[x, y, channel] = vt
        self._samples_mv[x, y, channel] = nt

    cpdef double error(self, int x, int y, int channel):
        return _std_error(self._variance_mv[x, y, channel], self._samples_mv[x, y, channel])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ndarray errors(self):

        cdef:
            int nx, ny, x, y, c
            ndarray errors
            double[:,:,::1] errors_mv

        nx, ny = self.pixels
        errors = zeros((nx, ny, self.channels), dtype=float64)

        errors_mv = errors
        for x in range(nx):
            for y in range(ny):
                for c in range(self.channels):
                    errors_mv[x, y, c] = _std_error(self._variance_mv[x, y, c], self._samples_mv[x, y, c])

        return errors

    cpdef object clear(self):
        self._new_buffers()

    cdef inline void _new_buffers(self):
        nx, ny = self.pixels
        self.mean = zeros((nx, ny, self.channels), dtype=float64)
        self.variance = zeros((nx, ny, self.channels), dtype=float64)
        self.samples = zeros((nx, ny, self.channels), dtype=int32)
        self._mean_mv = self.mean
        self._variance_mv = self.variance
        self._samples_mv = self.samples

    cdef inline object _bounds_check(self, int x, int y, int channel):

        cdef int nx, ny

        if channel < 0 or channel >= self.channels:
            raise ValueError("Channel index is out of range.")

        nx, ny = self.pixels

        if x < 0 or x >= nx:
            raise ValueError("Pixel x index is out of range.")

        if y < 0 or y >= ny:
            raise ValueError("Pixel y index is out of range.")


# todo: move to core
@cython.cdivision(True)
cdef inline double _std_error(double v, int n):
    """
    Calculates the standard error from the variance.

    :param v: Variance.
    :param n: Number of samples.
    :return: The standard error.
    """

    if n <= 0 or v <= 0:
        return 0
    return sqrt(v / n)


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


# todo: move to core
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

    cdef double temp

    # ensure set x is the largest set
    if nx < ny:
        _swap_int(&nx, &ny)
        _swap_double(&mx, &my)
        _swap_double(&vx, &vy)

    # most common case first
    if nx > 1 and ny > 1:

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

        return

    # special cases
    if nx == 0 and ny == 0:

        # no samples
        nt[0] = 0
        mt[0] = 0
        vt[0] = 0

    elif nx == 1:

        if ny == 0:

            # single sample
            nt[0] = 1
            mt[0] = mx
            vt[0] = 0

        else:

            # two independent samples, combine
            nt[0] = 2
            mt[0] = 0.5 * (mx + my)
            temp = mx - mt[0]
            vt[0] = 2*temp*temp

    elif nx > 1:

        # two samples in set x
        nt[0] = nx
        mt[0] = mx
        vt[0] = vx

        if ny == 1:

            # single sample from set y
            _add_sample(my, mt, vt, nt)


# todo: move to core
cdef inline void _swap_int(int *a, int *b):

        cdef int temp
        temp = a[0]
        a[0] = b[0]
        b[0] = temp


# todo: move to core
cdef inline void _swap_double(double *a, double *b):

        cdef double temp
        temp = a[0]
        a[0] = b[0]
        b[0] = temp
