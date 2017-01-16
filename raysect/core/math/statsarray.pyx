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


cdef class StatsBin:

    def __init__(self):

        self.mean = 0.0
        self.variance = 0.0
        self.samples = 0

    cpdef object clear(self):
        self.mean = 0.0
        self.variance = 0.0
        self.samples = 0

    cpdef StatsBin copy(self):
        obj = StatsBin()
        obj.mean = self.mean
        obj.variance = self.variance
        obj.samples = self.samples
        return obj

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object add_sample(self, double sample):
        _add_sample(sample, &self.mean, &self.variance, &self.samples)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object combine_samples(self, double mean, double variance, int sample_count):

        cdef:
            int na, nb, nt = 0
            double ma, mb, mt = 0
            double va, vb, vt = 0

        # validate
        if sample_count < 1:
            raise ValueError('Number of samples must not be less than 1.')

        # clamp variance to zero
        # occasionally numerical accuracy limits can result in values < 0
        if variance < 0:
            variance = 0

        # stored sample count, mean and variance
        ma = self.mean
        va = self.variance
        na = self.samples

        # external sample count, mean and variance
        mb = mean
        vb = variance
        nb = sample_count

        # calculate statistics
        _combine_samples(ma, va, na, mb, vb, nb, &mt, &vt, &nt)

        # update frame values
        self.mean = mt
        self.variance = vt
        self.samples = nt

    cpdef double error(self):
        return _std_error(self.variance, self.samples)


cdef class StatsArray1D:

    def __init__(self, length):

        if length <= 0:
            raise ValueError("There must be at least one element.")
        self.length = length

        # generate buffers
        self._new_buffers()

    @property
    def shape(self):
        return (self.length, )

    cpdef object clear(self):
        self._new_buffers()

    cpdef StatsArray1D copy(self):
        obj = StatsArray1D(self.length)
        obj.mean[:] = self.mean[:]
        obj.variance[:] = self.variance[:]
        obj.samples[:] = self.samples[:]
        return obj

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object add_sample(self, int x, double sample):
        cdef:
            int n
            double m, v

        self._bounds_check(x)

        # initial values
        m = self.mean_mv[x]
        v = self.variance_mv[x]
        n = self.samples_mv[x]

        # calculate statistics
        _add_sample(sample, &m, &v, &n)

        # update frame values
        self.mean_mv[x] = m
        self.variance_mv[x] = v
        self.samples_mv[x] = n

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object combine_samples(self, int x, double mean, double variance, int sample_count):

        cdef:
            int na, nb, nt = 0
            double ma, mb, mt = 0
            double va, vb, vt = 0

        self._bounds_check(x)

        # validate
        if sample_count < 1:
            raise ValueError('Number of samples must not be less than 1.')

        # clamp variance to zero
        # occasionally numerical accuracy limits can result in values < 0
        if variance < 0:
            variance = 0

        # stored sample count, mean and variance
        ma = self.mean_mv[x]
        va = self.variance_mv[x]
        na = self.samples_mv[x]

        # external sample count, mean and variance
        mb = mean
        vb = variance
        nb = sample_count

        # calculate statistics
        _combine_samples(ma, va, na, mb, vb, nb, &mt, &vt, &nt)

        # update frame values
        self.mean_mv[x] = mt
        self.variance_mv[x] = vt
        self.samples_mv[x] = nt

    cpdef double error(self, int x):
        return _std_error(self.variance_mv[x], self.samples_mv[x])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ndarray errors(self):

        cdef:
            int x
            ndarray errors
            double[::1] errors_mv

        errors = zeros((self.length,), dtype=float64)

        errors_mv = errors
        for x in range(self.length):
            errors_mv[x] = _std_error(self.variance_mv[x], self.samples_mv[x])

        return errors

    cdef inline void _new_buffers(self):
        self.mean = zeros((self.length,), dtype=float64)
        self.variance = zeros((self.length, ), dtype=float64)
        self.samples = zeros((self.length, ), dtype=int32)
        self.mean_mv = self.mean
        self.variance_mv = self.variance
        self.samples_mv = self.samples

    cdef inline object _bounds_check(self, int x):

        if x < 0 or x >= self.length:
            raise ValueError("Index is out of range.")


cdef class StatsArray2D:

    def __init__(self, nx, ny):

        if nx < 1:
            raise ValueError("Number of x axis elements must be >= 1.")

        if ny < 1:
            raise ValueError("Number of y axis elements must be >= 1.")

        self.nx = nx
        self.ny = ny

        # generate frame buffers
        self._new_buffers()

    @property
    def shape(self):
        return self.nx, self.ny

    cpdef object clear(self):
        self._new_buffers()

    cpdef StatsArray2D copy(self):
        obj = StatsArray2D(self.nx, self.ny)
        obj.mean[:] = self.mean[:]
        obj.variance[:] = self.variance[:]
        obj.samples[:] = self.samples[:]
        return obj

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object add_sample(self, int x, int y, double sample):
        cdef:
            int n
            double m, v

        self._bounds_check(x, y)

        # initial values
        m = self.mean_mv[x, y]
        v = self.variance_mv[x, y]
        n = self.samples_mv[x, y]

        # calculate statistics
        _add_sample(sample, &m, &v, &n)

        # update frame values
        self.mean_mv[x, y] = m
        self.variance_mv[x, y] = v
        self.samples_mv[x, y] = n

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object combine_samples(self, int x, int y, double mean, double variance, int sample_count):

        cdef:
            int nx, ny, nt = 0
            double mx, my, mt = 0
            double vx, vy, vt = 0

        self._bounds_check(x, y)

        # validate
        if sample_count < 1:
            raise ValueError('Number of samples must not be less than 1.')

        # clamp variance to zero
        # occasionally numerical accuracy limits can result in values < 0
        if variance < 0:
            variance = 0

        # stored sample count, mean and variance
        mx = self.mean_mv[x, y]
        vx = self.variance_mv[x, y]
        nx = self.samples_mv[x, y]

        # external sample count, mean and variance
        my = mean
        vy = variance
        ny = sample_count

        # calculate statistics
        _combine_samples(mx, vx, nx, my, vy, ny, &mt, &vt, &nt)

        # update frame values
        self.mean_mv[x, y] = mt
        self.variance_mv[x, y] = vt
        self.samples_mv[x, y] = nt

    cpdef double error(self, int x, int y):
        return _std_error(self.variance_mv[x, y], self.samples_mv[x, y])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ndarray errors(self):

        cdef:
            int x, y
            ndarray errors
            double[:,::1] errors_mv

        errors = zeros((self.nx, self.ny), dtype=float64)

        errors_mv = errors
        for x in range(self.nx):
            for y in range(self.ny):
                errors_mv[x, y] = _std_error(self.variance_mv[x, y], self.samples_mv[x, y])

        return errors

    cdef inline void _new_buffers(self):
        self.mean = zeros((self.nx, self.ny), dtype=float64)
        self.variance = zeros((self.nx, self.ny), dtype=float64)
        self.samples = zeros((self.nx, self.ny), dtype=int32)
        self.mean_mv = self.mean
        self.variance_mv = self.variance
        self.samples_mv = self.samples

    cdef inline object _bounds_check(self, int x, int y):

        if x < 0 or x >= self.nx:
            raise ValueError("Index x is out of range.")

        if y < 0 or y >= self.ny:
            raise ValueError("Index y is out of range.")


cdef class StatsArray3D:

    def __init__(self, nx, ny, nz):

        if nx < 1:
            raise ValueError("Number of x axis elements must be >= 1.")

        if ny < 1:
            raise ValueError("Number of y axis elements must be >= 1.")

        if nz < 1:
            raise ValueError("Number of z axis elements must be >= 1.")

        self.nx = nx
        self.ny = ny
        self.nz = nz

        # generate frame buffers
        self._new_buffers()

    @property
    def shape(self):
        return self.nx, self.ny, self.nz

    cpdef object clear(self):
        self._new_buffers()

    cpdef StatsArray3D copy(self):
        obj = StatsArray3D(self.nx, self.ny, self.nz)
        obj.mean[:] = self.mean[:]
        obj.variance[:] = self.variance[:]
        obj.samples[:] = self.samples[:]
        return obj

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object add_sample(self, int x, int y, int z, double sample):
        cdef:
            int n
            double m, v

        self._bounds_check(x, y, z)

        # initial values
        m = self.mean_mv[x, y, z]
        v = self.variance_mv[x, y, z]
        n = self.samples_mv[x, y, z]

        # calculate statistics
        _add_sample(sample, &m, &v, &n)

        # update frame values
        self.mean_mv[x, y, z] = m
        self.variance_mv[x, y, z] = v
        self.samples_mv[x, y, z] = n

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object combine_samples(self, int x, int y, int z, double mean, double variance, int sample_count):

        cdef:
            int na, nb, nt = 0
            double ma, mb, mt = 0
            double va, vb, vt = 0

        self._bounds_check(x, y, z)

        # validate
        if sample_count < 1:
            raise ValueError('Number of samples must not be less than 1.')

        # clamp variance to zero
        # occasionally numerical accuracy limits can result in values < 0
        if variance < 0:
            variance = 0

        # stored sample count, mean and variance
        ma = self.mean_mv[x, y, z]
        va = self.variance_mv[x, y, z]
        na = self.samples_mv[x, y, z]

        # external sample count, mean and variance
        mb = mean
        vb = variance
        nb = sample_count

        # calculate statistics
        _combine_samples(ma, va, na, mb, vb, nb, &mt, &vt, &nt)

        # update frame values
        self.mean_mv[x, y, z] = mt
        self.variance_mv[x, y, z] = vt
        self.samples_mv[x, y, z] = nt

    cpdef double error(self, int x, int y, int z):
        return _std_error(self.variance_mv[x, y, z], self.samples_mv[x, y, z])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ndarray errors(self):

        cdef:
            int x, y, z
            ndarray errors
            double[:,:,::1] errors_mv

        errors = zeros((self.nx, self.ny, self.nz), dtype=float64)

        errors_mv = errors
        for x in range(self.nx):
            for y in range(self.ny):
                for z in range(self.nz):
                    errors_mv[x, y, z] = _std_error(self.variance_mv[x, y, z], self.samples_mv[x, y, z])

        return errors

    cdef inline void _new_buffers(self):
        self.mean = zeros((self.nx, self.ny, self.nz), dtype=float64)
        self.variance = zeros((self.nx, self.ny, self.nz), dtype=float64)
        self.samples = zeros((self.nx, self.ny, self.nz), dtype=int32)
        self.mean_mv = self.mean
        self.variance_mv = self.variance
        self.samples_mv = self.samples

    cdef inline object _bounds_check(self, int x, int y, int z):

        if x < 0 or x >= self.nx:
            raise ValueError("Index x is out of range.")

        if y < 0 or y >= self.ny:
            raise ValueError("Index y is out of range.")

        if z < 0 or z >= self.nz:
            raise ValueError("Index z is out of range.")


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
