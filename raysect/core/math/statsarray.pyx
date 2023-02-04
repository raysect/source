# cython: language_level=3

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

from numpy import zeros, float64, int32

from libc.math cimport sqrt
from raysect.core.math.cython.utility cimport swap_int, swap_double
cimport cython


cdef class StatsBin:
    """
    Class for storing a single numerical sampling result and its associated statistics.

    :ivar float mean: The mean value of the samples.
    :ivar float variance: The variance of the collected samples.
    :ivar int samples: The total number of samples in the set.
    """

    def __init__(self):

        self.mean = 0.0
        self.variance = 0.0
        self.samples = 0

    cpdef object clear(self):
        """ Erase the current statistics stored in this StatsBin. """
        self.mean = 0.0
        self.variance = 0.0
        self.samples = 0

    cpdef StatsBin copy(self):
        """ Instantiate a new StatsBin object with the same statistical results. """
        obj = StatsBin()
        obj.mean = self.mean
        obj.variance = self.variance
        obj.samples = self.samples
        return obj

    cpdef object add_sample(self, double sample):
        """
        Add a single sample to this StatsBin.

        :param float sample: The sample value to be added.
        """
        _add_sample(sample, &self.mean, &self.variance, &self.samples)

    cpdef object combine_samples(self, double mean, double variance, int sample_count):
        """
        Combine the statistics from another set of samples with the results already stored in this StatsBin.

        :param float mean: The mean of the new samples
        :param float variance: The variance of the new samples
        :param int sample_count: The number of new samples that were taken.
        """

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
        """ Compute the standard error of this sample distribution. """
        return _std_error(self.variance, self.samples)


cdef class StatsArray1D:
    """
    Class for storing a 1D array of sampling results and their associated statistics.

    :param int length: The length of the 1D samples array.

    :ivar ndarray mean: The mean value of the samples.
    :ivar ndarray variance: The variance of the collected samples.
    :ivar ndarray samples: The total number of samples in the set.
    :ivar int length: The length of the 1D samples array.
    """

    def __init__(self, length):

        if length <= 0:
            raise ValueError("There must be at least one element.")
        self.length = length

        # generate buffers
        self._new_buffers()

    def __getstate__(self):
        return self.length, self.mean, self.variance, self.samples

    def __setstate__(self, state):

        self.length, self.mean, self.variance, self.samples = state

        # reconstruct memory views
        self.mean_mv = self.mean
        self.variance_mv = self.variance
        self.samples_mv = self.samples

    @property
    def shape(self):
        """ The numpy style array shape of the underlying StatsArray. """

        return (self.length, )

    cpdef object clear(self):
        """ Erase the current statistics stored in this StatsArray. """
        self._new_buffers()

    @cython.initializedcheck(False)
    cpdef StatsArray1D copy(self):
        """ Instantiate a new StatsArray1D object with the same statistical results. """
        obj = StatsArray1D(self.length)
        obj.mean_mv[:] = self.mean_mv[:]
        obj.variance_mv[:] = self.variance_mv[:]
        obj.samples_mv[:] = self.samples_mv[:]
        return obj

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef object add_sample(self, int x, double sample):
        """
        Add a single sample to the StatsArray1D element x.

        :param int x: The position index where the sample should be added.
        :param float sample: The sample value to be added.
        """
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
    @cython.initializedcheck(False)
    cpdef object combine_samples(self, int x, double mean, double variance, int sample_count):
        """
        Combine the statistics from a given set of samples with the results already stored in
        this StatsArray at index position x.

        :param int x: The index position where these results are to be added.
        :param float mean: The mean of the new samples
        :param float variance: The variance of the new samples
        :param int sample_count: The number of new samples that were taken.
        """

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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef double error(self, int x):
        """
        Compute the standard error of the results at index position x.

        :param int x: The index position at which to compute the standard error.
        """
        self._bounds_check(x)
        return _std_error(self.variance_mv[x], self.samples_mv[x])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef ndarray errors(self):
        """
        Compute the standard errors of all the results stored in this StatsArray.

        :rtype: ndarray
        """

        cdef:
            int x
            ndarray errors
            double[::1] errors_mv

        errors = zeros((self.length,), dtype=float64)
        errors_mv = errors
        for x in range(self.length):
            errors_mv[x] = _std_error(self.variance_mv[x], self.samples_mv[x])
        return errors

    cdef void _new_buffers(self):
        self.mean = zeros((self.length,), dtype=float64)
        self.variance = zeros((self.length, ), dtype=float64)
        self.samples = zeros((self.length, ), dtype=int32)
        self.mean_mv = self.mean
        self.variance_mv = self.variance
        self.samples_mv = self.samples

    cdef object _bounds_check(self, int x):
        if x < 0 or x >= self.length:
            raise ValueError("Index is out of range.")


cdef class StatsArray2D:
    """
    Class for storing a 2D array of sampling results and their associated statistics.

    :param int nx: The number of array samples along the x direction.
    :param int ny: The number of array samples along the y direction.

    :ivar ndarray mean: The mean value of the samples.
    :ivar ndarray variance: The variance of the collected samples.
    :ivar ndarray samples: The total number of samples in the set.
    :ivar int nx: The number of array samples along the x direction.
    :ivar int ny: The number of array samples along the y direction.
    """

    def __init__(self, nx, ny):

        if nx < 1:
            raise ValueError("Number of x axis elements must be >= 1.")

        if ny < 1:
            raise ValueError("Number of y axis elements must be >= 1.")

        self.nx = nx
        self.ny = ny

        # generate frame buffers
        self._new_buffers()

    def __getstate__(self):
        return self.nx, self.ny, self.mean, self.variance, self.samples

    def __setstate__(self, state):
        self.nx, self.ny, self.mean, self.variance, self.samples = state

        # reconstruct memory views
        self.mean_mv = self.mean
        self.variance_mv = self.variance
        self.samples_mv = self.samples

    @property
    def shape(self):
        """ The numpy style array shape of the underlying StatsArray. """
        return self.nx, self.ny

    cpdef object clear(self):
        """ Erase the current statistics stored in this StatsArray. """
        self._new_buffers()

    @cython.initializedcheck(False)
    cpdef StatsArray2D copy(self):
        """ Instantiate a new StatsArray2D object with the same statistical results. """
        obj = StatsArray2D(self.nx, self.ny)
        obj.mean_mv[:] = self.mean_mv[:]
        obj.variance_mv[:] = self.variance_mv[:]
        obj.samples_mv[:] = self.samples_mv[:]
        return obj

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef object add_sample(self, int x, int y, double sample):
        """
        Add a single sample to the StatsArray2D results stored at element x, y.

        :param int x: The x position index where the sample should be added.
        :param int y: The y position index where the sample should be added.
        :param float sample: The sample value to be added.
        """

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
    @cython.initializedcheck(False)
    cpdef object combine_samples(self, int x, int y, double mean, double variance, int sample_count):
        """
        Combine the statistics from a given set of samples with the results already stored in
        this StatsArray at index position x, y.

        :param int x: The x index position where these results are to be added.
        :param int y: The y index position where these results are to be added.
        :param float mean: The mean of the new samples
        :param float variance: The variance of the new samples
        :param int sample_count: The number of new samples that were taken.
        """

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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef double error(self, int x, int y):
        """
        Compute the standard error of the results at index position x, y.

        :param int x: The x index position at which to compute the standard error.
        :param int y: The y index position at which to compute the standard error.
        """
        self._bounds_check(x, y)
        return _std_error(self.variance_mv[x, y], self.samples_mv[x, y])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef ndarray errors(self):
        """
        Compute the standard errors of all the results stored in this StatsArray.

        :rtype: ndarray
        """

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

    cdef void _new_buffers(self):
        self.mean = zeros((self.nx, self.ny), dtype=float64)
        self.variance = zeros((self.nx, self.ny), dtype=float64)
        self.samples = zeros((self.nx, self.ny), dtype=int32)
        self.mean_mv = self.mean
        self.variance_mv = self.variance
        self.samples_mv = self.samples

    cdef object _bounds_check(self, int x, int y):

        if x < 0 or x >= self.nx:
            raise ValueError("Index x is out of range.")

        if y < 0 or y >= self.ny:
            raise ValueError("Index y is out of range.")


cdef class StatsArray3D:
    """
    Class for storing a 3D array of sampling results and their associated statistics.

    :param int nx: The number of array samples along the x direction.
    :param int ny: The number of array samples along the y direction.
    :param int nz: The number of array samples along the z direction.

    :ivar ndarray mean: The mean value of the samples.
    :ivar ndarray variance: The variance of the collected samples.
    :ivar ndarray samples: The total number of samples in the set.
    :ivar int nx: The number of array samples along the x direction.
    :ivar int ny: The number of array samples along the y direction.
    :ivar int nz: The number of array samples along the z direction.
    """

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

    def __getstate__(self):
        return self.nx, self.ny, self.nz, self.mean, self.variance, self.samples

    def __setstate__(self, state):

        self.nx, self.ny, self.nz, self.mean, self.variance, self.samples = state

        # reconstruct memory views
        self.mean_mv = self.mean
        self.variance_mv = self.variance
        self.samples_mv = self.samples

    @property
    def shape(self):
        """ The numpy style array shape of the underlying StatsArray. """
        return self.nx, self.ny, self.nz

    cpdef object clear(self):
        """ Erase the current statistics stored in this StatsArray. """
        self._new_buffers()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef StatsArray3D copy(self):
        """ Instantiate a new StatsArray3D object with the same statistical results. """
        obj = StatsArray3D(self.nx, self.ny, self.nz)
        obj.mean_mv[:] = self.mean_mv[:]
        obj.variance_mv[:] = self.variance_mv[:]
        obj.samples_mv[:] = self.samples_mv[:]
        return obj

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef object add_sample(self, int x, int y, int z, double sample):
        """
        Add a single sample to the StatsArray3D results stored at element x, y, z.

        :param int x: The x position index where the sample should be added.
        :param int y: The y position index where the sample should be added.
        :param int z: The z position index where the sample should be added.
        :param float sample: The sample value to be added.
        """

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
    @cython.initializedcheck(False)
    cpdef object combine_samples(self, int x, int y, int z, double mean, double variance, int sample_count):
        """
        Combine the statistics from a given set of samples with the results already stored in
        this StatsArray at index position x, y, z.

        :param int x: The x index position where these results are to be added.
        :param int y: The y index position where these results are to be added.
        :param int z: The z index position where these results are to be added.
        :param float mean: The mean of the new samples
        :param float variance: The variance of the new samples
        :param int sample_count: The number of new samples that were taken.
        """

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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef double error(self, int x, int y, int z):
        """
        Compute the standard error of the results at index position x, y, z.

        :param int x: The x index position at which to compute the standard error.
        :param int y: The y index position at which to compute the standard error.
        :param int z: The z index position at which to compute the standard error.
        """
        self._bounds_check(x, y, z)
        return _std_error(self.variance_mv[x, y, z], self.samples_mv[x, y, z])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef ndarray errors(self):
        """
        Compute the standard errors of all the results stored in this StatsArray.

        :rtype: ndarray
        """

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

    cdef void _new_buffers(self):
        self.mean = zeros((self.nx, self.ny, self.nz), dtype=float64)
        self.variance = zeros((self.nx, self.ny, self.nz), dtype=float64)
        self.samples = zeros((self.nx, self.ny, self.nz), dtype=int32)
        self.mean_mv = self.mean
        self.variance_mv = self.variance
        self.samples_mv = self.samples

    cdef object _bounds_check(self, int x, int y, int z):

        if x < 0 or x >= self.nx:
            raise ValueError("Index x is out of range.")

        if y < 0 or y >= self.ny:
            raise ValueError("Index y is out of range.")

        if z < 0 or z >= self.nz:
            raise ValueError("Index z is out of range.")


@cython.cdivision(True)
cdef double _std_error(double v, int n) nogil:
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
cdef void _add_sample(double sample, double *m, double *v, int *n) nogil:
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
cdef void _combine_samples(double mx, double vx, int nx, double my, double vy, int ny, double *mt, double *vt, int *nt) nogil:
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
        swap_int(&nx, &ny)
        swap_double(&mx, &my)
        swap_double(&vx, &vy)

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
