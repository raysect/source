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

from numpy cimport ndarray


cdef class StatsBin:

    cdef:
        readonly double mean
        readonly double variance
        readonly int samples

    cpdef object clear(self)

    cpdef StatsBin copy(self)

    cpdef object add_sample(self, double sample)

    cpdef object combine_samples(self, double mean, double variance, int sample_count)

    cpdef double error(self)


cdef class StatsArray1D:

    cdef:
        readonly int length
        readonly ndarray mean
        readonly ndarray variance
        readonly ndarray samples
        double[::1] mean_mv
        double[::1] variance_mv
        int[::1] samples_mv

    cpdef object clear(self)

    cpdef StatsArray1D copy(self)

    cpdef object add_sample(self, int x, double sample)

    cpdef object combine_samples(self, int x, double mean, double variance, int sample_count)

    cpdef double error(self, int x)

    cpdef ndarray errors(self)

    cdef void _new_buffers(self)

    cdef object _bounds_check(self, int x)


cdef class StatsArray2D:

    cdef:
        readonly int nx, ny
        readonly ndarray mean
        readonly ndarray variance
        readonly ndarray samples
        double[:,::1] mean_mv
        double[:,::1] variance_mv
        int[:,::1] samples_mv

    cpdef object clear(self)

    cpdef StatsArray2D copy(self)

    cpdef object add_sample(self, int x, int y, double sample)

    cpdef object combine_samples(self, int x, int y, double mean, double variance, int sample_count)

    cpdef double error(self, int x, int y)

    cpdef ndarray errors(self)

    cdef void _new_buffers(self)

    cdef object _bounds_check(self, int x, int y)


cdef class StatsArray3D:

    cdef:
        readonly int nx, ny, nz
        readonly ndarray mean
        readonly ndarray variance
        readonly ndarray samples
        double[:,:,::1] mean_mv
        double[:,:,::1] variance_mv
        int[:,:,::1] samples_mv

    cpdef object clear(self)

    cpdef StatsArray3D copy(self)

    cpdef object add_sample(self, int x, int y, int z, double sample)

    cpdef object combine_samples(self, int x, int y, int z, double mean, double variance, int sample_count)

    cpdef double error(self, int x, int y, int z)

    cpdef ndarray errors(self)

    cdef void _new_buffers(self)

    cdef object _bounds_check(self, int x, int y, int z)
