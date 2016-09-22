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

from numpy import zeros, float64, int64
from numpy cimport ndarray


cdef class Frame2D:

    cdef:
        int nx, ny
        readonly ndarray value
        readonly ndarray variance
        readonly ndarray samples

    def __init__(self, pixels):

        nx, ny = pixels
        if nx < 1 or ny < 1:
            raise ValueError("Pixels must be a tuple of x and y dimensions, both of which must be >= 1.")

        self.nx = nx
        self.ny = ny

        # generate frame buffers
        self._new_buffers(self)

    @property
    def pixels(self):
        return self.nx, self.ny

    cpdef object add_sample(self, x, y, value):
        pass

    cpdef object combine_samples(self, x, y, mean, variance, sample_count):
        pass

    cpdef object clear(self):
        self._new_buffers()

    cdef inline void _new_buffers(self):
        self.value = zeros((self.nx, self.ny), dtype=float64)
        self.variance = zeros((self.nx, self.ny), dtype=float64)
        self.samples = zeros((self.nx, self.ny), dtype=int64)





