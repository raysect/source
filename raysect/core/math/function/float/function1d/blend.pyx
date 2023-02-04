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

from raysect.core.math.function.float.function1d.autowrap cimport autowrap_function1d
from raysect.core.math.cython cimport clamp


cdef class Blend1D(Function1D):
    """
    Performs a linear interpolation between two scalar functions, modulated by a 3rd scalar function.

    The value of the scalar mask function is used to interpolated between the
    values returned by the two functions. Mathematically the value returned by
    this function is as follows:

    .. math::
        v = (1 - f_m(x)) f_1(x) + f_m(x) f_2(x)

    The value of the mask function is clamped to the range [0, 1] if the sampled
    value exceeds the required range.
    """

    def __init__(self, object f1, object f2, object mask):
        """
        :param float.Function1D f1: First scalar function.
        :param float.Function1D f2: Second scalar function.
        :param float.Function1D mask: Scalar function returning a value in the range [0, 1].
        """

        self._f1 = autowrap_function1d(f1)
        self._f2 = autowrap_function1d(f2)
        self._mask = autowrap_function1d(mask)

    cdef double evaluate(self, double x) except? -1e999:

        cdef double t = clamp(self._mask.evaluate(x), 0.0, 1.0)

        # sample endpoints directly
        if t == 0:
            return self._f1.evaluate(x)

        if t == 1:
            return self._f2.evaluate(x)

        # lerp between function values
        cdef double f1 = self._f1.evaluate(x)
        cdef double f2 = self._f2.evaluate(x)
        return (1 - t) * f1 + t * f2
