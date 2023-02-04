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

from raysect.core.math.function.base cimport Function
from raysect.core.math.function.float.base cimport FloatFunction


cdef class Function2D(FloatFunction):

    cdef double evaluate(self, double x, double y) except? -1e999


cdef class AddFunction2D(Function2D):
    cdef Function2D _function1, _function2


cdef class SubtractFunction2D(Function2D):
    cdef Function2D _function1, _function2


cdef class MultiplyFunction2D(Function2D):
    cdef Function2D _function1, _function2


cdef class DivideFunction2D(Function2D):
    cdef Function2D _function1, _function2


cdef class ModuloFunction2D(Function2D):
    cdef Function2D _function1, _function2


cdef class PowFunction2D(Function2D):
    cdef Function2D _function1, _function2


cdef class AbsFunction2D(Function2D):
    cdef Function2D _function


cdef class EqualsFunction2D(Function2D):
    cdef Function2D _function1, _function2


cdef class NotEqualsFunction2D(Function2D):
    cdef Function2D _function1, _function2


cdef class LessThanFunction2D(Function2D):
    cdef Function2D _function1, _function2


cdef class GreaterThanFunction2D(Function2D):
    cdef Function2D _function1, _function2


cdef class LessEqualsFunction2D(Function2D):
    cdef Function2D _function1, _function2


cdef class GreaterEqualsFunction2D(Function2D):
    cdef Function2D _function1, _function2


cdef class AddScalar2D(Function2D):
    cdef double _value
    cdef Function2D _function


cdef class SubtractScalar2D(Function2D):
    cdef double _value
    cdef Function2D _function


cdef class MultiplyScalar2D(Function2D):
    cdef double _value
    cdef Function2D _function


cdef class DivideScalar2D(Function2D):
    cdef double _value
    cdef Function2D _function


cdef class ModuloScalarFunction2D(Function2D):
    cdef double _value
    cdef Function2D _function


cdef class ModuloFunctionScalar2D(Function2D):
    cdef double _value
    cdef Function2D _function


cdef class PowScalarFunction2D(Function2D):
    cdef double _value
    cdef Function2D _function


cdef class PowFunctionScalar2D(Function2D):
    cdef double _value
    cdef Function2D _function


cdef class EqualsScalar2D(Function2D):
    cdef double _value
    cdef Function2D _function


cdef class NotEqualsScalar2D(Function2D):
    cdef double _value
    cdef Function2D _function


cdef class LessThanScalar2D(Function2D):
    cdef double _value
    cdef Function2D _function


cdef class GreaterThanScalar2D(Function2D):
    cdef double _value
    cdef Function2D _function


cdef class LessEqualsScalar2D(Function2D):
    cdef double _value
    cdef Function2D _function


cdef class GreaterEqualsScalar2D(Function2D):
    cdef double _value
    cdef Function2D _function


cdef inline bint is_callable(object f):
    """
    Tests if an object is a python callable or a Function2D object.

    :param object f: Object to test.
    :return: True if callable, False otherwise.
    """
    if isinstance(f, Function2D):
        return True

    # other function classes are incompatible
    if isinstance(f, Function):
        return False

    return callable(f)

