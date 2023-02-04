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

from raysect.core.math.vector cimport Vector3D
from raysect.core.math.function.base cimport Function
from raysect.core.math.function.vector3d.base cimport Vector3DFunction
from raysect.core.math.function.float cimport Function3D as FloatFunction3D


cdef class Function3D(Vector3DFunction):
    cdef Vector3D evaluate(self, double x, double y, double z)


cdef class AddFunction3D(Function3D):
    cdef Function3D _function1, _function2


cdef class SubtractFunction3D(Function3D):
    cdef Function3D _function1, _function2


cdef class MultiplyFunction3D(Function3D):
    cdef Function3D _function1
    cdef FloatFunction3D _function2


cdef class DivideFunction3D(Function3D):
    cdef Function3D _function1
    cdef FloatFunction3D _function2


cdef class NegFunction3D(Function3D):
    cdef Function3D _function


cdef class EqualsFunction3D(FloatFunction3D):
    cdef Function3D _function1, _function2


cdef class NotEqualsFunction3D(FloatFunction3D):
    cdef Function3D _function1, _function2


cdef inline bint is_callable(object f):
    """
    Tests if an object is a python callable or a vector3d.Function3D object.

    :param object f: Object to test.
    :return: True if callable, False otherwise.
    """
    if isinstance(f, Function3D):
        return True

    # other function classes are incompatible
    if isinstance(f, Function):
        return False

    return callable(f)
