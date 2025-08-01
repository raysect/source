# cython: language_level=3

# Copyright (c) 2014-2025, Dr Alex Meakins, Raysect Project
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

import numbers
from cpython.object cimport Py_LT, Py_EQ, Py_GT, Py_LE, Py_NE, Py_GE
cimport cython
from libc.math cimport floor
from raysect.core.math.vector cimport Vector3D
from raysect.core.math.function.float.function1d.base cimport is_callable as float_is_callable
from raysect.core.math.function.float.function1d.autowrap cimport autowrap_function1d as autowrap_floatfunction1d
from .autowrap cimport autowrap_function1d


cdef class Function1D(Vector3DFunction):
    """
    Cython optimised class for representing an arbitrary 1D vector function.

    Using __call__() in cython is slow. This class provides an overloadable
    cython cdef evaluate() method which has much less overhead than a python
    function call.

    For use in cython code only, this class cannot be extended via python.

    To create a new function object, inherit this class and implement the
    evaluate() method. The new function object can then be used with any code
    that accepts a function object returning a Vector3D.
    """

    cdef Vector3D evaluate(self, double x):
        raise NotImplementedError("The evaluate() method has not been implemented.")

    def __call__(self, double x):
        """ Evaluate the function f(x)

        :param float x: function parameter x
        :rtype: float
        """
        return self.evaluate(x)

    def __add__(self, object b):
        if is_callable(b) or isinstance(b, Vector3D):
            return AddFunction1D(self, b)
        return NotImplemented
    
    def __radd__(self, object a):
        return self.__add__(a)

    def __sub__(self, object b):
        if is_callable(b) or isinstance(b, Vector3D):
            return SubtractFunction1D(self, b)
        return NotImplemented
    
    def __rsub__(self, object a):
        if is_callable(a) or isinstance(a, Vector3D):
            return SubtractFunction1D(a, self)

    def __mul__(self, object b):
        if float_is_callable(b) or isinstance(b, numbers.Real):
            return MultiplyFunction1D(self, b)
        return NotImplemented
    
    def __rmul__(self, object a):
        return self.__mul__(a)

    def __truediv__(self, object b):
        if float_is_callable(b) or isinstance(b, numbers.Real):
            return DivideFunction1D(self, b)
        return NotImplemented

    def __neg__(self):
        return NegFunction1D(self)

    def __richcmp__(self, object other, int op):
        if is_callable(other) or isinstance(other, Vector3D):
            if op == Py_EQ:
                return EqualsFunction1D(self, other)
            if op == Py_NE:
                return NotEqualsFunction1D(self, other)
        return NotImplemented


cdef class AddFunction1D(Function1D):
    """
    A vector3d.Function1D class that implements the addition of the results of two vector3d.Function1D objects: f1() + f2()

    This class is not intended to be used directly, but rather returned as the result of an __add__() call on a
    Function1D object.

    :param object function1: A vector3d.Function1D object or Python callable.
    :param object function2: A vector3d.Function1D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function1d(function1)
        self._function2 = autowrap_function1d(function2)

    cdef Vector3D evaluate(self, double x):
        return self._function1.evaluate(x).add(self._function2.evaluate(x))


cdef class SubtractFunction1D(Function1D):
    """
    A vector3d.Function1D class that implements the subtraction of the results of two vector3d.Function1D objects: f1() - f2()

    This class is not intended to be used directly, but rather returned as the result of a __sub__() call on a
    Function1D object.

    :param object function1: A vector3d.Function1D object or Python callable.
    :param object function2: A vector3d.Function1D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function1d(function1)
        self._function2 = autowrap_function1d(function2)

    cdef Vector3D evaluate(self, double x):
        return self._function1.evaluate(x).sub(self._function2.evaluate(x))


cdef class MultiplyFunction1D(Function1D):
    """
    A vector3d.Function1D class that implements the multiplication of the result of a vector3d.Function1D object with the result of a float.Function1D object scalar: f1() * f2().

    This class is not intended to be used directly, but rather returned as the result of a __sub__() call on a
    vector3d.Function1D object.

    :param object function1: A vector3d.Function1D object or Python callable returning a Vector3D.
    :param object function2: A float.Function1D object or Python callable returning a double.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function1d(function1)
        self._function2 = autowrap_floatfunction1d(function2)

    cdef Vector3D evaluate(self, double x):
        return self._function1.evaluate(x).mul(self._function2.evaluate(x))


cdef class DivideFunction1D(Function1D):
    """
    A vector3d.Function1D class that implements the division of the results of a vector3d.Function1D object and a float.Function1D object: f1() / f2()

    This class is not intended to be used directly, but rather returned as the result of a __truediv__() call on a
    vector3d.Function1D object.

    :param object function1: A vector3d.Function1D object or Python callable returning a Vector3D.
    :param object function2: A float.Function1D object or Python callable returning a double.
    """

    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function1d(function1)
        self._function2 = autowrap_floatfunction1d(function2)

    @cython.cdivision(True)
    cdef Vector3D evaluate(self, double x):
        cdef double denominator = self._function2.evaluate(x)
        if denominator == 0.0:
            raise ZeroDivisionError("Function used as the denominator of the division returned a zero value.")
        return self._function1.evaluate(x).div(denominator)


cdef class NegFunction1D(Function1D):
    """
    A vector3d.Function1D class that implements the negation of the result of a vector3d.Function1D: -f().

    This class is not intended to be used directly, but rather returned as the result of a __neg__() call on a
    vector3d.Function1D object.

    :param object function: A vector3d.Function1D object or Python callable.
    """
    def __init__(self, object function):
        self._function = autowrap_function1d(function)

    cdef Vector3D evaluate(self, double x):
        return self._function.evaluate(x).neg()


cdef class EqualsFunction1D(FloatFunction1D):
    """
    A float.Function1D class that tests the equality of the results of two vector3d.Function1D objects: f1() == f2()

    This class is not intended to be used directly, but rather returned as the result of an __eq__() call on a
    vector3d.Function1D object.

    N.B. This is a float.Function1D class, so returns a double rather than a Vector3D.

    :param object function1: A vector3d.Function1D object or Python callable.
    :param object function2: A vector3d.Function1D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function1d(function1)
        self._function2 = autowrap_function1d(function2)

    cdef double evaluate(self, double x) except? -1e999:
        cdef Vector3D v1, v2
        v1 = self._function1.evaluate(x)
        v2 = self._function2.evaluate(x)
        return 1.0 if (v1.x == v2.x and v1.y == v2.y and v1.z == v2.z) else 0.0


cdef class NotEqualsFunction1D(FloatFunction1D):
    """
    A float.Function1D class that tests the inequality of the results of two vector3d.Function1D objects: f1() != f2()

    This class is not intended to be used directly, but rather returned as the result of an __neq__() call on a
    vector3d.Function1D object.

    N.B. This is a float.Function1D class, so returns a double rather than a Vector3D.

    :param object function1: A vector3d.Function1D object or Python callable.
    :param object function2: A vector3d.Function1D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function1d(function1)
        self._function2 = autowrap_function1d(function2)

    cdef double evaluate(self, double x) except? -1e999:
        cdef Vector3D v1, v2
        v1 = self._function1.evaluate(x)
        v2 = self._function2.evaluate(x)
        return 0.0 if (v1.x == v2.x and v1.y == v2.y and v1.z == v2.z) else 1.0
