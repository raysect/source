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

import numbers
from cpython.object cimport Py_LT, Py_EQ, Py_GT, Py_LE, Py_NE, Py_GE
cimport cython
from libc.math cimport floor
from raysect.core.math.vector cimport Vector3D
from raysect.core.math.function.float.function3d.base cimport is_callable as float_is_callable
from raysect.core.math.function.float.function3d.autowrap cimport autowrap_function3d as autowrap_floatfunction3d
from .autowrap cimport autowrap_function3d


cdef class Function3D(Vector3DFunction):
    """
    Cython optimised class for representing an arbitrary 3D vector function.

    Using __call__() in cython is slow. This class provides an overloadable
    cython cdef evaluate() method which has much less overhead than a python
    function call.

    For use in cython code only, this class cannot be extended via python.

    To create a new function object, inherit this class and implement the
    evaluate() method. The new function object can then be used with any code
    that accepts a function object returning a Vector3D.
    """

    cdef Vector3D evaluate(self, double x, double y, double z):
        raise NotImplementedError("The evaluate() method has not been implemented.")

    def __call__(self, double x, double y, double z):
        """ Evaluate the function f(x, y, z)

        :param float x: function parameter x
        :param float y: function parameter y
        :rtype: float
        """
        return self.evaluate(x, y, z)

    def __add__(object a, object b):
        if is_callable(a) or isinstance(a, Vector3D):
            if is_callable(b) or isinstance(b, Vector3D):
                return AddFunction3D(a, b)
        return NotImplemented

    def __sub__(object a, object b):
        if is_callable(a) or isinstance(a, Vector3D):
            if is_callable(b) or isinstance(b, Vector3D):
                return SubtractFunction3D(a, b)
        return NotImplemented

    def __mul__(object a, object b):
        if is_callable(a) or isinstance(a, Vector3D):
            if float_is_callable(b) or isinstance(b, numbers.Real):
                return MultiplyFunction3D(a, b)
        if is_callable(b) or isinstance(b, Vector3D):
            if float_is_callable(a) or isinstance(a, numbers.Real):
                return MultiplyFunction3D(b, a)
        return NotImplemented

    def __truediv__(object a, object b):
        if is_callable(a) or isinstance(a, Vector3D):
            if float_is_callable(b) or isinstance(b, numbers.Real):
                return DivideFunction3D(a, b)
        return NotImplemented

    def __neg__(self):
        return NegFunction3D(self)

    def __richcmp__(self, object other, int op):
        if is_callable(other) or isinstance(other, Vector3D):
            if op == Py_EQ:
                return EqualsFunction3D(self, other)
            if op == Py_NE:
                return NotEqualsFunction3D(self, other)
        return NotImplemented


cdef class AddFunction3D(Function3D):
    """
    A vector3d.Function3D class that implements the addition of the results of two vector3d.Function3D objects: f1() + f2()

    This class is not intended to be used directly, but rather returned as the result of an __add__() call on a
    Function3D object.

    :param object function1: A vector3d.Function3D object or Python callable.
    :param object function2: A vector3d.Function3D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function3d(function1)
        self._function2 = autowrap_function3d(function2)

    cdef Vector3D evaluate(self, double x, double y, double z):
        return self._function1.evaluate(x, y, z).add(self._function2.evaluate(x, y, z))


cdef class SubtractFunction3D(Function3D):
    """
    A vector3d.Function3D class that implements the subtraction of the results of two vector3d.Function3D objects: f1() - f2()

    This class is not intended to be used directly, but rather returned as the result of a __sub__() call on a
    Function3D object.

    :param object function1: A vector3d.Function3D object or Python callable.
    :param object function2: A vector3d.Function3D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function3d(function1)
        self._function2 = autowrap_function3d(function2)

    cdef Vector3D evaluate(self, double x, double y, double z):
        return self._function1.evaluate(x, y, z).sub(self._function2.evaluate(x, y, z))


cdef class MultiplyFunction3D(Function3D):
    """
    A vector3d.Function3D class that implements the multiplication of the result of a vector3d.Function3D object with the result of a float.Function3D object scalar: f1() * f2().

    This class is not intended to be used directly, but rather returned as the result of a __sub__() call on a
    vector3d.Function3D object.

    :param object function1: A vector3d.Function3D object or Python callable returning a Vector3D.
    :param object function2: A float.Function3D object or Python callable returning a double.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function3d(function1)
        self._function2 = autowrap_floatfunction3d(function2)

    cdef Vector3D evaluate(self, double x, double y, double z):
        return self._function1.evaluate(x, y, z).mul(self._function2.evaluate(x, y, z))


cdef class DivideFunction3D(Function3D):
    """
    A vector3d.Function3D class that implements the division of the results of a vector3d.Function3D object and a float.Function3D object: f1() / f2()

    This class is not intended to be used directly, but rather returned as the result of a __truediv__() call on a
    vector3d.Function3D object.

    :param object function1: A vector3d.Function3D object or Python callable returning a Vector3D.
    :param object function2: A float.Function3D object or Python callable returning a double.
    """

    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function3d(function1)
        self._function2 = autowrap_floatfunction3d(function2)

    @cython.cdivision(True)
    cdef Vector3D evaluate(self, double x, double y, double z):
        cdef double denominator = self._function2.evaluate(x, y, z)
        if denominator == 0.0:
            raise ZeroDivisionError("Function used as the denominator of the division returned a zero value.")
        return self._function1.evaluate(x, y, z).div(denominator)


cdef class NegFunction3D(Function3D):
    """
    A vector3d.Function3D class that implements the negation of the result of a vector3d.Function3D: -f().

    This class is not intended to be used directly, but rather returned as the result of a __neg__() call on a
    vector3d.Function3D object.

    :param object function: A vector3d.Function3D object or Python callable.
    """
    def __init__(self, object function):
        self._function = autowrap_function3d(function)

    cdef Vector3D evaluate(self, double x, double y, double z):
        return self._function.evaluate(x, y, z).neg()


cdef class EqualsFunction3D(FloatFunction3D):
    """
    A float.Function3D class that tests the equality of the results of two vector3d.Function3D objects: f1() == f2()

    This class is not intended to be used directly, but rather returned as the result of an __eq__() call on a
    vector3d.Function3D object.

    N.B. This is a float.Function3D class, so returns a double rather than a Vector3D.

    :param object function1: A vector3d.Function3D object or Python callable.
    :param object function2: A vector3d.Function3D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function3d(function1)
        self._function2 = autowrap_function3d(function2)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        cdef Vector3D v1, v2
        v1 = self._function1.evaluate(x, y, z)
        v2 = self._function2.evaluate(x, y, z)
        return 1.0 if (v1.x == v2.x and v1.y == v2.y and v1.z == v2.z) else 0.0


cdef class NotEqualsFunction3D(FloatFunction3D):
    """
    A float.Function3D class that tests the inequality of the results of two vector3d.Function3D objects: f1() != f2()

    This class is not intended to be used directly, but rather returned as the result of an __neq__() call on a
    vector3d.Function3D object.

    N.B. This is a float.Function3D class, so returns a double rather than a Vector3D.

    :param object function1: A vector3d.Function3D object or Python callable.
    :param object function2: A vector3d.Function3D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function3d(function1)
        self._function2 = autowrap_function3d(function2)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        cdef Vector3D v1, v2
        v1 = self._function1.evaluate(x, y, z)
        v2 = self._function2.evaluate(x, y, z)
        return 0.0 if (v1.x == v2.x and v1.y == v2.y and v1.z == v2.z) else 1.0
