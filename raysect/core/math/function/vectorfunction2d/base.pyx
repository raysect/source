# cython: language_level=3

# Copyright (c) 2014-2020, Dr Alex Meakins, Raysect Project
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
from raysect.core.math.function.function2d.base cimport Function2D
from raysect.core.math.function.function2d.autowrap cimport autowrap_function2d
from raysect.core.math.function.function2d.base cimport is_callable as f2d_is_callable
from .autowrap cimport autowrap_vectorfunction2d


cdef class VectorFunction2D:
    """
    Cython optimised class for representing an arbitrary 2D vector function.

    Using __call__() in cython is slow. This class provides an overloadable
    cython cdef evaluate() method which has much less overhead than a python
    function call.

    For use in cython code only, this class cannot be extended via python.

    To create a new function object, inherit this class and implement the
    evaluate() method. The new function object can then be used with any code
    that accepts a function object returning a Vector3D.
    """

    cdef Vector3D evaluate(self, double x, double y):
        raise NotImplementedError("The evaluate() method has not been implemented.")

    def __call__(self, double x, double y):
        """ Evaluate the function f(x, y)

        :param float x: function parameter x
        :param float y: function parameter y
        :rtype: float
        """
        return self.evaluate(x, y)

    def __add__(object a, object b):
        if is_callable(a) or isinstance(a, Vector3D):
            if is_callable(b) or isinstance(b, Vector3D):
                return AddVectorFunction2D(a, b)
        return NotImplemented

    def __sub__(object a, object b):
        if is_callable(a) or isinstance(a, Vector3D):
            if is_callable(b) or isinstance(b, Vector3D):
                return SubtractVectorFunction2D(a, b)
        return NotImplemented

    def __mul__(object a, object b):
        if is_callable(a) or isinstance(a, Vector3D):
            if f2d_is_callable(b) or isinstance(b, numbers.Real):
                return MultiplyVectorFunction2D(a, b)
        if is_callable(b) or isinstance(b, Vector3D):
            if f2d_is_callable(a) or isinstance(a, numbers.Real):
                return MultiplyVectorFunction2D(b, a)
        return NotImplemented

    def __truediv__(object a, object b):
        if is_callable(a) or isinstance(a, Vector3D):
            if f2d_is_callable(b) or isinstance(b, numbers.Real):
                return DivideVectorFunction2D(a, b)
        return NotImplemented

    def __neg__(self):
        return NegVectorFunction2D(self)

    def __richcmp__(self, object other, int op):
        if is_callable(other) or isinstance(other, Vector3D):
            if op == Py_EQ:
                return EqualsVectorFunction2D(self, other)
            if op == Py_NE:
                return NotEqualsVectorFunction2D(self, other)
        return NotImplemented


cdef class AddVectorFunction2D(VectorFunction2D):
    """
    A VectorFunction2D class that implements the addition of the results of two VectorFunction2D objects: f1() + f2()

    This class is not intended to be used directly, but rather returned as the result of an __add__() call on a
    VectorFunction2D object.

    :param object function1: A VectorFunction2D object or Python callable.
    :param object function2: A VectorFunction2D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_vectorfunction2d(function1)
        self._function2 = autowrap_vectorfunction2d(function2)

    cdef Vector3D evaluate(self, double x, double y):
        return self._function1.evaluate(x, y).add(self._function2.evaluate(x, y))


cdef class SubtractVectorFunction2D(VectorFunction2D):
    """
    A VectorFunction2D class that implements the subtraction of the results of two VectorFunction2D objects: f1() - f2()

    This class is not intended to be used directly, but rather returned as the result of a __sub__() call on a
    VectorFunction2D object.

    :param object function1: A VectorFunction2D object or Python callable.
    :param object function2: A VectorFunction2D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_vectorfunction2d(function1)
        self._function2 = autowrap_vectorfunction2d(function2)

    cdef Vector3D evaluate(self, double x, double y):
        return self._function1.evaluate(x, y).sub(self._function2.evaluate(x, y))


cdef class MultiplyVectorFunction2D(VectorFunction2D):
    """
    A VectorFunction2D class that implements the multiplication of the result of a VectorFunction2D object with the result of a Function2D object scalar: f1() * f2().

    This class is not intended to be used directly, but rather returned as the result of a __sub__() call on a
    VectorFunction2D object.

    :param object function1: A VectorFunction2D object or Python callable returning a Vector3D.
    :param object function2: A Function2D object or Python callable returning a double.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_vectorfunction2d(function1)
        self._function2 = autowrap_function2d(function2)

    cdef Vector3D evaluate(self, double x, double y):
        return self._function1.evaluate(x, y).mul(self._function2.evaluate(x, y))


cdef class DivideVectorFunction2D(VectorFunction2D):
    """
    A VectorFunction2D class that implements the division of the results of a VectorFunction2D object and a Function2D object: f1() / f2()

    This class is not intended to be used directly, but rather returned as the result of a __truediv__() call on a
    VectorFunction2D object.

    :param object function1: A VectorFunction2D object or Python callable returning a Vector3D.
    :param object function2: A Function2D object or Python callable returning a double.
    """

    def __init__(self, object function1, object function2):
        self._function1 = autowrap_vectorfunction2d(function1)
        self._function2 = autowrap_function2d(function2)

    @cython.cdivision(True)
    cdef Vector3D evaluate(self, double x, double y):
        cdef double denominator = self._function2.evaluate(x, y)
        if denominator == 0.0:
            raise ZeroDivisionError("Function used as the denominator of the division returned a zero value.")
        return self._function1.evaluate(x, y).div(denominator)


cdef class NegVectorFunction2D(VectorFunction2D):
    """
    A VectorFunction2D class that implements the negation of the result of a VectorFunction2D: -f().

    This class is not intended to be used directly, but rather returned as the result of a __neg__() call on a
    VectorFunction2D object.

    :param object function1: A vectorFunction2D object or Python callable.
    """
    def __init__(self, object function1):
        self._function1 = autowrap_vectorfunction2d(function1)

    cdef Vector3D evaluate(self, double x, double y):
        return self._function1.evaluate(x, y).neg()


cdef class EqualsVectorFunction2D(Function2D):
    """
    A Function2D class that tests the equality of the results of two VectorFunction2D objects: f1() == f2()

    This class is not intended to be used directly, but rather returned as the result of an __eq__() call on a
    VectorFunction2D object.

    N.B. This is a Function2D class, so returns a double rather than a Vector3D.

    :param object function1: A VectorFunction2D object or Python callable.
    :param object function2: A VectorFunction2D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_vectorfunction2d(function1)
        self._function2 = autowrap_vectorfunction2d(function2)

    cdef double evaluate(self, double x, double y) except? -1e999:
        cdef Vector3D v1, v2
        v1 = self._function1.evaluate(x, y)
        v2 = self._function2.evaluate(x, y)
        return 1.0 if (v1.x == v2.x and v1.y == v2.y and v1.z == v2.z) else 0.0


cdef class NotEqualsVectorFunction2D(Function2D):
    """
    A Function2D class that tests the inequality of the results of two VectorFunction2D objects: f1() != f2()

    This class is not intended to be used directly, but rather returned as the result of an __neq__() call on a
    VectorFunction2D object.

    N.B. This is a Function2D class, so returns a double rather than a Vector3D.

    :param object function1: A VectorFunction2D object or Python callable.
    :param object function2: A VectorFunction2D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_vectorfunction2d(function1)
        self._function2 = autowrap_vectorfunction2d(function2)

    cdef double evaluate(self, double x, double y) except? -1e999:
        cdef Vector3D v1, v2
        v1 = self._function1.evaluate(x, y)
        v2 = self._function2.evaluate(x, y)
        return 0.0 if (v1.x == v2.x and v1.y == v2.y and v1.z == v2.z) else 1.0


cdef bint is_callable(object f):
    """
    Tests if an object is a python callable or VectorFunction2D object.

    :param object f: Object to test.
    :return: True if callable, False otherwise.
    """
    # VectorFunction2D and Function2D objects are not equivalent.
    if isinstance(f, Function2D):
        return False
    return isinstance(f, VectorFunction2D) or callable(f)
