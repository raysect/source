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
from .autowrap cimport autowrap_function2d


cdef class Function2D(FloatFunction):
    """
    Cython optimised class for representing an arbitrary 2D function returning a float.

    Using __call__() in cython is slow. This class provides an overloadable
    cython cdef evaluate() method which has much less overhead than a python
    function call.

    For use in cython code only, this class cannot be extended via python.

    To create a new function object, inherit this class and implement the
    evaluate() method. The new function object can then be used with any code
    that accepts a function object.
    """

    cdef double evaluate(self, double x, double y) except? -1e999:
        raise NotImplementedError("The evaluate() method has not been implemented.")

    def __call__(self, double x, double y):
        """ Evaluate the function f(x, y)

        :param float x: function parameter x
        :param float y: function parameter y
        :rtype: float
        """
        return self.evaluate(x, y)

    def __add__(object a, object b):
        if is_callable(a):
            if is_callable(b):
                # a() + b()
                return AddFunction2D(a, b)
            elif isinstance(b, numbers.Real):
                # a() + B -> B + a()
                return AddScalar2D(<double> b, a)
        elif isinstance(a, numbers.Real):
            if is_callable(b):
                # A + b()
                return AddScalar2D(<double> a, b)
        return NotImplemented

    def __sub__(object a, object b):
        if is_callable(a):
            if is_callable(b):
                # a() - b()
                return SubtractFunction2D(a, b)
            elif isinstance(b, numbers.Real):
                # a() - B -> -B + a()
                return AddScalar2D(-(<double> b), a)
        elif isinstance(a, numbers.Real):
            if is_callable(b):
                # A - b()
                return SubtractScalar2D(<double> a, b)
        return NotImplemented

    def __mul__(object a, object b):
        if is_callable(a):
            if is_callable(b):
                # a() * b()
                return MultiplyFunction2D(a, b)
            elif isinstance(b, numbers.Real):
                # a() * B -> B * a()
                return MultiplyScalar2D(<double> b, a)
        elif isinstance(a, numbers.Real):
            if is_callable(b):
                # A * b()
                return MultiplyScalar2D(<double> a, b)
        return NotImplemented

    @cython.cdivision(True)
    def __truediv__(object a, object b):
        cdef double v
        if is_callable(a):
            if is_callable(b):
                # a() / b()
                return DivideFunction2D(a, b)
            elif isinstance(b, numbers.Real):
                # a() / B -> 1/B * a()
                v = <double> b
                if v == 0.0:
                    raise ZeroDivisionError("Scalar used as the denominator of the division is zero valued.")
                return MultiplyScalar2D(1/v, a)
        elif isinstance(a, numbers.Real):
            if is_callable(b):
                # A * b()
                return DivideScalar2D(<double> a, b)
        return NotImplemented

    def __mod__(object a, object b):
        cdef double v
        if is_callable(a):
            if is_callable(b):
                # a() % b()
                return ModuloFunction2D(a, b)
            elif isinstance(b, numbers.Real):
                # a() % B
                v = <double> b
                if v == 0.0:
                    raise ZeroDivisionError("Scalar used as the divisor of the division is zero valued.")
                return ModuloFunctionScalar2D(a, v)
        elif isinstance(a, numbers.Real):
            if is_callable(b):
                # A % b()
                return ModuloScalarFunction2D(<double> a, b)
        return NotImplemented

    def __neg__(self):
        return MultiplyScalar2D(-1, self)

    def __pow__(object a, object b, object c):
        if c is not None:
            # Optimised implementation of pow(a, b, c) not available: fall back
            # to general implementation
            return (a ** b) % c
        if is_callable(a):
            if is_callable(b):
                # a() ** b()
                return PowFunction2D(a, b)
            elif isinstance(b, numbers.Real):
                # a() ** b
                return PowFunctionScalar2D(a, <double> b)
        elif isinstance(a, numbers.Real):
            if is_callable(b):
                # a ** b()
                return PowScalarFunction2D(<double> a, b)
        return NotImplemented

    def __abs__(self):
        return AbsFunction2D(self)

    def __richcmp__(self, object other, int op):
        if is_callable(other):
            if op == Py_EQ:
                return EqualsFunction2D(self, other)
            if op == Py_NE:
                return NotEqualsFunction2D(self, other)
            if op == Py_LT:
                return LessThanFunction2D(self, other)
            if op == Py_GT:
                return GreaterThanFunction2D(self, other)
            if op == Py_LE:
                return LessEqualsFunction2D(self, other)
            if op == Py_GE:
                return GreaterEqualsFunction2D(self, other)
        if isinstance(other, numbers.Real):
            if op == Py_EQ:
                return EqualsScalar2D(<double> other, self)
            if op == Py_NE:
                return NotEqualsScalar2D(<double> other, self)
            if op == Py_LT:
                # f() < K -> K > f
                return GreaterThanScalar2D(<double> other, self)
            if op == Py_GT:
                # f() > K -> K < f
                return LessThanScalar2D(<double> other, self)
            if op == Py_LE:
                # f() <= K -> K >= f
                return GreaterEqualsScalar2D(<double> other, self)
            if op == Py_GE:
                # f() >= K -> K <= f
                return LessEqualsScalar2D(<double> other, self)
        return NotImplemented


cdef class AddFunction2D(Function2D):
    """
    A Function2D class that implements the addition of the results of two Function2D objects: f1() + f2()

    This class is not intended to be used directly, but rather returned as the result of an __add__() call on a
    Function2D object.

    :param object function1: A Function2D object or Python callable.
    :param object function2: A Function2D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function2d(function1)
        self._function2 = autowrap_function2d(function2)

    cdef double evaluate(self, double x, double y) except? -1e999:
        return self._function1.evaluate(x, y) + self._function2.evaluate(x, y)


cdef class SubtractFunction2D(Function2D):
    """
    A Function2D class that implements the subtraction of the results of two Function2D objects: f1() - f2()

    This class is not intended to be used directly, but rather returned as the result of a __sub__() call on a
    Function2D object.

    :param object function1: A Function2D object or Python callable.
    :param object function2: A Function2D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function2d(function1)
        self._function2 = autowrap_function2d(function2)

    cdef double evaluate(self, double x, double y) except? -1e999:
        return self._function1.evaluate(x, y) - self._function2.evaluate(x, y)


cdef class MultiplyFunction2D(Function2D):
    """
    A Function2D class that implements the multiplication of the results of two Function2D objects: f1() * f2()

    This class is not intended to be used directly, but rather returned as the result of a __mul__() call on a
    Function2D object.

    :param object function1: A Function2D object or Python callable.
    :param object function2: A Function2D object or Python callable.
    """

    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function2d(function1)
        self._function2 = autowrap_function2d(function2)

    cdef double evaluate(self, double x, double y) except? -1e999:
        return self._function1.evaluate(x, y) * self._function2.evaluate(x, y)


cdef class DivideFunction2D(Function2D):
    """
    A Function2D class that implements the division of the results of two Function2D objects: f1() / f2()

    This class is not intended to be used directly, but rather returned as the result of a __truediv__() call on a
    Function2D object.

    :param object function1: A Function2D object or Python callable.
    :param object function2: A Function2D object or Python callable.
    """

    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function2d(function1)
        self._function2 = autowrap_function2d(function2)

    @cython.cdivision(True)
    cdef double evaluate(self, double x, double y) except? -1e999:
        cdef double denominator = self._function2.evaluate(x, y)
        if denominator == 0.0:
            raise ZeroDivisionError("Function used as the denominator of the division returned a zero value.")
        return self._function1.evaluate(x, y) / denominator


cdef class ModuloFunction2D(Function2D):
    """
    A Function2D class that implements the modulo of the results of two Function2D objects: f1() % f2()

    This class is not intended to be used directly, but rather returned as the result of a __mod__() call on a
    Function2D object.

    :param object function1: A Function2D object or Python callable.
    :param object function2: A Function2D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function2d(function1)
        self._function2 = autowrap_function2d(function2)

    @cython.cdivision(True)
    cdef double evaluate(self, double x, double y) except? -1e999:
        cdef double divisor = self._function2.evaluate(x, y)
        if divisor == 0.0:
            raise ZeroDivisionError("Function used as the divisor of the modulo returned a zero value.")
        return self._function1.evaluate(x, y) % divisor


cdef class PowFunction2D(Function2D):
    """
    A Function2D class that implements the pow() operator on two Function2D objects.

    This class is not intended to be used directly, but rather returned as the result of a __pow__() call on a
    Function2D object.

    :param object function1: A Function2D object or Python callable.
    :param object function2: A Function2D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function2d(function1)
        self._function2 = autowrap_function2d(function2)

    cdef double evaluate(self, double x, double y) except? -1e999:
        cdef double base, exponent
        base = self._function1.evaluate(x, y)
        exponent = self._function2.evaluate(x, y)
        if base < 0 and floor(exponent) != exponent:  # Would return a complex value rather than double
            raise ValueError("Negative base and non-integral exponent is not supported")
        if base == 0 and exponent < 0:
            raise ZeroDivisionError("0.0 cannot be raised to a negative power")
        return base ** exponent


cdef class AbsFunction2D(Function2D):
    """
    A Function2D class that implements the absolute value of the result of a Function2D object: abs(f()).

    This class is not intended to be used directly, but rather returned as the
    result of an __abs__() call on a Function2D object.

    :param object function: A Function2D object or Python callable.
    """
    def __init__(self, object function):
        self._function = autowrap_function2d(function)

    cdef double evaluate(self, double x, double y) except? -1e999:
        return abs(self._function.evaluate(x, y))


cdef class EqualsFunction2D(Function2D):
    """
    A Function2D class that tests the equality of the results of two Function2D objects: f1() == f2()

    This class is not intended to be used directly, but rather returned as the result of an __eq__() call on a
    Function2D object.

    :param object function1: A Function2D object or Python callable.
    :param object function2: A Function2D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function2d(function1)
        self._function2 = autowrap_function2d(function2)

    cdef double evaluate(self, double x, double y) except? -1e999:
        return self._function1.evaluate(x, y) == self._function2.evaluate(x, y)


cdef class NotEqualsFunction2D(Function2D):
    """
    A Function2D class that tests the inequality of the results of two Function2D objects: f1() != f2()

    This class is not intended to be used directly, but rather returned as the result of an __ne__() call on a
    Function2D object.

    :param object function1: A Function2D object or Python callable.
    :param object function2: A Function2D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function2d(function1)
        self._function2 = autowrap_function2d(function2)

    cdef double evaluate(self, double x, double y) except? -1e999:
        return self._function1.evaluate(x, y) != self._function2.evaluate(x, y)


cdef class LessThanFunction2D(Function2D):
    """
    A Function2D class that implements < of the results of two Function2D objects: f1() < f2()

    This class is not intended to be used directly, but rather returned as the result of an __lt__() call on a
    Function2D object.

    :param object function1: A Function2D object or Python callable.
    :param object function2: A Function2D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function2d(function1)
        self._function2 = autowrap_function2d(function2)

    cdef double evaluate(self, double x, double y) except? -1e999:
        return self._function1.evaluate(x, y) < self._function2.evaluate(x, y)


cdef class GreaterThanFunction2D(Function2D):
    """
    A Function2D class that implements > of the results of two Function2D objects: f1() > f2()

    This class is not intended to be used directly, but rather returned as the result of a __gt__() call on a
    Function2D object.

    :param object function1: A Function2D object or Python callable.
    :param object function2: A Function2D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function2d(function1)
        self._function2 = autowrap_function2d(function2)

    cdef double evaluate(self, double x, double y) except? -1e999:
        return self._function1.evaluate(x, y) > self._function2.evaluate(x, y)


cdef class LessEqualsFunction2D(Function2D):
    """
    A Function2D class that implements <= of the results of two Function2D objects: f1() <= f2()

    This class is not intended to be used directly, but rather returned as the result of an __le__() call on a
    Function2D object.

    :param object function1: A Function2D object or Python callable.
    :param object function2: A Function2D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function2d(function1)
        self._function2 = autowrap_function2d(function2)

    cdef double evaluate(self, double x, double y) except? -1e999:
        return self._function1.evaluate(x, y) <= self._function2.evaluate(x, y)


cdef class GreaterEqualsFunction2D(Function2D):
    """
    A Function2D class that implements >= of the results of two Function2D objects: f1() >= f2()

    This class is not intended to be used directly, but rather returned as the result of an __ge__() call on a
    Function2D object.

    :param object function1: A Function2D object or Python callable.
    :param object function2: A Function2D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function2d(function1)
        self._function2 = autowrap_function2d(function2)

    cdef double evaluate(self, double x, double y) except? -1e999:
        return self._function1.evaluate(x, y) >= self._function2.evaluate(x, y)


cdef class AddScalar2D(Function2D):
    """
    A Function2D class that implements the addition of scalar and the result of a Function2D object: K + f()

    This class is not intended to be used directly, but rather returned as the result of an __add__() call on a
    Function2D object.

    :param float value: A double value.
    :param object function: A Function2D object or Python callable.
    """

    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function2d(function)

    cdef double evaluate(self, double x, double y) except? -1e999:
        return self._value + self._function.evaluate(x, y)


cdef class SubtractScalar2D(Function2D):
    """
    A Function2D class that implements the subtraction of scalar and the result of a Function2D object: K - f()

    This class is not intended to be used directly, but rather returned as the result of an __sub__() call on a
    Function2D object.

    :param value: A double value.
    :param function: A Function2D object or Python callable.
    """

    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function2d(function)

    cdef double evaluate(self, double x, double y) except? -1e999:
        return self._value - self._function.evaluate(x, y)


cdef class MultiplyScalar2D(Function2D):
    """
    A Function2D class that implements the multiplication of scalar and the result of a Function2D object: K * f()

    This class is not intended to be used directly, but rather returned as the result of an __mul__() call on a
    Function2D object.

    :param value: A double value.
    :param function: A Function2D object or Python callable.
    """

    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function2d(function)

    cdef double evaluate(self, double x, double y) except? -1e999:
        return self._value * self._function.evaluate(x, y)


cdef class DivideScalar2D(Function2D):
    """
    A Function2D class that implements the subtraction of scalar and the result of a Function2D object: K / f()

    This class is not intended to be used directly, but rather returned as the result of an __div__() call on a
    Function2D object.

    :param value: A double value.
    :param function: A Function2D object or Python callable.
    """

    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function2d(function)

    @cython.cdivision(True)
    cdef double evaluate(self, double x, double y) except? -1e999:
        cdef double denominator = self._function.evaluate(x, y)
        if denominator == 0.0:
            raise ZeroDivisionError("Function used as the denominator of the division returned a zero value.")
        return self._value / denominator


cdef class ModuloScalarFunction2D(Function2D):
    """
    A Function2D class that implements the modulo of scalar and the result of a Function2D object: K % f()

    This class is not intended to be used directly, but rather returned as the result of a __mod__() call on a
    Function2D object.

    :param float value: A double value.
    :param object function: A Function2D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function2d(function)

    @cython.cdivision(True)
    cdef double evaluate(self, double x, double y) except? -1e999:
        cdef double divisor = self._function.evaluate(x, y)
        if divisor == 0.0:
            raise ZeroDivisionError("Function used as the divisor of the modulo returned a zero value.")
        return self._value % divisor


cdef class ModuloFunctionScalar2D(Function2D):
    """
    A Function2D class that implements the modulo of the result of a Function2D object and a scalar: f() % K

    This class is not intended to be used directly, but rather returned as the result of a __mod__() call on a
    Function2D object.

    :param object function: A Function2D object or Python callable.
    :param float value: A double value.
    """
    def __init__(self, object function, double value):
        if value == 0:
            raise ValueError("Divisor cannot be zero")
        self._value = value
        self._function = autowrap_function2d(function)

    @cython.cdivision(True)
    cdef double evaluate(self, double x, double y) except? -1e999:
        return self._function.evaluate(x, y) % self._value


cdef class PowScalarFunction2D(Function2D):
    """
    A Function2D class that implements the pow of scalar and the result of a Function2D object: K ** f()

    This class is not intended to be used directly, but rather returned as the result of an __pow__() call on a
    Function2D object.

    :param float value: A double value.
    :param object function: A Function2D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function2d(function)

    cdef double evaluate(self, double x, double y) except? -1e999:
        cdef double exponent = self._function.evaluate(x, y)
        if self._value < 0 and floor(exponent) != exponent:
            raise ValueError("Negative base and non-integral exponent is not supported")
        if self._value == 0 and exponent < 0:
            raise ZeroDivisionError("0.0 cannot be raised to a negative power")
        return self._value ** exponent


cdef class PowFunctionScalar2D(Function2D):
    """
    A Function2D class that implements the pow of the result of a Function2D object and a scalar: f() ** K

    This class is not intended to be used directly, but rather returned as the result of an __pow__() call on a
    Function2D object.

    :param object function: A Function2D object or Python callable.
    :param float value: A double value.
    """
    def __init__(self, object function, double value):
        self._value = value
        self._function = autowrap_function2d(function)

    cdef double evaluate(self, double x, double y) except? -1e999:
        cdef double base = self._function.evaluate(x, y)
        if base < 0 and floor(self._value) != self._value:
            raise ValueError("Negative base and non-integral exponent is not supported")
        if base == 0 and self._value < 0:
            raise ZeroDivisionError("0.0 cannot be raised to a negative power")
        return base ** self._value


cdef class EqualsScalar2D(Function2D):
    """
    A Function2D class that tests the equality of a scalar and the result of a Function2D object: K == f2()

    This class is not intended to be used directly, but rather returned as the result of an __eq__() call on a
    Function2D object.

    :param value: A double value.
    :param object function: A Function2D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function2d(function)

    cdef double evaluate(self, double x, double y) except? -1e999:
        return self._value == self._function.evaluate(x, y)


cdef class NotEqualsScalar2D(Function2D):
    """
    A Function2D class that tests the inequality of a scalar and the result of a Function2D object: K != f2()

    This class is not intended to be used directly, but rather returned as the result of an __ne__() call on a
    Function2D object.

    :param value: A double value.
    :param object function: A Function2D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function2d(function)

    cdef double evaluate(self, double x, double y) except? -1e999:
        return self._value != self._function.evaluate(x, y)


cdef class LessThanScalar2D(Function2D):
    """
    A Function2D class that implements < of a scalar and the result of a Function2D object: K < f2()

    This class is not intended to be used directly, but rather returned as the result of an __lt__() call on a
    Function2D object.

    :param value: A double value.
    :param object function: A Function2D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function2d(function)

    cdef double evaluate(self, double x, double y) except? -1e999:
        return self._value < self._function.evaluate(x, y)


cdef class GreaterThanScalar2D(Function2D):
    """
    A Function2D class that implements > of a scalar and the result of a Function2D object: K > f2()

    This class is not intended to be used directly, but rather returned as the result of a __gt__() call on a
    Function2D object.

    :param value: A double value.
    :param object function: A Function2D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function2d(function)

    cdef double evaluate(self, double x, double y) except? -1e999:
        return self._value > self._function.evaluate(x, y)


cdef class LessEqualsScalar2D(Function2D):
    """
    A Function2D class that implements <= of a scalar and the result of a Function2D object: K <= f2()

    This class is not intended to be used directly, but rather returned as the result of an __le__() call on a
    Function2D object.

    :param value: A double value.
    :param object function: A Function2D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function2d(function)

    cdef double evaluate(self, double x, double y) except? -1e999:
        return self._value <= self._function.evaluate(x, y)


cdef class GreaterEqualsScalar2D(Function2D):
    """
    A Function2D class that implements >= of a scalar and the result of a Function2D object: K >= f2()

    This class is not intended to be used directly, but rather returned as the result of an __ge__() call on a
    Function2D object.

    :param value: A double value.
    :param object function: A Function2D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function2d(function)

    cdef double evaluate(self, double x, double y) except? -1e999:
        return self._value >= self._function.evaluate(x, y)
