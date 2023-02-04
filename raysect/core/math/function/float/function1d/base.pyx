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
from .autowrap cimport autowrap_function1d


cdef class Function1D(FloatFunction):
    """
    Cython optimised class for representing an arbitrary 1D function returning a float.

    Using __call__() in cython is slow. This class provides an overloadable
    cython cdef evaluate() method which has much less overhead than a python
    function call.

    For use in cython code only, this class cannot be extended via python.

    To create a new function object, inherit this class and implement the
    evaluate() method. The new function object can then be used with any code
    that accepts a function object.
    """

    cdef double evaluate(self, double x) except? -1e999:
        raise NotImplementedError("The evaluate() method has not been implemented.")

    def __call__(self, double x):
        """ Evaluate the function f(x)

        :param float x: function parameter x
        :rtype: float
        """
        return self.evaluate(x)

    def __repr__(self):
        return 'Function1D(x)'

    def __add__(object a, object b):
        if is_callable(a):
            if is_callable(b):
                # a() + b()
                return AddFunction1D(a, b)
            elif isinstance(b, numbers.Real):
                # a() + B -> B + a()
                return AddScalar1D(<double> b, a)
        elif isinstance(a, numbers.Real):
            if is_callable(b):
                # A + b()
                return AddScalar1D(<double> a, b)
        return NotImplemented

    def __sub__(object a, object b):
        if is_callable(a):
            if is_callable(b):
                # a() - b()
                return SubtractFunction1D(a, b)
            elif isinstance(b, numbers.Real):
                # a() - B -> -B + a()
                return AddScalar1D(-(<double> b), a)
        elif isinstance(a, numbers.Real):
            if is_callable(b):
                # A - b()
                return SubtractScalar1D(<double> a, b)
        return NotImplemented

    def __mul__(object a, object b):
        if is_callable(a):
            if is_callable(b):
                # a() * b()
                return MultiplyFunction1D(a, b)
            elif isinstance(b, numbers.Real):
                # a() * B -> B * a()
                return MultiplyScalar1D(<double> b, a)
        elif isinstance(a, numbers.Real):
            if is_callable(b):
                # A * b()
                return MultiplyScalar1D(<double> a, b)
        return NotImplemented

    @cython.cdivision(True)
    def __truediv__(object a, object b):
        cdef double v
        if is_callable(a):
            if is_callable(b):
                # a() / b()
                return DivideFunction1D(a, b)
            elif isinstance(b, numbers.Real):
                # a() / B -> 1/B * a()
                v = <double> b
                if v == 0.0:
                    raise ZeroDivisionError("Scalar used as the denominator of the division is zero valued.")
                return MultiplyScalar1D(1/v, a)
        elif isinstance(a, numbers.Real):
            if is_callable(b):
                # A * b()
                return DivideScalar1D(<double> a, b)
        return NotImplemented

    def __mod__(object a, object b):
        cdef double v
        if is_callable(a):
            if is_callable(b):
                # a() % b()
                return ModuloFunction1D(a, b)
            elif isinstance(b, numbers.Real):
                # a() % B
                v = <double> b
                if v == 0.0:
                    raise ZeroDivisionError("Scalar used as the divisor of the division is zero valued.")
                return ModuloFunctionScalar1D(a, v)
        elif isinstance(a, numbers.Real):
            if is_callable(b):
                # A % b()
                return ModuloScalarFunction1D(<double> a, b)
        return NotImplemented

    def __neg__(self):
        return MultiplyScalar1D(-1, self)

    def __pow__(object a, object b, object c):
        if c is not None:
            # Optimised implementation of pow(a, b, c) not available: fall back
            # to general implementation
            return (a ** b) % c
        if is_callable(a):
            if is_callable(b):
                # a() ** b()
                return PowFunction1D(a, b)
            elif isinstance(b, numbers.Real):
                # a() ** b
                return PowFunctionScalar1D(a, <double> b)
        elif isinstance(a, numbers.Real):
            if is_callable(b):
                # a ** b()
                return PowScalarFunction1D(<double> a, b)
        return NotImplemented

    def __abs__(self):
        return AbsFunction1D(self)

    def __richcmp__(self, object other, int op):
        if is_callable(other):
            if op == Py_EQ:
                return EqualsFunction1D(self, other)
            if op == Py_NE:
                return NotEqualsFunction1D(self, other)
            if op == Py_LT:
                return LessThanFunction1D(self, other)
            if op == Py_GT:
                return GreaterThanFunction1D(self, other)
            if op == Py_LE:
                return LessEqualsFunction1D(self, other)
            if op == Py_GE:
                return GreaterEqualsFunction1D(self, other)
        if isinstance(other, numbers.Real):
            if op == Py_EQ:
                return EqualsScalar1D(<double> other, self)
            if op == Py_NE:
                return NotEqualsScalar1D(<double> other, self)
            if op == Py_LT:
                # f() < K -> K > f
                return GreaterThanScalar1D(<double> other, self)
            if op == Py_GT:
                # f() > K -> K < f
                return LessThanScalar1D(<double> other, self)
            if op == Py_LE:
                # f() <= K -> K >= f
                return GreaterEqualsScalar1D(<double> other, self)
            if op == Py_GE:
                # f() >= K -> K <= f
                return LessEqualsScalar1D(<double> other, self)
        return NotImplemented


cdef class AddFunction1D(Function1D):
    """
    A Function1D class that implements the addition of the results of two Function1D objects: f1() + f2()

    This class is not intended to be used directly, but rather returned as the result of an __add__() call on a
    Function1D object.

    :param object function1: A Function1D object or Python callable.
    :param object function2: A Function1D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function1d(function1)
        self._function2 = autowrap_function1d(function2)

    cdef double evaluate(self, double x) except? -1e999:
        return self._function1.evaluate(x) + self._function2.evaluate(x)

    def __repr__(self):
        return '({}) + ({})'.format(self._function1, self._function2)


cdef class SubtractFunction1D(Function1D):
    """
    A Function1D class that implements the subtraction of the results of two Function1D objects: f1() - f2()

    This class is not intended to be used directly, but rather returned as the result of a __sub__() call on a
    Function1D object.

    :param object function1: A Function1D object or Python callable.
    :param object function2: A Function1D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function1d(function1)
        self._function2 = autowrap_function1d(function2)

    cdef double evaluate(self, double x) except? -1e999:
        return self._function1.evaluate(x) - self._function2.evaluate(x)

    def __repr__(self):
        return '({}) - ({})'.format(self._function1, self._function2)


cdef class MultiplyFunction1D(Function1D):
    """
    A Function1D class that implements the multiplication of the results of two Function1D objects: f1() * f2()

    This class is not intended to be used directly, but rather returned as the result of a __mul__() call on a
    Function1D object.

    :param object function1: A Function1D object or Python callable.
    :param object function2: A Function1D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function1d(function1)
        self._function2 = autowrap_function1d(function2)

    cdef double evaluate(self, double x) except? -1e999:
        return self._function1.evaluate(x) * self._function2.evaluate(x)

    def __repr__(self):
        return '({}) * ({})'.format(self._function1, self._function2)


cdef class DivideFunction1D(Function1D):
    """
    A Function1D class that implements the division of the results of two Function1D objects: f1() / f2()

    This class is not intended to be used directly, but rather returned as the result of a __truediv__() call on a
    Function1D object.

    :param object function1: A Function1D object or Python callable.
    :param object function2: A Function1D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function1d(function1)
        self._function2 = autowrap_function1d(function2)

    @cython.cdivision(True)
    cdef double evaluate(self, double x) except? -1e999:
        cdef double denominator = self._function2.evaluate(x)
        if denominator == 0.0:
            raise ZeroDivisionError("Function used as the denominator of the division returned a zero value.")
        return self._function1.evaluate(x) / denominator

    def __repr__(self):
        return '({}) / ({})'.format(self._function1, self._function2)


cdef class ModuloFunction1D(Function1D):
    """
    A Function1D class that implements the modulo of the results of two Function1D objects: f1() % f2()

    This class is not intended to be used directly, but rather returned as the result of a __mod__() call on a
    Function1D object.

    :param object function1: A Function1D object or Python callable.
    :param object function2: A Function1D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function1d(function1)
        self._function2 = autowrap_function1d(function2)

    @cython.cdivision(True)
    cdef double evaluate(self, double x) except? -1e999:
        cdef double divisor = self._function2.evaluate(x)
        if divisor == 0.0:
            raise ZeroDivisionError("Function used as the divisor of the modulo returned a zero value.")
        return self._function1.evaluate(x) % divisor

    def __repr__(self):
        return '({}) % ({})'.format(self._function1, self._function2)


cdef class PowFunction1D(Function1D):
    """
    A Function1D class that implements the pow() operator on two Function1D objects.

    This class is not intended to be used directly, but rather returned as the result of a __pow__() call on a
    Function1D object.

    :param object function1: A Function1D object or Python callable.
    :param object function2: A Function1D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function1d(function1)
        self._function2 = autowrap_function1d(function2)

    cdef double evaluate(self, double x) except? -1e999:
        cdef double base, exponent
        base = self._function1.evaluate(x)
        exponent = self._function2.evaluate(x)
        if base < 0 and floor(exponent) != exponent:  # Would return a complex value rather than double
            raise ValueError("Negative base and non-integral exponent is not supported")
        if base == 0 and exponent < 0:
            raise ZeroDivisionError("0.0 cannot be raised to a negative power")
        return base ** exponent

    def __repr__(self):
        return '({})**({})'.format(self._function1, self._function2)


cdef class AbsFunction1D(Function1D):
    """
    A Function1D class that implements the absolute value of the result of a Function1D object: abs(f()).

    This class is not intended to be used directly, but rather returned as the
    result of an __abs__() call on a Function1D object.

    :param object function: A Function1D object or Python callable.
    """
    def __init__(self, object function):
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        return abs(self._function.evaluate(x))

    def __repr__(self):
        return 'abs({})'.format(self._function)


cdef class EqualsFunction1D(Function1D):
    """
    A Function1D class that tests the equality of the results of two Function1D objects: f1() == f2()

    This class is not intended to be used directly, but rather returned as the result of an __eq__() call on a
    Function1D object.

    :param object function1: A Function1D object or Python callable.
    :param object function2: A Function1D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function1d(function1)
        self._function2 = autowrap_function1d(function2)

    cdef double evaluate(self, double x) except? -1e999:
        return self._function1.evaluate(x) == self._function2.evaluate(x)

    def __repr__(self):
        return '({}) == ({})'.format(self._function1, self._function2)


cdef class NotEqualsFunction1D(Function1D):
    """
    A Function1D class that tests the inequality of the results of two Function1D objects: f1() != f2()

    This class is not intended to be used directly, but rather returned as the result of an __ne__() call on a
    Function1D object.

    :param object function1: A Function1D object or Python callable.
    :param object function2: A Function1D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function1d(function1)
        self._function2 = autowrap_function1d(function2)

    cdef double evaluate(self, double x) except? -1e999:
        return self._function1.evaluate(x) != self._function2.evaluate(x)

    def __repr__(self):
        return '({}) != ({})'.format(self._function1, self._function2)


cdef class LessThanFunction1D(Function1D):
    """
    A Function1D class that implements < of the results of two Function1D objects: f1() < f2()

    This class is not intended to be used directly, but rather returned as the result of an __lt__() call on a
    Function1D object.

    :param object function1: A Function1D object or Python callable.
    :param object function2: A Function1D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function1d(function1)
        self._function2 = autowrap_function1d(function2)

    cdef double evaluate(self, double x) except? -1e999:
        return self._function1.evaluate(x) < self._function2.evaluate(x)

    def __repr__(self):
        return '({}) < ({})'.format(self._function1, self._function2)


cdef class GreaterThanFunction1D(Function1D):
    """
    A Function1D class that implements > of the results of two Function1D objects: f1() > f2()

    This class is not intended to be used directly, but rather returned as the result of a __gt__() call on a
    Function1D object.

    :param object function1: A Function1D object or Python callable.
    :param object function2: A Function1D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function1d(function1)
        self._function2 = autowrap_function1d(function2)

    cdef double evaluate(self, double x) except? -1e999:
        return self._function1.evaluate(x) > self._function2.evaluate(x)

    def __repr__(self):
        return '({}) > ({})'.format(self._function1, self._function2)


cdef class LessEqualsFunction1D(Function1D):
    """
    A Function1D class that implements <= of the results of two Function1D objects: f1() <= f2()

    This class is not intended to be used directly, but rather returned as the result of an __le__() call on a
    Function1D object.

    :param object function1: A Function1D object or Python callable.
    :param object function2: A Function1D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function1d(function1)
        self._function2 = autowrap_function1d(function2)

    cdef double evaluate(self, double x) except? -1e999:
        return self._function1.evaluate(x) <= self._function2.evaluate(x)

    def __repr__(self):
        return '({}) <= ({})'.format(self._function1, self._function2)


cdef class GreaterEqualsFunction1D(Function1D):
    """
    A Function1D class that implements >= of the results of two Function1D objects: f1() >= f2()

    This class is not intended to be used directly, but rather returned as the result of an __ge__() call on a
    Function1D object.

    :param object function1: A Function1D object or Python callable.
    :param object function2: A Function1D object or Python callable.
    """
    def __init__(self, object function1, object function2):
        self._function1 = autowrap_function1d(function1)
        self._function2 = autowrap_function1d(function2)

    cdef double evaluate(self, double x) except? -1e999:
        return self._function1.evaluate(x) >= self._function2.evaluate(x)

    def __repr__(self):
        return '({}) >= ({})'.format(self._function1, self._function2)


cdef class AddScalar1D(Function1D):
    """
    A Function1D class that implements the addition of scalar and the result of a Function1D object: K + f()

    This class is not intended to be used directly, but rather returned as the result of an __add__() call on a
    Function1D object.

    :param float value: A double value.
    :param object function: A Function1D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        return self._value + self._function.evaluate(x)

    def __repr__(self):
        return '{} + ({})'.format(self._value, self._function)


cdef class SubtractScalar1D(Function1D):
    """
    A Function1D class that implements the subtraction of scalar and the result of a Function1D object: K - f()

    This class is not intended to be used directly, but rather returned as the result of an __sub__() call on a
    Function1D object.

    :param double value: A double value.
    :param object function: A Function1D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        return self._value - self._function.evaluate(x)

    def __repr__(self):
        return '{} - ({})'.format(self._value, self._function)


cdef class MultiplyScalar1D(Function1D):
    """
    A Function1D class that implements the multiplication of scalar and the result of a Function1D object: K * f()

    This class is not intended to be used directly, but rather returned as the result of an __mul__() call on a
    Function1D object.

    :param float value: A double value.
    :param object function: A Function1D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        return self._value * self._function.evaluate(x)

    def __repr__(self):
        return '{} * ({})'.format(self._value, self._function)


cdef class DivideScalar1D(Function1D):
    """
    A Function1D class that implements the division of scalar and the result of a Function1D object: K / f()

    This class is not intended to be used directly, but rather returned as the result of an __div__() call on a
    Function1D object.

    :param float value: A double value.
    :param object function: A Function1D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function1d(function)

    @cython.cdivision(True)
    cdef double evaluate(self, double x) except? -1e999:
        cdef double denominator = self._function.evaluate(x)
        if denominator == 0.0:
            raise ZeroDivisionError("Function used as the denominator of the division returned a zero value.")
        return self._value / denominator

    def __repr__(self):
        return '{} / ({})'.format(self._value, self._function)


cdef class ModuloScalarFunction1D(Function1D):
    """
    A Function1D class that implements the modulo of scalar and the result of a Function1D object: K % f()

    This class is not intended to be used directly, but rather returned as the result of a __mod__() call on a
    Function1D object.

    :param float value: A double value.
    :param object function: A Function1D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function1d(function)

    @cython.cdivision(True)
    cdef double evaluate(self, double x) except? -1e999:
        cdef double divisor = self._function.evaluate(x)
        if divisor == 0.0:
            raise ZeroDivisionError("Function used as the divisor of the modulo returned a zero value.")
        return self._value % divisor

    def __repr__(self):
        return '{} % ({})'.format(self._value, self._function)


cdef class ModuloFunctionScalar1D(Function1D):
    """
    A Function1D class that implements the modulo of the result of a Function1D object and a scalar: f() % K

    This class is not intended to be used directly, but rather returned as the result of a __mod__() call on a
    Function1D object.

    :param object function: A Function1D object or Python callable.
    :param float value: A double value.
    """
    def __init__(self, object function, double value):
        if value == 0:
            raise ValueError("Divisor cannot be zero")
        self._value = value
        self._function = autowrap_function1d(function)

    @cython.cdivision(True)
    cdef double evaluate(self, double x) except? -1e999:
        return self._function.evaluate(x) % self._value

    def __repr__(self):
        return '({}) % {}'.format(self._function, self._value)


cdef class PowScalarFunction1D(Function1D):
    """
    A Function1D class that implements the pow of scalar and the result of a Function1D object: K ** f()

    This class is not intended to be used directly, but rather returned as the result of an __pow__() call on a
    Function1D object.

    :param float value: A double value.
    :param object function: A Function1D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        cdef double exponent = self._function.evaluate(x)
        if self._value < 0 and floor(exponent) != exponent:
            raise ValueError("Negative base and non-integral exponent is not supported")
        if self._value == 0 and exponent < 0:
            raise ZeroDivisionError("0.0 cannot be raised to a negative power")
        return self._value ** exponent

    def __repr__(self):
        return '{}**({})'.format(self._value, self._function)


cdef class PowFunctionScalar1D(Function1D):
    """
    A Function1D class that implements the pow of the result of a Function1D object and a scalar: f() ** K

    This class is not intended to be used directly, but rather returned as the result of an __pow__() call on a
    Function1D object.

    :param object function: A Function1D object or Python callable.
    :param float value: A double value.
    """
    def __init__(self, object function, double value):
        self._value = value
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        cdef double base = self._function.evaluate(x)
        if base < 0 and floor(self._value) != self._value:
            raise ValueError("Negative base and non-integral exponent is not supported")
        if base == 0 and self._value < 0:
            raise ZeroDivisionError("0.0 cannot be raised to a negative power")
        return base ** self._value

    def __repr__(self):
        return '({})**{}'.format(self._function, self._value)


cdef class EqualsScalar1D(Function1D):
    """
    A Function1D class that tests the equality of a scalar and the result of a Function1D object: K == f2()

    This class is not intended to be used directly, but rather returned as the result of an __eq__() call on a
    Function1D object.

    :param value: A double value.
    :param object function: A Function1D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        return self._value == self._function.evaluate(x)

    def __repr__(self):
        return '{} == ({})'.format(self._value, self._function)


cdef class NotEqualsScalar1D(Function1D):
    """
    A Function1D class that tests the inequality of a scalar and the result of a Function1D object: K != f2()

    This class is not intended to be used directly, but rather returned as the result of an __ne__() call on a
    Function1D object.

    :param value: A double value.
    :param object function: A Function1D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        return self._value != self._function.evaluate(x)

    def __repr__(self):
        return '{} != ({})'.format(self._value, self._function)


cdef class LessThanScalar1D(Function1D):
    """
    A Function1D class that implements < of a scalar and the result of a Function1D object: K < f2()

    This class is not intended to be used directly, but rather returned as the result of an __lt__() call on a
    Function1D object.

    :param value: A double value.
    :param object function: A Function1D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        return self._value < self._function.evaluate(x)

    def __repr__(self):
        return '{} < ({})'.format(self._value, self._function)


cdef class GreaterThanScalar1D(Function1D):
    """
    A Function1D class that implements > of a scalar and the result of a Function1D object: K > f2()

    This class is not intended to be used directly, but rather returned as the result of a __gt__() call on a
    Function1D object.

    :param value: A double value.
    :param object function: A Function1D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        return self._value > self._function.evaluate(x)

    def __repr__(self):
        return '{} > ({})'.format(self._value, self._function)


cdef class LessEqualsScalar1D(Function1D):
    """
    A Function1D class that implements <= of a scalar and the result of a Function1D object: K <= f2()

    This class is not intended to be used directly, but rather returned as the result of an __le__() call on a
    Function1D object.

    :param value: A double value.
    :param object function: A Function1D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        return self._value <= self._function.evaluate(x)

    def __repr__(self):
        return '{} <= ({})'.format(self._value, self._function)


cdef class GreaterEqualsScalar1D(Function1D):
    """
    A Function1D class that implements >= of a scalar and the result of a Function1D object: K >= f2()

    This class is not intended to be used directly, but rather returned as the result of an __ge__() call on a
    Function1D object.

    :param value: A double value.
    :param object function: A Function1D object or Python callable.
    """
    def __init__(self, double value, object function):
        self._value = value
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        return self._value >= self._function.evaluate(x)

    def __repr__(self):
        return '{} >= ({})'.format(self._value, self._function)