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

cimport libc.math as cmath
from raysect.core.math.function.float.function1d.base cimport Function1D
from raysect.core.math.function.float.function1d.autowrap cimport autowrap_function1d


cdef class Exp1D(Function1D):
    """
    A Function1D class that implements the exponential of the result of a Function1D object: exp(f())

    :param Function1D function: A Function1D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        return cmath.exp(self._function.evaluate(x))

    def __repr__(self):
        return 'e^({})'.format(self._function)


cdef class Sin1D(Function1D):
    """
    A Function1D class that implements the sine of the result of a Function1D object: sin(f())

    :param Function1D function: A Function1D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        return cmath.sin(self._function.evaluate(x))

    def __repr__(self):
        return 'sin({})'.format(self._function)


cdef class Cos1D(Function1D):
    """
    A Function1D class that implements the cosine of the result of a Function1D object: cos(f())

    :param Function1D function: A Function1D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        return cmath.cos(self._function.evaluate(x))

    def __repr__(self):
        return 'cos({})'.format(self._function)


cdef class Tan1D(Function1D):
    """
    A Function1D class that implements the tangent of the result of a Function1D object: tan(f())

    :param Function1D function: A Function1D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        return cmath.tan(self._function.evaluate(x))

    def __repr__(self):
        return 'tan({})'.format(self._function)


cdef class Asin1D(Function1D):
    """
    A Function1D class that implements the arcsine of the result of a Function1D object: asin(f())

    :param Function1D function: A Function1D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        cdef double v = self._function.evaluate(x)
        if -1.0 <= v <= 1.0:
            return cmath.asin(v)
        raise ValueError("The function returned a value outside of the arcsine domain of [-1, 1].")

    def __repr__(self):
        return 'asin({})'.format(self._function)


cdef class Acos1D(Function1D):
    """
    A Function1D class that implements the arccosine of the result of a Function1D object: acos(f())

    :param Function1D function: A Function1D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        cdef double v = self._function.evaluate(x)
        if -1.0 <= v <= 1.0:
            return cmath.acos(v)
        raise ValueError("The function returned a value outside of the arccosine domain of [-1, 1].")

    def __repr__(self):
        return 'acos({})'.format(self._function)


cdef class Atan1D(Function1D):
    """
    A Function1D class that implements the arctangent of the result of a Function1D object: atan(f())

    :param Function1D function: A Function1D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        return cmath.atan(self._function.evaluate(x))

    def __repr__(self):
        return 'atan({})'.format(self._function)


cdef class Atan4Q1D(Function1D):
    """
    A Function1D class that implements the arctangent of the result of 2 Function1D objects: atan2(f1(), f2())

    This differs from Atan1D in that it takes separate functions for the
    numerator and denominator, in order to get the quadrant correct.

    :param Function1D numerator: A Function1D object representing the numerator
    :param Function1D denominator: A Function1D object representing the denominator
    """
    def __init__(self, object numerator, object denominator):
        self._numerator = autowrap_function1d(numerator)
        self._denominator = autowrap_function1d(denominator)

    cdef double evaluate(self, double x) except? -1e999:
        return cmath.atan2(self._numerator.evaluate(x), self._denominator.evaluate(x))

    def __repr__(self):
        return 'atan(({}) / ({}))'.format(self._numerator, self._denominator)


cdef class Sqrt1D(Function1D):
    """
    A Function1D class that implements the square root of the result of a Function1D object: sqrt(f())

    :param Function1D function: A Function1D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        cdef double f = self._function.evaluate(x)
        if f < 0: # complex values are not supported
            raise ValueError("Math domain error in sqrt({0}). Sqrt of a negative value is not supported.".format(f))
        return cmath.sqrt(f)

    def __repr__(self):
        return 'sqrt({})'.format(self._function)


cdef class Erf1D(Function1D):
    """
    A Function1D class that implements the error function of the result of a Function1D object: erf(f())

    :param Function1D function: A Function1D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        return cmath.erf(self._function.evaluate(x))

    def __repr__(self):
        return 'erf({})'.format(self._function)
