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
from raysect.core.math.function.float.function3d.base cimport Function3D
from raysect.core.math.function.float.function3d.autowrap cimport autowrap_function3d


cdef class Exp3D(Function3D):
    """
    A Function3D class that implements the exponential of the result of a Function3D object: exp(f())

    :param Function3D function: A Function3D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function3d(function)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        return cmath.exp(self._function.evaluate(x, y, z))


cdef class Sin3D(Function3D):
    """
    A Function3D class that implements the sine of the result of a Function3D object: sin(f())

    :param Function3D function: A Function3D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function3d(function)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        return cmath.sin(self._function.evaluate(x, y, z))


cdef class Cos3D(Function3D):
    """
    A Function3D class that implements the cosine of the result of a Function3D object: cos(f())

    :param Function3D function: A Function3D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function3d(function)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        return cmath.cos(self._function.evaluate(x, y, z))


cdef class Tan3D(Function3D):
    """
    A Function3D class that implements the tangent of the result of a Function3D object: tan(f())

    :param Function3D function: A Function3D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function3d(function)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        return cmath.tan(self._function.evaluate(x, y, z))


cdef class Asin3D(Function3D):
    """
    A Function3D class that implements the arcsine of the result of a Function3D object: asin(f())

    :param Function3D function: A Function3D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function3d(function)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        cdef double v = self._function.evaluate(x, y, z)
        if -1.0 <= v <= 1.0:
            return cmath.asin(v)
        raise ValueError("The function returned a value outside of the arcsine domain of [-1, 1].")


cdef class Acos3D(Function3D):
    """
    A Function3D class that implements the arccosine of the result of a Function3D object: acos(f())

    :param Function3D function: A Function3D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function3d(function)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        cdef double v = self._function.evaluate(x, y, z)
        if -1.0 <= v <= 1.0:
            return cmath.acos(v)
        raise ValueError("The function returned a value outside of the arccosine domain of [-1, 1].")


cdef class Atan3D(Function3D):
    """
    A Function3D class that implements the arctangent of the result of a Function3D object: atan(f())

    :param Function3D function: A Function3D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function3d(function)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        return cmath.atan(self._function.evaluate(x, y, z))


cdef class Atan4Q3D(Function3D):
    """
    A Function3D class that implements the arctangent of the result of 2 Function3D objects: atan2(f1(), f2())

    This differs from Atan3D in that it takes separate functions for the
    numerator and denominator, in order to get the quadrant correct.

    :param Function3D numerator: A Function3D object representing the numerator
    :param Function3D denominator: A Function3D object representing the denominator
    """
    def __init__(self, object numerator, object denominator):
        self._numerator = autowrap_function3d(numerator)
        self._denominator = autowrap_function3d(denominator)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        return cmath.atan2(self._numerator.evaluate(x, y, z), self._denominator.evaluate(x, y, z))


cdef class Sqrt3D(Function3D):
    """
    A Function3D class that implements the square root of the result of a Function3D object: sqrt(f())

    :param Function3D function: A Function3D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function3d(function)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        cdef double f = self._function.evaluate(x, y, z)
        if f < 0: # complex values are not supported
            raise ValueError("Math domain error in sqrt({0}). Sqrt of a negative value is not supported.".format(f))
        return cmath.sqrt(f)


cdef class Erf3D(Function3D):
    """
    A Function3D class that implements the error function of the result of a Function3D object: erf(f())

    :param Function3D function: A Function3D object.
    """
    def __init__(self, object function):
        self._function = autowrap_function3d(function)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        return cmath.erf(self._function.evaluate(x, y, z))