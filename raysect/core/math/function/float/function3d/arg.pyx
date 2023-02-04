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

from raysect.core.math.function.float.function3d.base cimport Function3D


cdef class Arg3D(Function3D):
    """
    Returns one of the arguments the function is passed, unmodified

    This is used to pass coordinates through to other functions in the
    function framework which expect a Function3D object.

    Valid options for argument are "x", "y" or "z".

    >>> argx = Arg3D("x")
    >>> argx(2, 3, 5)
    2.0
    >>> argy = Arg3D("y")
    >>> argy(2, 3, 5)
    3.0
    >>> argz = Arg3D("z")
    >>> argz(2, 3, 5)
    5.0
    >>> squarerx = argx**2
    >>> squarerx(2, 3, 5)
    4.0
    >>> squarery = argy**2
    >>> squarery(2, 3, 5)
    9.0
    >>> squarerz = argz**2
    >>> squarerz(2, 3, 5)
    25.0

    :param str argument: either "x", "y" or "z", the argument to return
    """
    def __init__(self, object argument):
        if argument == "x":
            self._argument = X
        elif argument == "y":
            self._argument = Y
        elif argument == "z":
            self._argument = Z
        else:
            raise ValueError("The argument to Arg3D must be either 'x', 'y' or 'z'")

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        return x if self._argument == X else y if self._argument == Y else z
