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
from raysect.core.math.function.base cimport Function
from raysect.core.math.function.float.function3d.base cimport Function3D
from raysect.core.math.function.float.function3d.constant cimport Constant3D


cdef class PythonFunction3D(Function3D):
    """
    Wraps a python callable object with a Function3D object.

    This class allows a python object to interact with cython code that requires
    a Function3D object. The python object must implement __call__() expecting
    three arguments.

    This class is intended to be used to transparently wrap python objects that
    are passed via constructors or methods into cython optimised code. It is not
    intended that the users should need to directly interact with these wrapping
    objects. Constructors and methods expecting a Function3D object should be
    designed to accept a generic python object and then test that object to
    determine if it is an instance of Function3D. If the object is not a
    Function3D object it should be wrapped using this class for internal use.

    See also: autowrap_function3d()
    """

    def __init__(self, object function):
        self.function = function

    cdef double evaluate(self, double x, double y, double z) except? -1e999:
        return self.function(x, y, z)


cdef Function3D autowrap_function3d(object obj):
    """
    Automatically wraps the supplied python object in a PythonFunction3D or Contant3D object.

    If this function is passed a valid Function3D object, then the Function3D
    object is simply returned without wrapping.

    If this function is passed a numerical scalar (int or float), a Constant3D
    object is returned.

    This convenience function is provided to simplify the handling of Function3D
    and python callable objects in constructors, functions and setters.
    """

    if isinstance(obj, Function3D):
        return <Function3D> obj
    elif isinstance(obj, Function):
        raise TypeError('A Function3D object is required.')
    elif isinstance(obj, numbers.Real):
        return Constant3D(obj)
    else:
        return PythonFunction3D(obj)


def _autowrap_function3d(obj):
    """Expose cython function for testing."""
    return autowrap_function3d(obj)
