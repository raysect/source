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
from raysect.core.math.function.vector3d.function2d.base cimport Function2D
from raysect.core.math.function.vector3d.function2d.constant cimport Constant2D


cdef class PythonFunction2D(Function2D):
    """
    Wraps a python callable object with a Function2D object.

    This class allows a python object to interact with cython code that requires
    a Vector3DFunction2D object. The python object must implement __call__() expecting
    two arguments.

    This class is intended to be used to transparently wrap python objects that
    are passed via constructors or methods into cython optimised code. It is not
    intended that the users should need to directly interact with these wrapping
    objects. Constructors and methods expecting a Vector3DFunction2D object should be
    designed to accept a generic python object and then test that object to
    determine if it is an instance of Vector3DFunction2D. If the object is not a
    Vector3DFunction2D object it should be wrapped using this class for internal use.

    See also: autowrap_vectorfunction2d()

    :param object function: the python function to wrap, __call__() function must
    be implemented on the object.
    """
    def __init__(self, object function):
        self.function = function

    cdef Vector3D evaluate(self, double x, double y):
        return self.function(x, y)


cdef Function2D autowrap_function2d(object obj):
    """
    Automatically wraps the supplied python object in a PythonVector3DFunction2D or ConstantVector2D object.

    If this function is passed a valid Vector3DFunction2D object, then the
    Vector3DFunction2D object is simply returned without wrapping.

    If this function is passed a Vector3D, a ConstantVector2D object is
    returned.

    This convenience function is provided to simplify the handling of
    Vector3DFunction2D and python callable objects in constructors, functions and
    setters.  """

    if isinstance(obj, Function2D):
        return <Function2D> obj
    elif isinstance(obj, Function):
        raise TypeError('A vector3d.Function2D object is required.')
    try:
        obj = Vector3D(*obj)
    except (TypeError, ValueError):  # Not an iterable which can be converted to Vector3D
        return PythonFunction2D(obj)
    else:
        return Constant2D(obj)


def _autowrap_function2d(obj):
    """Expose cython function for testing."""
    return autowrap_function2d(obj)
