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
from raysect.core.math.function.vector3d.function3d.base cimport Function3D


cdef class Constant3D(Function3D):
    """
    Wraps a Vector3D object with a Function3D object.

    This class allows a constant vector object to interact with cython code that
    requires a vector3d.Function3D object. The object must be convertible to a
    Vector3D. The value of the Vector3D constant will be returned independent of
    the arguments the function is called with.

    This class is intended to be used to transparently wrap python objects that
    are passed via constructors or methods into cython optimised code. It is not
    intended that the users should need to directly interact with these wrapping
    objects. Constructors and methods expecting a vector3d.function3D object should be
    designed to accept a generic python object and then test that object to
    determine if it is an instance of vector3d.Function3D. If the object is not a
    vector3d.Function3D object it should be wrapped using this class for internal use.

    See also: float3d.autowrap_function3d()

    :param object value: the constant value, convertible to Vector3D, to return when called.
    """
    def __init__(self, object value):
        self._value = Vector3D(*value)

    cdef Vector3D evaluate(self, double x, double y, double z):
        return self._value
