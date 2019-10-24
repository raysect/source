# cython: language_level=3

# Copyright (c) 2014-2019, Dr Alex Meakins, Raysect Project
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
from .base cimport Function3D, PythonFunction3D
from .constant cimport Constant3D


cdef Function3D autowrap_function3d(object function):
    """
    Automatically wraps the supplied python object in a PythonFunction3D or Contant3D object.

    If this function is passed a valid Function3D object, then the Function3D
    object is simply returned without wrapping.

    If this function is passed a numerical scalar (int or float), a Constant3D
    object is returned.

    This convenience function is provided to simplify the handling of Function3D
    and python callable objects in constructors, functions and setters.
    """

    if isinstance(function, Function3D):
        return <Function3D> function
    elif isinstance(function, numbers.Real):
        return Constant3D(function)
    else:
        return PythonFunction3D(function)


def _autowrap_function3d(function):
    """Expose cython function for testing."""
    return autowrap_function3d(function)
