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
from .base cimport Function1D, PythonFunction1D
from .constant cimport Constant1D


cdef Function1D autowrap_function1d(object function):
    """
    Automatically wraps the supplied python object in a PythonFunction1D or Contant1D object.

    If this function is passed a valid Function1D object, then the Function1D
    object is simply returned without wrapping.

    If this function is passed a numerical scalar (int or float), a Constant1D
    object is returned.

    This convenience function is provided to simplify the handling of Function1D
    and python callable objects in constructors, functions and setters.
    """

    if isinstance(function, Function1D):
        return <Function1D> function
    elif isinstance(function, numbers.Real):
        return Constant1D(function)
    else:
        return PythonFunction1D(function)


def _autowrap_function1d(function):
    """Expose cython function for testing."""
    return autowrap_function1d(function)
