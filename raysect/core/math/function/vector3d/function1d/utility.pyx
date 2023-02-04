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

from raysect.core.math.vector cimport Vector3D, new_vector3d
from raysect.core.math.function.float cimport autowrap_function1d as autowrap_floatfunction1d
from .base cimport Function1D



cdef class FloatToVector3DFunction1D(Function1D):
    """
    Combines three float.Function1D objects to produce a vector3d.Function1D.

    The three float.Function1D objects correspond to the x, y and z components
    of the resulting vector object.

    :param float.Function1D x_function: the Vx(x) 1d function.
    :param float.Function1D y_function: the Vy(x) 1d function.
    :param float.Function1D z_function: the Vz(x) 1d function.

    .. code-block:: pycon

       >>> from raysect.core.math.function.float import Sqrt1D, Exp1D, Arg1D
       >>> from raysect.core.math.function.vector3d import FloatToVector3DFunction1D
       >>>
       >>> vx = 1  # Will be auto-wrapped to Constant1D(1)
       >>> vy = Arg1D('y')
       >>> vz = Sqrt1D(Arg1D('x'))
       >>>
       >>> fv = FloatToVector3DFunction1D(vx, vy, vz)
       >>> fv(4.0, 6.2)
       Vector3D(1.0, 6.2, 2.0)
    """

    def __init__(self, object x_function, object y_function, object z_function):
        self._x = autowrap_floatfunction1d(x_function)
        self._y = autowrap_floatfunction1d(y_function)
        self._z = autowrap_floatfunction1d(z_function)

    cdef Vector3D evaluate(self, double x):
        cdef double vx, vy, vz
        vx = self._x.evaluate(x)
        vy = self._y.evaluate(x)
        vz = self._z.evaluate(x)
        return new_vector3d(vx, vy, vz)
