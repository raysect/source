# cython: language_level=3

# Copyright (c) 2014, Dr Alex Meakins, Raysect Project
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

# 1D functions -----------------------------------------------------------------

cdef class Function1D:
    """
    Cython optimised class for representing an arbitrary 1D function.

    Using __call__() in cython is slow. This class provides an overloadable
    cython cdef evaluate() method which has much less overhead than a python
    function call.

    For use in cython code only, this class cannot be extended via python.

    To create a new function object, inherit this class and implement the
    evaluate() method. The new function object can then be used with any code
    that accepts a function object.
    """

    cdef double evaluate(self, double x) except *:

        raise NotImplementedError("The evaluate() method has not been implemented.")

    def __call__(self, double x):

        return self.evaluate(x)


cdef class PythonFunction1D(Function1D):
    """
    Wraps a python callable object with a Function1D object.

    This class allows a python object to interact with cython code that requires
    a Function1D object. The python object must implement __call__() expecting
    one argument.

    This class is intended to be used to transparently wrap python objects that
    are passed via constructors or methods into cython optimised code. It is not
    intended that the users should need to directly interact with these wrapping
    objects. Constructors and methods expecting a Function1D object should be
    designed to accept a generic python object and then test that object to
    determine if it is an instance of Function1D. If the object is not a
    Function1D object it should be wrapped using this class for internal use.

    See also: autowrap_function1d()
    """

    def __init__(self, object function):

        self.function = function

    cdef double evaluate(self, double x) except *:

        return self.function(x)


cdef inline Function1D autowrap_function1d(object function):
    """
    Automatically wraps the supplied python object in a PythonFunction1D object.

    If this function is passed a valid Function1D object, then the Function1D
    object is simply returned without wrapping.

    This convinience function is provided to simplify the handling of Function1D
    and python callable objects in constuctors, functions and setters.
    """

    if isinstance(function, Function1D):

        return <Function1D> function

    else:

        return PythonFunction1D(function)


# 2D functions -----------------------------------------------------------------

cdef class Function2D:
    """
    Cython optimised class for representing an arbitrary 2D function.

    Using __call__() in cython is slow. This class provides an overloadable
    cython cdef evaluate() method which has much less overhead than a python
    function call.

    For use in cython code only, this class cannot be extended via python.

    To create a new function object, inherit this class and implement the
    evaluate() method. The new function object can then be used with any code
    that accepts a function object.
    """

    cdef double evaluate(self, double x, double y) except *:

        raise NotImplementedError("The evaluate() method has not been implemented.")

    def __call__(self, double x, double y):

        return self.evaluate(x, y)


cdef class PythonFunction2D(Function2D):
    """
    Wraps a python callable object with a Function2D object.

    This class allows a python object to interact with cython code that requires
    a Function2D object. The python object must implement __call__() expecting
    two arguments.

    This class is intended to be used to transparently wrap python objects that
    are passed via constructors or methods into cython optimised code. It is not
    intended that the users should need to directly interact with these wrapping
    objects. Constructors and methods expecting a Function2D object should be
    designed to accept a generic python object and then test that object to
    determine if it is an instance of Function2D. If the object is not a
    Function2D object it should be wrapped using this class for internal use.

    See also: autowrap_function2d()
    """

    def __init__(self, object function):

        self.function = function

    cdef double evaluate(self, double x, double y) except *:

        return self.function(x, y)


cdef inline Function2D autowrap_function2d(object function):
    """
    Automatically wraps the supplied python object in a PythonFunction2D object.

    If this function is passed a valid Function2D object, then the Function2D
    object is simply returned without wrapping.

    This convinience function is provided to simplify the handling of Function2D
    and python callable objects in constuctors, functions and setters.
    """

    if isinstance(function, Function2D):

        return <Function2D> function

    else:

        return PythonFunction2D(function)


# 3D functions -----------------------------------------------------------------

cdef class Function3D:
    """
    Cython optimised class for representing an arbitrary 3D function.

    Using __call__() in cython is slow. This class provides an overloadable
    cython cdef evaluate() method which has much less overhead than a python
    function call.

    For use in cython code only, this class cannot be extended via python.

    To create a new function object, inherit this class and implement the
    evaluate() method. The new function object can then be used with any code
    that accepts a function object.
    """

    cdef double evaluate(self, double x, double y, double z) except *:

        raise NotImplementedError("The evaluate() method has not been implemented.")

    def __call__(self, double x, double y, double z):

        return self.evaluate(x, y, z)


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

    cdef double evaluate(self, double x, double y, double z) except *:

        return self.function(x, y, z)


cdef inline Function3D autowrap_function3d(object function):
    """
    Automatically wraps the supplied python object in a PythonFunction3D object.

    If this function is passed a valid Function3D object, then the Function3D
    object is simply returned without wrapping.

    This convinience function is provided to simplify the handling of Function3D
    and python callable objects in constuctors, functions and setters.
    """

    if isinstance(function, Function3D):

        return <Function3D> function

    else:

        return PythonFunction3D(function)

