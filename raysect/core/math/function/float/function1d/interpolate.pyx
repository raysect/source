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

"""
Interpolation functions for float.Function1D

Interpolators are accessed through interface class Interpolator1DArray.
"""

import numpy as np
cimport cython
from raysect.core.math.cython.interpolation.linear cimport linear1d
from raysect.core.math.cython.interpolation.cubic cimport calc_coefficients_1d, evaluate_cubic_1d
from raysect.core.math.cython.utility cimport find_index, lerp


cdef class Interpolator1DArray(Function1D):
    """
    A configurable interpolator for 1D arrays.

    Coordinate array (x) and data array (f) are sorted and transformed into Numpy arrays.
    The resulting Numpy arrays are stored as read only. I.e. `writeable` flag of self.x and self.f
    is set to False. Alteration of the flag may result in unwanted behaviour.

    :param object x: 1D array-like object of real values storing the x spline knot positions.
    :param object f: 1D array-like object of real values storing the spline knot function value at x.
    :param str interpolation_type: Type of interpolation to use. Options are:
        `linear`: Interpolates the data using piecewise linear interpolation.
        `cubic`: Interpolates the data using piecewise cubic interpolation.
    :param str extrapolation_type: Type of extrapolation to use. Options are:
        `none`: Attempt to access data outside of x's range will yield ValueError.
        `nearest`: Extrapolation results is the nearest position x value in the interpolation domain.
        `linear`: Extrapolate bilinearly the interpolation function.
        `quadratic`: Extrapolate quadratically the interpolation function. Constrains the function at the edge, and the
        derivative both at the edge and 1 spline knot from the edge.
    :param double extrapolation_range: Limits the range where extrapolation is permitted. Requesting data beyond the
        extrapolation range results in ValueError. Extrapolation range will be applied as padding symmetrically to both
        ends of the interpolation range (x).

    .. code-block:: python

        >>> from raysect.core.math.function.float.function1d.interpolate import Interpolator1DArray
        >>>
        >>> x = np.linspace(-1., 1., 20)
        >>> f = np.exp(-x**2)
        >>> interpolator1D = Interpolate1DArray(x, f, 'cubic', 'nearest', 1.0)
        >>> # Interpolation
        >>> interpolator1D(0.2)
        0.9607850606581484
        >>> # Extrapolation
        >>> interpolator1D(1.1)
        0.36787944117144233
        >>> # Extrapolation out of bounds
        >>> interpolator1D(2.1)
        ValueError: The specified value (x=2.1) is outside of extrapolation range.

    :note: All input derivatives used in calculations use the previous and next indices in the spline knot arrays.
        At the edge of the spline knot arrays the index of the edge of the array is is used instead.
    :note: x and f arrays must be of equal length.
    :note: x must be a monotonically increasing array.

    """

    def __init__(self, object x, object f, str interpolation_type,
                 str extrapolation_type, double extrapolation_range):

        x = np.array(x, dtype=np.float64, order='c')
        f = np.array(f, dtype=np.float64, order='c')

        # extrapolation_range must be greater than or equal to 0.
        if extrapolation_range < 0:
            raise ValueError('extrapolation_range must be greater than or equal to 0.')

        # dimensions checks
        if x.ndim != 1:
            raise ValueError(f'The x array must be 1D. Got {x.shape}.')

        if f.ndim != 1:
            raise ValueError(f'The f array must be 1D. Got {f.shape}.')

        if x.shape[0] < 2:
            raise ValueError(f'There must be at least 2 spline knots to interpolate. The shape of the spline knot array is ({x.shape}).')

        if x.shape != f.shape:
            raise ValueError(f'Shape mismatch between x array ({x.shape}) and f array ({f.shape}).')

        # test monotonicity
        if (np.diff(x) <= 0).any():
            raise ValueError('The x array must be monotonically increasing.')

        self.x = x
        self.f = f

        # obtain memory views for fast data access
        self._x_mv = x
        self._f_mv = f

        # prevent users being able to change the data arrays
        self.x.flags.writeable = False
        self.f.flags.writeable = False

        self._last_index = self.x.shape[0] - 1
        self._extrapolation_range = extrapolation_range

        # Check the requested interpolation type exists.
        interpolation_type = interpolation_type.lower()
        if interpolation_type not in id_to_interpolator:
            raise ValueError(f'Interpolation type {interpolation_type} not found. Options are {id_to_interpolator.keys()}.')

        # Check the requested extrapolation type exists.
        extrapolation_type = extrapolation_type.lower()
        if extrapolation_type not in id_to_extrapolator:
            raise ValueError(f'Extrapolation type {interpolation_type} not found. Options are {id_to_extrapolator.keys()}.')

        # Permit combinations of interpolator and extrapolator where the order of extrapolator is higher than interpolator.
        if extrapolation_type not in permitted_interpolation_combinations[interpolation_type]:
            raise ValueError(f'Extrapolation type {extrapolation_type} not compatible with interpolation type {interpolation_type}.')

        # Create the interpolator and extrapolator objects.
        self._interpolator = id_to_interpolator[interpolation_type](self._x_mv, self._f_mv)
        self._extrapolator = id_to_extrapolator[extrapolation_type](self._x_mv, self._f_mv)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double px) except? -1e999:
        """
        Evaluates the interpolating function.

        :param double px: the point for which an interpolated value is required.
        :return: the interpolated value at point x.
        """

        cdef int index = find_index(self._x_mv, px)

        # find_index returns -1 in the lower extrapolation region, the index of the bin lower than px. The last index
        # is returned if greater than or equal to the largest bin edge, greater is handled by the extrapolator, equal is handled by the interpolator.
        if index == -1:
            if px < self._x_mv[0] - self._extrapolation_range:
                raise ValueError(f'The specified value (x={px}) is outside of extrapolation range.')
            return self._extrapolator.evaluate(px, index)

        elif index == self._last_index and px > self.x[self._last_index]:
            if px > self._x_mv[self._last_index] + self._extrapolation_range:
                raise ValueError(f'The specified value (x={px}) is outside of extrapolation range.')
            return self._extrapolator.evaluate(px, index)

        elif px == self.x[self._last_index]:
            return self._interpolator.evaluate(px, index - 1)

        else:
            return self._interpolator.evaluate(px, index)

    @property
    def domain(self):
        """
        Returns min/max interval of 'x' array.
        Order: min(x), max(x).
        """
        return self._x_mv[0], self._x_mv[self._last_index]


cdef class _Interpolator1D:
    """
    Base class for 1D interpolators.

    :param x: 1D memory view of the spline point x positions.
    :param f: 1D memory view of the function value at spline point x positions.
    """

    ID = None

    def __init__(self, double[::1] x, double[::1] f):
        self._x = x
        self._f = f
        self._last_index = self._x.shape[0] - 1

    cdef double evaluate(self, double px, int index) except? -1e999:
        """
        Calculates interpolated value at a requested point.
    
        :param double px: the point for which an interpolated value is required.
        :param int index: the lower index of the bin containing point px. (Result of bisection search).
        """
        raise NotImplementedError('_Interpolator is an abstract base class.')

    # cdef double _analytic_gradient(self, double px, int index, int order):
    #     """
    #     Calculates the interpolator's derivative of a valid order at a requested point.
    #
    #     :param double px: the point for which an interpolated value is required.
    #     :param int index: the lower index of the bin containing point px. (Result of bisection search).
    #     """
    #     raise NotImplementedError('_Interpolator is an abstract base class.')


cdef class _Interpolator1DLinear(_Interpolator1D):
    """
    Linear interpolation of 1D function.

    :param x: 1D memory view of the spline point x positions.
    :param f: 1D memory view of the function value at spline point x positions.
    """

    ID = 'linear'

    def __init__(self, double[::1] x, double[::1] f):
        super().__init__(x, f)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double px, int index) except? -1e999:
        return linear1d(self._x[index], self._x[index + 1], self._f[index], self._f[index + 1], px)

    # @cython.cdivision(True)
    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.initializedcheck(False)
    # cdef double _analytic_gradient(self, double px, int index, int order):
    #
    #     cdef double grad
    #
    #     if order == 1:
    #         grad = (self._f[index + 1] - self._f[index]) / (self._x[index + 1] - self._x[index])
    #     elif order > 1:
    #         grad = 0
    #     else:
    #         raise ValueError('The derivative order must be 1 for the linear interpolator, order = 0 should be an evaluation, greater values return.')
    #
    #     return grad


cdef class _Interpolator1DCubic(_Interpolator1D):
    """
    Cubic interpolation of 1D function

    When called, stores cubic polynomial coefficients from the value of the function at the neighboring spline points
    and the gradient at the neighbouring spline points based on central difference gradients. The polynomial
    coefficients and gradients are calculated between each spline knots normalised to between 0 and 1.

    :param x: 1D memory view of the spline point x positions.
    :param f: 1D memory view of the function value at spline point x positions.
    """

    ID = 'cubic'

    def __init__(self, double[::1] x, double[::1] f):

        super().__init__(x, f)

        # where the spline coefficients (a) have been calculated the value is set to 1, 0 otherwise
        self._calculated = np.zeros((self._last_index,), dtype=np.uint8)

        # store the cubic spline coefficients, where increasing index values are the coefficients for the coefficients of higher powers of x in the last dimension.
        self._a = np.zeros((self._last_index, 4), dtype=np.float64)
        self._array_derivative = _ArrayDerivative1D(self._x, self._f)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double px, int index) except? -1e999:

        # rescale x between 0 and 1
        cdef double nx
        cdef double[2] f, dfdx
        cdef double[4] a
        cdef int i

        # Calculate the coefficients (and gradients at each spline point) if they dont exist
        if not self._calculated[index]:

            f[0] = self._f[index]
            f[1] = self._f[index + 1]

            dfdx[0] = self._array_derivative.evaluate(index, rescale_norm=False)
            dfdx[1] = self._array_derivative.evaluate(index + 1, rescale_norm=True)

            calc_coefficients_1d(f, dfdx, a)

            for i in range(4):
                self._a[index, i] = a[i]
            self._calculated[index] = 1

        else:
            for i in range(4):
                a[i] = self._a[index, i]

        # obtain normalised x coordinate inside cell
        nx = (px - self._x[index]) / (self._x[index + 1] - self._x[index])

        return evaluate_cubic_1d(a, nx)

    # @cython.cdivision(True)
    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.initializedcheck(False)
    # cdef double _analytic_gradient(self, double px, int index, int order):
    #
    #     cdef double grad
    #     cdef double nx
    #     cdef double[2] f, dfdx
    #     cdef double rdx
    #     cdef double[4] a
    #
    #     rdx = 1.0 / (self._x[index + 1] - self._x[index])
    #     nx = (px - self._x[index]) * rdx
    #
    #     f[0] = self._f[index]
    #     f[1] = self._f[index + 1]
    #
    #     dfdx[0] = self._array_derivative.evaluate(index, rescale_norm=False)
    #     dfdx[1] = self._array_derivative.evaluate(index + 1, rescale_norm=True)
    #
    #     calc_coefficients_1d(f, dfdx, a)
    #
    #     if order == 1:
    #         grad = 3*a[0]*nx*nx + 2*a[1]*nx + a[2]
    #         grad *= rdx
    #
    #     elif order == 2:
    #         grad = 6*a[0]*nx + 2*a[1]
    #         grad *= rdx*rdx
    #
    #     elif order == 3:
    #         grad = 6*a[0]
    #         grad *= rdx*rdx*rdx
    #
    #     elif order > 3:
    #         grad = 0
    #
    #     else:
    #         raise ValueError('Order must be an integer greater than or equal to 1.')
    #
    #     return grad


cdef class _Extrapolator1D:
    """
    Base class for Function1D extrapolators.

    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    """

    ID = NotImplemented

    def __init__(self, double[::1] x, double[::1] f):
        self._x = x
        self._f = f
        self._last_index = self._x.shape[0] - 1

    cdef double evaluate(self, double px, int index) except? -1e999:
        raise NotImplementedError(f'{self.__class__} not implemented.')

    # cdef double _analytic_gradient(self, double px, int index, int order):
    #     raise NotImplementedError(f'{self.__class__} not implemented.')


cdef class _Extrapolator1DNone(_Extrapolator1D):
    """
    Extrapolator that does nothing.
    """

    ID = 'none'

    def __init__(self, double [::1] x, double[::1] f):
           super().__init__(x, f)

    cdef double evaluate(self, double px, int index)  except? -1e999:
        raise ValueError(f'Extrapolation not available. Interpolate within function range {self._x[0]}-{self._x[self._last_index]}.')

    # cdef double _analytic_gradient(self, double px, int index, int order):
    #     raise ValueError(f'Extrapolation not available. Interpolate within function range {self._x[0]}-{self._x[self._last_index]}.')


cdef class _Extrapolator1DNearest(_Extrapolator1D):
    """
    Extrapolator that returns nearest input value.

    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    """

    ID = 'nearest'

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double px, int index) except? -1e999:

        if index == -1:
            return self._f[0]
        elif index == self._last_index:
            return self._f[self._last_index]
        else:
            raise ValueError(f'Cannot evaluate value of function at point {px}. Bad data?')

    # @cython.initializedcheck(False)
    # cdef double _analytic_gradient(self, double px, int index, int order):
    #     return 0.0


cdef class _Extrapolator1DLinear(_Extrapolator1D):
    """
    Extrapolator that extrapolates linearly.

    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    """

    ID = 'linear'

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double px, int index) except? -1e999:

        # The index returned from find_index is -1 at the array start or the length of the array at the end of array.
        if index == -1:
            index = 0
        elif index == self._last_index:
            index = self._last_index - 1
        else:
            raise ValueError('Invalid extrapolator index. Must be -1 for lower and shape-1 for upper extrapolation.')

        # Use a linear interpolator function to extrapolate instead
        return lerp(self._x[index], self._x[index + 1], self._f[index], self._f[index + 1], px)

    # @cython.cdivision(True)
    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.initializedcheck(False)
    # cdef double _analytic_gradient(self, double px, int index, int order):
    #
    #     # The index returned from find_index is -1 at the array start or the length of the array at the end of array.
    #     if index == -1:
    #         index = 0
    #     elif index == self._last_index:
    #         index = self._last_index - 1
    #     else:
    #         raise ValueError('Invalid extrapolator index. Must be -1 for lower and shape-1 for upper extrapolation.')
    #
    #     if order == 1:
    #         return (self._f[index + 1] - self._f[index]) / (self._x[index + 1] - self._x[index])
    #     elif order > 1:
    #         return 0.0
    #     else:
    #         raise ValueError('order must be an integer greater than or equal to 1.')


cdef class _Extrapolator1DQuadratic(_Extrapolator1D):
    """
    Extrapolator that extrapolates quadratically.

    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    """

    ID = 'quadratic'

    def __init__(self, double [::1] x, double[::1] f):

        cdef double[2] dfdx_start, dfdx_end
        cdef _ArrayDerivative1D array_derivative

        super().__init__(x, f)
        self._last_index = self._x.shape[0] - 1
        array_derivative = _ArrayDerivative1D(self._x, self._f)
        dfdx_start[0] = array_derivative.evaluate(0, rescale_norm=False)

        # Need to have the first derivatives normalised to the distance between spline knot 0->1 (not 1->2),
        # So un-normalise then re-normalise.
        dfdx_start[1] = array_derivative.evaluate(1, rescale_norm=True)

        dfdx_end[0] = array_derivative.evaluate(self._last_index - 1, rescale_norm=False)
        dfdx_end[1] = array_derivative.evaluate(self._last_index, rescale_norm=True)

        self._calculate_quadratic_coefficients_start(f[0], dfdx_start[0], dfdx_start[1], self._a_first)
        self._calculate_quadratic_coefficients_end(f[self._last_index],  dfdx_end[0], dfdx_end[1], self._a_last)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef void _calculate_quadratic_coefficients_start(self, double f1, double df1_dx, double df2_dx, double[3] a):
        """
        Calculate the coefficients for a quadratic spline where 2 spline knots are normalised to between 0 and 1. 
        """

        a[0] = -0.5*df1_dx + 0.5*df2_dx
        a[1] = df1_dx
        a[2] = f1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef void _calculate_quadratic_coefficients_end(self, double f2, double df1_dx, double df2_dx, double[3] a):
        """
        Calculate the coefficients for a quadratic spline where 2 spline knots are normalised to between 0 and 1. 
        """

        a[0] = - 0.5*df1_dx + 0.5*df2_dx
        a[1] = df1_dx
        a[2] = f2 - 0.5*df1_dx - 0.5*df2_dx

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double px, int index) except? -1e999:

        cdef double nx

        # The index returned from find_index is -1 at the array start or the length of the array at the end of array.
        if index == -1:
            nx = (px - self._x[0]) / (self._x[1] - self._x[0])
            return self._a_first[0]*nx*nx + self._a_first[1]*nx + self._a_first[2]

        elif index == self._last_index:
            nx = (px - self._x[self._last_index - 1]) / (self._x[self._last_index] - self._x[self._last_index - 1])
            return self._a_last[0]*nx*nx + self._a_last[1]*nx + self._a_last[2]

        else:
            raise ValueError('Invalid extrapolator index. Must be -1 for lower and shape-1 for upper extrapolation.')

    # @cython.cdivision(True)
    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.initializedcheck(False)
    # cdef double _analytic_gradient(self, double px, int index, int order):
    #
    #     cdef double rdx, nx
    #     cdef double grad
    #
    #     # The index returned from find_index is -1 at the array start or the length of the array at the end of array.
    #     if index == -1:
    #
    #         rdx = 1.0 / (self._x[1] - self._x[0])
    #         nx =  (px - self._x[0]) * rdx
    #
    #         if order == 1:
    #             grad = 2.0*self._a_first[0]*nx + self._a_first[1]
    #             grad *= rdx
    #
    #         elif order == 2:
    #             grad = 2.0*self._a_first[0]
    #             grad *= rdx*rdx
    #
    #         elif order > 2:
    #             grad = 0
    #
    #         else:
    #             raise ValueError('order must be an integer greater than or equal to 1.')
    #
    #     elif index == self._last_index:
    #
    #         rdx = 1.0 / (self._x[self._last_index] - self._x[self._last_index - 1])
    #         nx = (px - self._x[self._last_index - 1]) * rdx
    #
    #         if order == 1:
    #             grad = 2.0*self._a_last[0]*nx + self._a_last[1]
    #             grad *= rdx
    #
    #         elif order == 2:
    #             grad = 2.0*self._a_last[0]
    #             grad *= rdx*rdx
    #
    #         elif order > 2:
    #             grad = 0
    #
    #         else:
    #             raise ValueError('order must be an integer greater than or equal to 1.')
    #
    #     else:
    #         raise ValueError('Invalid extrapolator index. Must be -1 for lower and shape-1 for upper extrapolation.')
    #
    #     return grad


cdef class _ArrayDerivative1D:
    """
    Gradient method that returns the approximate derivative of a desired order at a specified grid point.

    These methods of finding derivatives are only valid on a 1D grid of points, at the values at the points. Other
    derivative method would be dependent on the interpolator types.

    :param x: 1D memory view of the spline point x positions.
    :param f: 1D memory view of the function value at spline point x positions.
    """
    def __init__(self, double[::1] x, double[::1] f):

        self._x = x
        self._f = f
        self._last_index = self._x.shape[0] - 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, int index, bint rescale_norm) except? -1e999:
        """
        Evaluate the derivative of specific order at a grid point.

        The array of spline knots is reduced to a 2 to 3 points for gradient evaluation depending on if the requested
        derivative is near the edge or not (respectively).

        :param index: The lower index of the x array cell to evaluate.
        :param rescale_norm: A boolean as whether to rescale to the delta before x[index] or after (default).
        """

        cdef double dfdn

        # Find if at the edge of the grid, and in what direction. Then evaluate the gradient.
        if index == 0:
            return self._evaluate_edge_x(index)
        elif index == self._last_index:
            return self._evaluate_edge_x(index - 1)

        dfdn = self._evaluate_x(index)
        if rescale_norm:
            return self._rescale_lower_normalisation(dfdn, self._x[index - 1], self._x[index], self._x[index + 1])
        return dfdn

    @cython.cdivision(True)
    cdef double _rescale_lower_normalisation(self, double dfdn, double x_lower, double x, double x_upper):
        """
        Derivatives that are normalised to the unit square (x_upper - x) = 1 are un-normalised, then re-normalised to
        (x - x_lower)
        """
        return dfdn * (x - x_lower) / (x_upper - x)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate_edge_x(self, int index):
        """
        Calculate the 1st derivative on an unevenly spaced array as a 1st order approximation.
        
        A taylor expansion of f(x) with changes in x:
        f(x+dx0) = f(x) + dx0*fx(x) + O^2(dx0)
        Can be rearranged to find fx(x)
        fx(x) = (f(x+dx0) - f(x))/dx0
        On unit normalisation dx0 = 1, so this is te final equation. At either edge of the grid, the gradient is 
        normalised to the first or last array width to make sure this is always the case.
        """

        return self._f[index + 1] - self._f[index]

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate_x(self, int index):
        """
        Calculate the 1st derivative on an unevenly spaced array as a 2nd order approximation.

        A taylor expansion of f(x) with changes in x:
        
            f(x+dx0) = f(x) + dx0*fx(x) + dx0^2*fxx(x)/2 + O^3(dx0)
            f(x-dx1) = f(x) - dx1*fx(x) + dx1^2*fxx(x)/2 + O^3(dx1)
        
        Can be multiplied by dx1^2 or dx0^2 respectively then taken away to rearrange for the derivative fx(x) as
        second order terms cancel, this is a second order approximation.
        
            f(x+dx0)*dx1^2 - f(x-dx1)*dx0^2 = f(x)*(dx1^2 - dx0^2) + dx0*fx(x)*dx1^2 + dx1*fx(x)*dx0^2
            fx(x) = [f(x+dx0)*dx1^2 - f(x-dx1)*dx0^2 - f(x)*(dx1^2 - dx0^2)] / (dx0*dx1^2 + dx1*dx0^2)
        
        Which simplifies in the unit normalisation (such that dx0 = 1) to:
        
            fx(x) = [f(x+dx0)*dx1^2 - f(x-dx1) - f(x)*(dx1^2 - 1)]/(dx1^2 + dx1)

        If dx0 = dx1 the central difference approximation is recovered.
        
        .. WARNING:: For speed, this function does not perform any zero division, type or bounds
          checking. Supplying malformed data may result in data corruption or a
          segmentation fault.
        
        """

        cdef double x1_n = (self._x[index] - self._x[index - 1]) / (self._x[index + 1] - self._x[index])
        cdef double x1_n2 = x1_n * x1_n
        return (self._f[index + 1]*x1_n2 - self._f[index - 1] - self._f[index]*(x1_n2 - 1.0)) / (x1_n + x1_n2)


id_to_interpolator = {
    _Interpolator1DLinear.ID: _Interpolator1DLinear,
    _Interpolator1DCubic.ID: _Interpolator1DCubic
}


id_to_extrapolator = {
    _Extrapolator1DNone.ID: _Extrapolator1DNone,
    _Extrapolator1DNearest.ID: _Extrapolator1DNearest,
    _Extrapolator1DLinear.ID: _Extrapolator1DLinear,
    _Extrapolator1DQuadratic.ID: _Extrapolator1DQuadratic
}


permitted_interpolation_combinations = {
    _Interpolator1DLinear.ID: [_Extrapolator1DNone.ID, _Extrapolator1DNearest.ID, _Extrapolator1DLinear.ID],
    _Interpolator1DCubic.ID: [_Extrapolator1DNone.ID, _Extrapolator1DNearest.ID, _Extrapolator1DLinear.ID,
                              _Extrapolator1DQuadratic.ID]
}
