# cython: language_level=3

# Copyright (c) 2014-2020, Dr Alex Meakins, Raysect Project
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

Interpolators are accessed through interface class Interpolate1D.
"""

import numpy as np
cimport cython
from raysect.core.math.cython.interpolation.linear cimport linear1d
from raysect.core.math.cython.interpolation.cubic cimport calc_coefficients_1d, evaluate_cubic_1d
from raysect.core.math.cython.utility cimport find_index, lerp


cdef class Interpolate1D(Function1D):
    """
    Interface class for Function1D interpolators.

    Coordinate array (x) and data array (f) are sorted and transformed into Numpy arrays.
    The resulting Numpy arrays are stored as read only. I.e. `writeable` flag of self.x and self.f
    is set to False. Alteration of the flag may result in wanted behaviour.

    :note: x and f arrays must be of equal length.

    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.

    :param str interpolation_type: Type of interpolation to use. Options are:
    `linear_interp`: Interpolates the data using linear interpolation.
    `cubic_interp`: Interpolates the data using cubic interpolation.

    :param str extrapolation_type: Type of extrapolation to use. Options are:
    `no_extrap`: Attempt to access data outside of x's range will yield ValueError
    `nearest_extrap`: Extrapolation results is the nearest position x value in the interpolation domain.
    `linear_extrap`: Extrapolate linearly the interpolation function
    `cubic_extrap`: Extrapolate cubically the interpolation function

    :param double extrapolation_range: Limits the range where extrapolation is permitted. Requesting data beyond the
    extrapolation range results in ValueError. Extrapolation range will be applied as padding symmetrically to both
    ends of the interpolation range (x).
    """

    def __init__(self, object x, object f, str interpolation_type,
                 str extrapolation_type, double extrapolation_range):


        self.x = np.array(x, dtype=np.float64)
        self.x.flags.writeable = False
        self.f = np.array(f, dtype=np.float64)
        self.f.flags.writeable = False

        self._x_mv = x
        self._f_mv = f
        self._last_index = self.x.shape[0] - 1
        self._extrapolation_range = extrapolation_range

        # dimensions checks
        if x.ndim != 1:
            raise ValueError(f'The x array must be 1D. Got {x.shape}.')

        if f.ndim != 1:
            raise ValueError(f'The x array must be 1D. Got {f.shape}.')

        if x.shape != f.shape:
            raise ValueError(f'Shape mismatch between x array ({x.shape}) and f array ({f.shape}).')

        # test monotonicity
        if (np.diff(x) <= 0).any():
            raise ValueError('The x array must be monotonically increasing.')

        # create interpolator per interapolation_type argument
        interpolation_type = interpolation_type.lower()
        if interpolation_type not in id_to_interpolator:
            raise ValueError(f'Interpolation type {interpolation_type} not found. options are {id_to_interpolator.keys()}')


        self._interpolator = id_to_interpolator[interpolation_type](self._x_mv, self._f_mv)

        # create extrapolator per extrapolation_type argument
        extrapolation_type = extrapolation_type.lower()
        if extrapolation_type not in id_to_extrapolator:
            raise ValueError(f'Extrapolation type {interpolation_type} not found. options are {id_to_extrapolator.keys()}')

        self._extrapolator = id_to_extrapolator[extrapolation_type](self._x_mv, self._f_mv, extrapolation_range)


    cdef double evaluate(self, double px) except? -1e999:
        """
        Evaluates the interpolating function.

        :param double px: the point for which an interpolated value is required
        :return: the interpolated value at point x.
        """
        cdef int index = find_index(self._x_mv, px)

        if index == -1:
            if px < self._x_mv[0] - self._extrapolation_range:
                raise ValueError(
                    f'The specified value (x={px}) is outside of extrapolation range.')
            return self._extrapolator.evaluate(px, index)
        elif index == self._last_index:
            if px > self._x_mv[self._last_index] + self._extrapolation_range:
                raise ValueError(
                    f'The specified value (x={px}) is outside of extrapolation range.')
            return self._extrapolator.evaluate(px, index)
        else:
            return self._interpolator.evaluate(px, index)

    @property
    def domain(self):
        """
        Returns min/max interval of 'x' array.
        Order: min(x), max(x)
        """
        return np.min(self._x_mv), np.max(self._x_mv)


cdef class _Interpolator1D:
    """Base class for 1D interpolators. """

    ID = NotImplemented
    def __init__(self, double[::1] x, double[::1] f):
        self._x = x
        self._f = f
        self._last_index = self._x.shape[0] - 1

    cdef double evaluate(self, double px, int index) except? -1e999:
        """
        Calculates interpolated value at given point. 
    
        :param double px: the point for which an interpolated value is required
        :param int index: the lower index of the bin containing point px. (Result of bisection search).   
        """
        raise NotImplementedError('_Interpolator is an abstract base class.')


cdef class _Interpolator1DLinear(_Interpolator1D):
    """
    Linear interpolation of 1D function.

    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    :param Extrapolator1D extrapolator: extrapolator object
    """

    ID = 'linear'

    def __init__(self, double[::1] x, double[::1] f):
        super().__init__(x, f)

    cdef double evaluate(self, double px, int index) except? -1e999:
        return linear1d(self._x[index], self._x[index + 1], self._f[index], self._f[index + 1], px)


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

        # Where 'a' has been calculated already the mask value = 1
        self._mask_a = np.zeros((self._last_index,), dtype=np.float64)
        self._a = np.zeros((self._last_index, 4), dtype=np.float64)
        self._a_mv = self._a

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef double _calc_gradient(self, double[::1] x_spline, double[::1] y_spline, int index):
        """
        Calculate the normalised gradient at x_spline[index] based on the central difference approximation unless at
        the edges of the array x_spline.

        At x[i], the gradient is normally estimated using the central difference approximation [y[i-1], y[i+1]]/2
        For a normalised range x[i], x[i+1] between 0 and 1, this is the same except for unevenly spaced data.
        Unevenly spaced data has a normalisation x[i-1] - x[i+1] != 2, it is defined as x_eff in this function by
        re-scaling the distance x[i-1] - x[i+1] using normalisation (x[i+1] - x[i]) = 1.

        At the start and end of the array, the forward or backward difference approximation is calculated over
        a  (x[i+1] - x[i]) = 1 or  (x[i] - x[i-1]) = 1 respectively. The end spline gradient is not used for
        extrapolation

        .. WARNING:: For speed, this function does not perform any zero division, type or bounds
          checking. Supplying malformed data may result in data corruption or a
          segmentation fault.

        :param x_spline: A memory view to a double array containing monotonically increasing values.
        :param y_spline: The desired spline points corresponding function returned values
        :param int index: The index of the lower spline point that the gradient is to be calculated for
        """
        cdef double dfdx
        cdef double x_eff
        if index == 0:
            dfdx = (y_spline[index + 1] - y_spline[index])
        elif index == self._last_index:
            dfdx = y_spline[index] - y_spline[index - 1]
        else:
            # Finding the normalised distance x_eff
            x_eff = (x_spline[index + 1] - x_spline[index - 1])/(x_spline[index + 1] - x_spline[index])
            if x_eff != 0:
                dfdx = (y_spline[index + 1]-y_spline[index - 1])/x_eff
            else:
                raise ZeroDivisionError('Two adjacent spline points have the same x value!')
        return dfdx

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double px, int index) except? -1e999:

        # rescale x between 0 and 1
        cdef double x_scal
        cdef double[2] f, dfdx
        cdef double x_bound
        cdef double[4] a

        x_bound = self._x[index + 1] - self._x[index]
        if x_bound != 0:
            x_scal = (px - self._x[index]) / x_bound
        else:
            raise ZeroDivisionError('Two adjacent spline points have the same x value!')

        # Calculate the coefficients (and gradients at each spline point) if they dont exist
        if not self._mask_a[index]:
            f[0] = self._f[index]
            f[1] = self._f[index + 1]
            dfdx[0] = self._calc_gradient(self._x, self._f, index)
            dfdx[1] = self._calc_gradient(self._x, self._f, index + 1)

            calc_coefficients_1d(f, dfdx, a)
            self._a_mv[index, :] = a
            self._mask_a[index] = 1
        else:
            a = self._a[index, :4]
        return evaluate_cubic_1d(a, x_scal)

    def _test_return_polynormial_coefficients(self, index_use):
        """ Expose cython function for testing. Input the index of the lower x spline point in the region of the spline"""
        cdef double[4] a
        cdef double[2] f, dfdx

        a_return = np.zeros((4, ))

        f[0] = self._f[index_use]
        f[1] = self._f[index_use + 1]
        dfdx[0] = self._calc_gradient(self._x, self._f, index_use)
        dfdx[1] = self._calc_gradient(self._x, self._f, index_use + 1)
        calc_coefficients_1d(f, dfdx, a)
        a_return = a
        return a_return

    def _test_calc_gradient(self, index_use):
        """ Expose cython function for testing. Input the spline points x, f"""
        return self._calc_gradient(self._x, self._f,  index_use)

    def _test_evaluate_directly(self, x):
        cdef int index = find_index(self._x, x)

        """ Expose cython function for testing. Input the spline points x, f"""
        return self.evaluate(x, index)


cdef class _Extrapolator1D:
    """
    Base class for Function1D extrapolators.

    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    :param double extrapolation_range: Range covered by the extrapolator. Padded symmetrically to both ends of the input.
    """

    ID = NotImplemented

    def __init__(self, double[::1] x, double[::1] f, double extrapolation_range):
        self._range = extrapolation_range
        self._x = x
        self._f = f
        self._last_index = self._x.shape[0] - 1

    cdef double evaluate(self, double px, int index) except? -1e999:
        raise NotImplementedError(f'{self.__class__} not implemented.')


cdef class _ExtrapolatorNone(_Extrapolator1D):
    """
    Extrapolator that does nothing.
    """

    ID = 'none'

    def __init__(self, double [::1] x, double[::1] f, double extrapolation_range):
           super().__init__(x, f, extrapolation_range)

    cdef double evaluate(self, double px, int index)  except? -1e999:
        raise ValueError(f'Extrapolation not available. Interpolate within function range {np.min(self._x)}-{np.max(self._x)}.')


cdef class _Extrapolator1DNearest(_Extrapolator1D):
    """
    Extrapolator that returns nearest input value
    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    :param double extrapolation_range: Range covered by the extrapolator.
    """

    ID = 'nearest'

    def __init__(self, double [::1] x, double[::1] f, double extrapolation_range):
        super().__init__(x, f, extrapolation_range)

    cdef double evaluate(self, double px, int index) except? -1e999:
        if px < self._x[0]:
            return self._f[0]
        elif px >= self._x[self._last_index]:
            return self._f[self._last_index]
        else:
            raise ValueError(f'Cannot evaluate value of function at point {px}. Bad data?')


cdef class _Extrapolator1DLinear(_Extrapolator1D):
    """
    Extrapolator that extrapolates linearly
    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    :param double extrapolation_range: Range covered by the extrapolator.
    """

    ID = 'linear'

    def __init__(self, double [::1] x, double[::1] f, double extrapolation_range):
        super().__init__(x, f, extrapolation_range)

        if x.shape[0] <= 1:
            raise ValueError(f'x array {np.shape(x)} must contain at least 2 spline points to linearly extrapolate.')

    cdef double evaluate(self, double px, int index) except? -1e999:
        # The index returned from find_index is -1 at the array start or the length of the array at the end of array
        if index == -1:
            index += 1
        elif index == self._last_index:
            index -= 1
        else:
            raise ValueError('Invalid extrapolator index. Must be -1 for lower and shape-1 for upper extrapolation')
        # Use a linear interpolator function to extrapolate instead
        return lerp(self._x[index], self._x[index + 1], self._f[index], self._f[index + 1], px)


cdef class _Extrapolator1DQuadratic(_Extrapolator1D):
    """
    Extrapolator that extrapolates quadratically
    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    :param double extrapolation_range: Range covered by the extrapolator.
    """

    ID = 'quadratic'

    def __init__(self, double [::1] x, double[::1] f, double extrapolation_range):
        cdef double[2] dfdx_start, dfdx_end

        super().__init__(x, f, extrapolation_range)
        self._last_index = self._x.shape[0] - 1

        dfdx_start[0] = self._calc_gradient(self._x, self._f, 0)
        dfdx_start[1] = self._calc_gradient(self._x, self._f, 1)

        dfdx_end[0] = self._calc_gradient(self._x, self._f, self._last_index - 1)
        dfdx_end[1] = self._calc_gradient(self._x, self._f, self._last_index)

        self._calculate_quadratic_coefficients_start(f[0], dfdx_start[0], dfdx_start[1], self._a_first)
        self._calculate_quadratic_coefficients_end(f[self._last_index],  dfdx_end[0], dfdx_end[1], self._a_last)
        if x.shape[0] <= 1:
            raise ValueError(
                f'x array {np.shape(x)} must contain at least 2 spline points to quadratically extrapolate.'
            )

    cdef void _calculate_quadratic_coefficients_start(self, double f1, double df1_dx, double df2_dx, double[3] a):
        """
        Calculate the coefficients for a quadratic spline where 2 spline knots are normalised to between 0 and 1, 


        """
        a[0] = -0.5*df1_dx + 0.5*df2_dx
        a[1] = df1_dx
        a[2] = f1

    cdef void _calculate_quadratic_coefficients_end(self, double f2, double df1_dx, double df2_dx, double[3] a):
        """
        Calculate the coefficients for a quadratic spline where 2 spline knots are normalised to between 0 and 1, 


        """
        a[0] = - 0.5*df1_dx + 0.5*df2_dx
        a[1] = df1_dx
        a[2] = f2 - 0.5*df1_dx - 0.5*df2_dx

    cdef double evaluate(self, double px, int index) except? -1e999:
        # The index returned from find_index is -1 at the array start or the length of the array at the end of array
        cdef double f_return
        cdef double x_scal
        if index == -1:
            index += 1
            x_scal =  (px - self._x[index])/(self._x[index + 1] - self._x[index])
            f_return = self._a_first[0]*x_scal**2 + self._a_first[1]*x_scal + self._a_first[2]
        elif index == self._last_index:
            index -= 1
            x_scal = (px - self._x[index])/(self._x[index + 1] - self._x[index])
            f_return = self._a_last[0]*x_scal**2 + self._a_last[1]*x_scal + self._a_last[2]
        else:
            raise ValueError('Invalid extrapolator index. Must be -1 for lower and shape-1 for upper extrapolation')

        return f_return

    def _test_evaluate_directly(self, x):
        cdef int index = find_index(self._x, x)

        """ Expose cython function for testing. Input the spline points x, f"""
        return self.evaluate(x, index)
    def _test_first_coefficients(self):
        """ Expose cython function for testing."""
        return self._a_first

    def _test_last_coefficients(self):
        """ Expose cython function for testing."""
        return self._a_last

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef double _calc_gradient(self, double[::1] x_spline, double[::1] y_spline, int index):
        """
        Calculate the normalised gradient at x_spline[index] based on the central difference approximation unless at
        the edges of the array x_spline.

        At x[i], the gradient is normally estimated using the central difference approximation [y[i-1], y[i+1]]/2
        For a normalised range x[i], x[i+1] between 0 and 1, this is the same except for unevenly spaced data.
        Unevenly spaced data has a normalisation x[i-1] - x[i+1] != 2, it is defined as x_eff in this function by
        re-scaling the distance x[i-1] - x[i+1] using normalisation (x[i+1] - x[i]) = 1.

        At the start and end of the array, the forward or backward difference approximation is calculated over
        a  (x[i+1] - x[i]) = 1 or  (x[i] - x[i-1]) = 1 respectively. The end spline gradient is not used for
        extrapolation

        .. WARNING:: For speed, this function does not perform any zero division, type or bounds
          checking. Supplying malformed data may result in data corruption or a
          segmentation fault.

        :param x_spline: A memory view to a double array containing monotonically increasing values.
        :param y_spline: The desired spline points corresponding function returned values
        :param int index: The index of the lower spline point that the gradient is to be calculated for
        """
        cdef double dfdx
        cdef double x_eff
        if index == 0:
            dfdx = (y_spline[index + 1] - y_spline[index])
        elif index == self._last_index:
            dfdx = y_spline[index] - y_spline[index - 1]
        else:
            # Finding the normalised distance x_eff
            x_eff = (x_spline[index + 1] - x_spline[index - 1])/(x_spline[index + 1] - x_spline[index])
            if x_eff != 0:
                dfdx = (y_spline[index + 1]-y_spline[index - 1])/x_eff
            else:
                raise ZeroDivisionError('Two adjacent spline points have the same x value!')
        return dfdx

id_to_interpolator = {
    _Interpolator1DLinear.ID: _Interpolator1DLinear,
    _Interpolator1DCubic.ID: _Interpolator1DCubic
}

id_to_extrapolator = {
    _ExtrapolatorNone.ID: _ExtrapolatorNone,
    _Extrapolator1DNearest.ID: _Extrapolator1DNearest,
    _Extrapolator1DLinear.ID: _Extrapolator1DLinear,
    _Extrapolator1DQuadratic.ID: _Extrapolator1DQuadratic
}