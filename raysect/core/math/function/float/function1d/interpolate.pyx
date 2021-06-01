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

Interpolators are accessed through interface class Interpolate1D, which
"""

import numpy as np
cimport cython
from raysect.core.math.cython.interpolation.linear cimport linear1d
from raysect.core.math.cython.interpolation.cubic cimport calc_coefficients_1d, evaluate_cubic_1d
from raysect.core.math.cython.utility cimport find_index, lerp

DEF INT_LINEAR = 0
DEF INT_CUBIC = 1
DEF INT_CUBIC_CONSTRAINED = 2

_INTERPOLATION_TYPES = {
    'linear': INT_LINEAR,
    'cubic': INT_CUBIC,
    'cubic_constrained': INT_CUBIC_CONSTRAINED
}

DEF EXT_NONE = 0
DEF EXT_NEAREST = 1
DEF EXT_LINEAR = 2
DEF EXT_QUADRATIC = 3

_EXTRAPOLATION_TYPES = {
    'none': EXT_NONE,
    'nearest': EXT_NEAREST,
    'linear': EXT_LINEAR,
    'quadratic': EXT_QUADRATIC
}

cdef class Interpolate1D(Function1D):
    """
    Interface class for Function1D interpolators

    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    :param str interpolation_type: Type of interpolation to use ('linear', 'cubic', 'cubic_constrained')
    :param str extrapolation_type: Type of extrapolation to use ('none', 'nearest', 'linear', 'quadratic')
    """
    def __init__(self, object x, object f, str interpolation_type,
                 str extrapolation_type, double extrapolation_range):

        interpolation_type = interpolation_type.lower()
        if  interpolation_type not in _INTERPOLATION_TYPES:
            raise ValueError(f'Interpolation type {interpolation_type} not found.')

        cdef int interpolator_idx = _INTERPOLATION_TYPES.get(interpolation_type)

        extrapolation_type = extrapolation_type.lower()
        if extrapolation_type not in _EXTRAPOLATION_TYPES:
            raise ValueError(f'Extrapolation type {extrapolation_type} not found.')

        cdef int extrapolator_idx = _EXTRAPOLATION_TYPES.get(extrapolation_type)

        self.x = np.array(x, dtype=np.float64)
        self.f = np.array(f, dtype=np.float64)
        self._x = x
        self._f = f
        self._last_index = self._x.shape[0] -1
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

        # create extrapolator per extrapolation_type argument
        if extrapolator_idx == EXT_NONE:
            self._extrapolator = _ExtrapolatorNone(x, f, extrapolation_range)
        elif extrapolator_idx == EXT_NEAREST:
            self._extrapolator = _Extrapolator1DNearest(x, f, extrapolation_range)
        elif extrapolator_idx == EXT_LINEAR:
            self._extrapolator = _Extrapolator1DLinear(x, f, extrapolation_range)
        elif extrapolator_idx == EXT_QUADRATIC:
            raise NotImplementedError("Quadratic extrapolator not implemented.")
        else:
            raise ValueError(f'Unsupported extrapolator type {extrapolation_type}.')

        # create extrapolator per interpolation_type argument
        if interpolator_idx == INT_LINEAR:
            self._interpolator = _Interpolator1DLinear(self._x, self._f)
        elif interpolator_idx == INT_CUBIC:
            self._interpolator = _Interpolator1DCubic(self._x, self._f)
        # elif interpolator_idx == INT_CUBIC_CONSTRAINED:
        #     raise NotImplementedError("Constrained cubic interpolator not implemented.")
        else:
            raise ValueError(f'Interpolation type {interpolation_type} not supported.')

    cdef double evaluate(self, double x) except? -1e999:
        """
        Evaluates the interpolating function.

        :param double x: the point for which an interpolated value is required
        :return: the interpolated value at point x.
        """
        cdef int index = find_index(self._x, x)

        if index == -1:
            if x < self._x[0] - self._extrapolation_range:
                raise ValueError(
                    f'The specified value (x={x}) is outside of extrapolation range.')
            return self._extrapolator.evaluate(x, index)
        elif index == self._last_index:
            if x > self._x[self._last_index] + self._extrapolation_range:
                raise ValueError(
                    f'The specified value (x={x}) is outside of extrapolation range.')
            return self._extrapolator.evaluate(x, index)
        else:
            return self._interpolator.evaluate(x, index)

    @property
    def domain(self):
        """
        Returns bounding box of the provided inputs.
        Order: min(x), max(x), min(f), max(f)
        :warning: doesn't take extrapolator into account at the moment
        """
        return np.min(self._x), np.max(self._x), np.min(self._f), np.max(self._f)


cdef class _Interpolator1D:
    """Base class for 1D interpolators. """

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
    def __init__(self, double[::1] x, double[::1] f):
        self._x = x
        self._f = f

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
    def __init__(self, double[::1] x, double[::1] f):
        self._x = x
        self._f = f

        cdef int n
        n = len(x)
        self._n = n
        # Where 'a' has been calculated already the mask value = 1
        self._mask_a = np.zeros((n - 1,), dtype=np.float64)
        self._a = np.zeros((n - 1, 4), dtype=np.float64)
        self._a_mv = self._a

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef double get_gradient(self, double[::1] x_spline, double[::1] y_spline, int index):
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
        elif index == self._n - 1:
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
        cdef int index_use = find_index(self._x, px)
        # rescale x between 0 and 1
        cdef double x_scal
        cdef double[2] f, dfdx

        cdef double x_bound = (self._x[index_use + 1] - self._x[index_use])
        if x_bound != 0:
            x_scal = (px - self._x[index_use]) / x_bound
        else:
            raise ZeroDivisionError('Two adjacent spline points have the same x value!')

        # Calculate the coefficients (and gradients at each spline point) if they dont exist
        cdef double[4] a
        if not self._mask_a[index_use]:
            f[0] = self._f[index_use]
            f[1] = self._f[index_use + 1]
            dfdx[0] = self.get_gradient(self._x, self._f, index_use)
            dfdx[1] = self.get_gradient(self._x, self._f, index_use + 1)

            calc_coefficients_1d(f, dfdx, a)
            self._a_mv[index_use, :] = a
            self._mask_a[index_use] = 1
        else:
            a = self._a[index_use, :4]
        return evaluate_cubic_1d(a, x_scal)

    def _test_return_polynormial_coefficients(self, index_use):
        """ Expose cython function for testing. Input the index of the lower x spline point in the region of the spline"""
        a_return = np.zeros((4, ))
        cdef double[4] a
        cdef double[2] f, dfdx
        f[0] = self._f[index_use]
        f[1] = self._f[index_use + 1]
        dfdx[0] = self.get_gradient(self._x, self._f, index_use)
        dfdx[1] = self.get_gradient(self._x, self._f, index_use + 1)
        calc_coefficients_1d(f, dfdx, a)
        a_return = a
        return a_return

    def _test_get_gradient(self, index_use):
        """ Expose cython function for testing. Input the spline points x, f"""
        return self.get_gradient(self._x, self._f,  index_use)

    def _test_evaluate_directly(self, x):
        cdef int index = find_index(self._x, x)

        """ Expose cython function for testing. Input the spline points x, f"""
        return self.evaluate(x, index)


cdef class _Interpolator1DCubicConstrained(_Interpolator1DCubic):
    """
    Cubic interpolation of 1D function, with constrained gradients at spline points near a maximum

    When called, stores cubic polynomial coefficients from the value of the function at the neighboring spline points
    and the gradient at the neighbouring spline points based on central difference gradients. The polynomial
    coefficients and gradients are calculated between each spline knots normalised to between 0 and 1.
    The gradients around the spline knots are constrained to 0

    :param x: 1D memory view of the spline point x positions.
    :param f: 1D memory view of the function value at spline point x positions.
    """
    def __init__(self, double[::1] x, double[::1] f):
        super().__init__(x, f)

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef double get_gradient(self, double[::1] x_spline, double[::1] y_spline, int index):
        """
        Calculate the normalised gradient at x_spline[index] based on the central difference approximation unless at
        the edges of the array x_spline.

        At x[i], the gradient is normally estimated using the central difference approximation [y[i-1], y[i+1]]/2
        For a normalised range x[i], x[i+1] between 0 and 1, this is the same except for unevenly spaced data.
        Unevenly spaced data has a normalisation x[i-1] - x[i+1] != 2, it is defined as x_eff in this function by
        re-scaling the distance x[i-1] - x[i+1] using normalisation (x[i+1] - x[i]) = 1.

        At the start and end of the array, the forward or backward difference approximation is calculated over
        a  (x[i+1] - x[i]) = 1 or  (x[i] - x[i-1]) = 1 respectively. The end spline gradient is not used for
        extrapolation. Additionally if the spline knot value is a minimum to the points surrounding it then

        .. WARNING:: For speed, this function does not perform any zero division, type or bounds
          checking. Supplying malformed data may result in data corruption or a
          segmentation fault.

        :param x_spline: A memory view to a double array containing monotonically increasing values.
        :param y_spline: The desired spline points corresponding function returned values
        :param int index: The index of the lower spline point that the gradient is to be calculated for
        """
        # Calculate central difference method, but at the start of end of the array use the forward/back difference
        cdef double dfdx
        cdef double x_eff
        if index == 0:
            dfdx = (y_spline[index + 1] - y_spline[index])
        elif index == self._n - 1:
            dfdx = y_spline[index] - y_spline[index - 1]
        else:
            gradient_set_0 = False
            # Set to 0 around the central point
            if y_spline[index + 1] < y_spline[index] and y_spline[index - 1] < y_spline[index]:
                dfdx = 0.
                gradient_set_0 = True
            elif y_spline[index + 1] > y_spline[index] and y_spline[index - 1] > y_spline[index]:
                dfdx = 0.
                gradient_set_0 = True

            # Set to 0 around the next point
            if index != self._n - 2:
                if y_spline[index + 2] < y_spline[index + 1] and y_spline[index] < y_spline[index + 1]:
                    dfdx = 0.
                    gradient_set_0 = True
                elif y_spline[index + 2] > y_spline[index + 1] and y_spline[index] > y_spline[index + 1]:
                    dfdx = 0.
                    gradient_set_0 = True

            # Set to 0 around the previous point
            if index != 1:
                if y_spline[index] < y_spline[index - 1] and y_spline[index-2] < y_spline[index - 1]:
                    dfdx = 0.
                    gradient_set_0 = True
                elif y_spline[index] > y_spline[index - 1] and y_spline[index-2] > y_spline[index - 1]:
                    dfdx = 0.
                    gradient_set_0 = True
            if not gradient_set_0:
                # Finding the normalised distance x_eff
                x_eff = (x_spline[index + 1] - x_spline[index - 1])/(x_spline[index + 1] - x_spline[index])
                if x_eff != 0:
                    dfdx = (y_spline[index + 1] - y_spline[index - 1])/x_eff
                else:
                    raise ZeroDivisionError('Two adjacent spline points have the same x value!')
        return dfdx


cdef class _Extrapolator1D:
    """
    Base class for Function1D extrapolators.

    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    :param double extrapolation_range: Range covered by the extrapolator. Padded symmetrically to both ends of the input.
    """
    typename = NotImplemented

    def __init__(self, double[::1] x, double[::1] f, double extrapolation_range):
        self._range = extrapolation_range
        self._x = x
        self._f = f

    cdef double extrapolate(self, double px, int order, int index, double rx) except? -1e999:
        raise NotImplementedError(f'{self.__class__} not implemented.')

    cdef double evaluate(self, double px, int index) except? -1e999:
        raise NotImplementedError(f'{self.__class__} not implemented.')


cdef class _ExtrapolatorNone(_Extrapolator1D):
    """
    Extrapolator that does nothing.
    """
    cdef double extrapolate(self, double px, int order, int index, double rx) except? -1e999:
        raise ValueError(f'Extrapolation not available. Interpolate within function range {np.min(self._x)}-{np.max(self._x)}.')

    cdef double evaluate(self, double px, int index)  except? -1e999:
        raise ValueError('Extrapolation not available.')


cdef class _Extrapolator1DNearest(_Extrapolator1D):
    """
    Extrapolator that returns nearest input value
    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    :param double extrapolation_range: Range covered by the extrapolator.
    """

    typename = 'nearest'

    def __init__(self, double [::1] x, double[::1] f, double extrapolation_range):
        super().__init__(x, f, extrapolation_range)
        self._last_index = self._x.shape[0] -1

    cdef double evaluate(self, double px, int index) except? -1e999:
        if px < self._x[0]:
            return self._f[0]
        elif px >= self._x[self._last_index]:
            return self._f[self._last_index]


cdef class _Extrapolator1DLinear(_Extrapolator1D):
    """
    Extrapolator that extrapolates linearly
    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    :param double extrapolation_range: Range covered by the extrapolator.
    """

    typename = 'linear'

    def __init__(self, double [::1] x, double[::1] f, double extrapolation_range):
        super().__init__(x, f, extrapolation_range)
        self._last_index = self._x.shape[0] -1

        if x.shape[0] <= 1:
            raise ValueError(f'x array {np.shape(x)} must contain at least 2 spline points to linearly extrapolate.')

    cdef double evaluate(self, double px, int index) except? -1e999:
        # The index returned from find_index is -1 at the array start or the length of the array at the end of array
        if index == -1:
            index += 1
        else:
            index -= 1
        # Use a linear interpolator function to extrapolate instead
        return lerp(self._x[index], self._x[index + 1], self._f[index], self._f[index + 1], px)


cdef class _Extrapolator1DQuadratic(_Extrapolator1D):
    """
    Extrapolator that extrapolates quadratically


    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    :param double extrapolation_range: Range covered by the extrapolator.
    """

    typename = 'quadratic'

    def __init__(self, double [::1] x, double[::1] f, double extrapolation_range):
        super().__init__(x, f, extrapolation_range)
        self._last_index = self._x.shape[0] - 1

        self.calculate_quadratic_coefficients(f[0], f[1], f[2], (x[2]-x[0])/(x[1]-x[0]), self._a_first)
        self.calculate_quadratic_coefficients(
            f[self._last_index-2], f[self._last_index-1], f[self._last_index],
            (x[self._last_index]-x[self._last_index-2])/(x[self._last_index-1]-x[self._last_index-2]), self._a_last
        )

        if x.shape[0] <= 1:
            raise ValueError(
                f'x array {np.shape(x)} must contain at least 2 spline points to quadratically extrapolate.'
            )

    cdef calculate_quadratic_coefficients(self, double f1, double f2, double f3, double x_scal_3, double[3] a):
        """
        Calculate the coefficients for a quadratic spline where 2 spline knots are normalised to between 0 and 1, 
        
        The first 2 spline knots are normalised between 0 and 1 and the 3rd (higher x) point has a normlised value 
        x_scal_3. The inverse of the matrix [[xn**2, xn, 1]...] is calculated then multiplied by [f1, f2, f3] here 
        """
        a[0] = f1*(1.-x_scal_3) + x_scal_3*f2 - f3
        a[1] = -(1. - x_scal_3**2)*f1 - f2*x_scal_3**2 + f3
        a[2] = (x_scal_3-x_scal_3**2)*f1
        a[0] = a[0]/(x_scal_3-x_scal_3**2)
        a[1] = a[1]/(x_scal_3-x_scal_3**2)
        a[2] = a[2]/(x_scal_3-x_scal_3**2)

    cdef double evaluate(self, double px, int index) except? -1e999:
        # The index returned from find_index is -1 at the array start or the length of the array at the end of array
        cdef double f_return
        cdef double x_scal
        if index == -1:
            index += 1
            x_scal =  (px - self._x[index])/(self._x[index + 1] - self._x[index])
            f_return = self._a_first[0]*x_scal**2 + self._a_first[1]*x_scal + self._a_first[2]
        else:
            index -= 1
            x_scal = (px - self._x[index-1])/(self._x[index] - self._x[index-1])
            f_return = self._a_last[0]*x_scal**2 + self._a_last[1]*x_scal + self._a_last[2]

        return f_return

    def _test_first_coefficients(self):
        """ Expose cython function for testing."""
        return self._a_first

    def _test_last_coefficients(self):
        """ Expose cython function for testing."""
        return self._a_last
