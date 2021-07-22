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


cdef double rescale_lower_normalisation(dfdn, x_lower, x, x_upper):
    """
    Derivatives that are normalised to the unt square (x_upper - x) = 1 are un-normalised, then re-normalised to
    (x - x_lower)
    """
    return dfdn * (x - x_lower)/(x_upper - x)


cdef class Interpolate1DArray(Function1D):
    """
    Interface class for Function1D interpolators.

    Coordinate array (x) and data array (f) are sorted and transformed into Numpy arrays.
    The resulting Numpy arrays are stored as read only. I.e. `writeable` flag of self.x and self.f
    is set to False. Alteration of the flag may result in unwanted behaviour.

    :note: x and f arrays must be of equal length.

    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.

    :param str interpolation_type: Type of interpolation to use. Options are:
    `linear_interp`: Interpolates the data using linear interpolation.
    `cubic_interp`: Interpolates the data using cubic interpolation.

    :param str extrapolation_type: Type of extrapolation to use. Options are:
    `no_extrap`: Attempt to access data outside of x's range will yield ValueError.
    `nearest_extrap`: Extrapolation results is the nearest position x value in the interpolation domain.
    `linear_extrap`: Extrapolate linearly the interpolation function.
    `cubic_extrap`: Extrapolate cubically the interpolation function.

    :param double extrapolation_range: Limits the range where extrapolation is permitted. Requesting data beyond the
    extrapolation range results in ValueError. Extrapolation range will be applied as padding symmetrically to both
    ends of the interpolation range (x).
    """

    def __init__(self, object x, object f, str interpolation_type,
                 str extrapolation_type, double extrapolation_range):

        # extrapolation_range must be greater than or equal to 0.
        if extrapolation_range < 0:
            raise ValueError('extrapolation_range must be greater than or equal to 0.')

        # dimensions checks
        if x.ndim != 1:
            raise ValueError(f'The x array must be 1D. Got {x.shape}.')

        if f.ndim != 1:
            raise ValueError(f'The f array must be 1D. Got {f.shape}.')

        if x.shape != f.shape:
            raise ValueError(f'Shape mismatch between x array ({x.shape}) and f array ({f.shape}).')

        # test monotonicity
        if (np.diff(x) <= 0).any():
            raise ValueError('The x array must be monotonically increasing.')

        self.x = np.array(x, dtype=np.float64, order='c')
        self.x.flags.writeable = False
        self.f = np.array(f, dtype=np.float64, order='c')
        self.f.flags.writeable = False

        self._x_mv = x
        self._f_mv = f
        self._last_index = self.x.shape[0] - 1
        self._extrapolation_range = extrapolation_range


        # create interpolator per interapolation_type argument
        interpolation_type = interpolation_type.lower()
        if interpolation_type not in id_to_interpolator:
            raise ValueError(f'Interpolation type {interpolation_type} not found. options are {id_to_interpolator.keys()}')


        self._interpolator = id_to_interpolator[interpolation_type](self._x_mv, self._f_mv)

        # create extrapolator per extrapolation_type argument
        extrapolation_type = extrapolation_type.lower()
        if extrapolation_type not in id_to_extrapolator:
            raise ValueError(f'Extrapolation type {interpolation_type} not found. options are {id_to_extrapolator.keys()}')

        self._extrapolator = id_to_extrapolator[extrapolation_type](self._x_mv, self._f_mv)
        # Permit combinations of interpolator and extrapolator that the order of extrapolator is higher than interpolator
        if extrapolation_type not in permitted_interpolation_combinations[interpolation_type]:
            raise ValueError(
                f'Extrapolation type {extrapolation_type} not compatible with interpolation type {interpolation_type}')

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
        elif index == self._last_index and px > self.x[self._last_index]:
            if px > self._x_mv[self._last_index] + self._extrapolation_range:
                raise ValueError(
                    f'The specified value (x={px}) is outside of extrapolation range.')
            return self._extrapolator.evaluate(px, index)
        elif px == self.x[self._last_index]:
            return self._interpolator.evaluate(px, index - 1)
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
    """
    Base class for 1D interpolators.

    :param x: 1D memory view of the spline point x positions.
    :param f: 1D memory view of the function value at spline point x positions.
    """

    ID = NotImplemented
    def __init__(self, double[::1] x, double[::1] f):
        self._x = x
        self._f = f
        self._last_index = self._x.shape[0] - 1

    cdef double evaluate(self, double px, int index) except? -1e999:
        """
        Calculates interpolated value at given point. 
    
        :param double px: the point for which an interpolated value is required.
        :param int index: the lower index of the bin containing point px. (Result of bisection search).   
        """
        raise NotImplementedError('_Interpolator is an abstract base class.')

    cdef double _analytic_gradient(self, double px, int index, int order):
        """
        Calculates interpolated value at given point. 

        :param double px: the point for which an interpolated value is required.
        :param int index: the lower index of the bin containing point px. (Result of bisection search).   
        """
        raise NotImplementedError('_Interpolator is an abstract base class.')


cdef class _Interpolator1DLinear(_Interpolator1D):
    """
    Linear interpolation of 1D function.

    :param x: 1D memory view of the spline point x positions.
    :param f: 1D memory view of the function value at spline point x positions.
    """

    ID = 'linear'

    def __init__(self, double[::1] x, double[::1] f):
        super().__init__(x, f)

    cdef double evaluate(self, double px, int index) except? -1e999:
        return linear1d(self._x[index], self._x[index + 1], self._f[index], self._f[index + 1], px)

    cdef double _analytic_gradient(self, double px, int index, int order):
        cdef double grad
        if order == 1:
            grad = (self._f[index + 1] - self._f[index])/(self._x[index + 1] - self._x[index])
        elif order > 1:
            grad = 0
        else:
            raise ValueError('order must be an integer greater than or equal to 1')
        return grad


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
            array_derivative = _ArrayDerivative1D(self._x, self._f)
            dfdx[0] = array_derivative.evaluate(index, derivative_order_x=1, rescale_norm_x=False)
            dfdx[1] = array_derivative.evaluate(index + 1, derivative_order_x=1, rescale_norm_x=True)

            calc_coefficients_1d(f, dfdx, a)
            self._a_mv[index, :] = a
            self._mask_a[index] = 1
        else:
            a = self._a[index, :4]
        return evaluate_cubic_1d(a, x_scal)

    cdef double _analytic_gradient(self, double px, int index, int order):
        cdef double grad
        cdef double x_scal
        cdef double[2] f, dfdx
        cdef double x_bound
        cdef double[4] a

        x_bound = self._x[index + 1] - self._x[index]
        if x_bound != 0:
            x_scal = (px - self._x[index]) / x_bound
        else:
            raise ZeroDivisionError('Two adjacent spline points have the same x value!')

        f[0] = self._f[index]
        f[1] = self._f[index + 1]
        array_derivative = _ArrayDerivative1D(self._x, self._f)
        dfdx[0] = array_derivative.evaluate(index, derivative_order_x=1, rescale_norm_x=False)
        dfdx[1] = array_derivative.evaluate(index + 1, derivative_order_x=1, rescale_norm_x=True)

        calc_coefficients_1d(f, dfdx, a)

        if order == 1:
            grad = 3.* a[0] * x_scal **2 + 2.* a[1] * x_scal + a[2]
            grad = grad/(self._x[index + 1] - self._x[index])
        elif order == 2:
            grad = 6.* a[0] * x_scal + 2.* a[1]
            grad = grad/(self._x[index + 1] - self._x[index])**2
        elif order == 3:
            grad = 6.* a[0]
            grad = grad/(self._x[index + 1] - self._x[index])**3
        elif order > 3:
            grad = 0
        else:
            raise ValueError('order must be an integer greater than or equal to 1')
        return grad


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

    cdef double _analytic_gradient(self, double px, int index, int order):
        raise NotImplementedError(f'{self.__class__} not implemented.')


cdef class _Extrapolator1DNone(_Extrapolator1D):
    """
    Extrapolator that does nothing.
    """

    ID = 'none'

    def __init__(self, double [::1] x, double[::1] f):
           super().__init__(x, f)

    cdef double evaluate(self, double px, int index)  except? -1e999:
        raise ValueError(f'Extrapolation not available. Interpolate within function range {np.min(self._x)}-{np.max(self._x)}.')

    cdef double _analytic_gradient(self, double px, int index, int order):
        raise ValueError(f'Extrapolation not available. Interpolate within function range {np.min(self._x)}-{np.max(self._x)}.')


cdef class _Extrapolator1DNearest(_Extrapolator1D):
    """
    Extrapolator that returns nearest input value.

    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    """

    ID = 'nearest'

    def __init__(self, double [::1] x, double[::1] f):
        super().__init__(x, f)

    cdef double evaluate(self, double px, int index) except? -1e999:
        if px < self._x[0]:
            return self._f[0]
        elif px >= self._x[self._last_index]:
            return self._f[self._last_index]
        else:
            raise ValueError(f'Cannot evaluate value of function at point {px}. Bad data?')

    cdef double _analytic_gradient(self, double px, int index, int order):
        cdef double grad = 0.
        return grad


cdef class _Extrapolator1DLinear(_Extrapolator1D):
    """
    Extrapolator that extrapolates linearly.

    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    """

    ID = 'linear'

    def __init__(self, double [::1] x, double[::1] f):
        super().__init__(x, f)

        if x.shape[0] <= 1:
            raise ValueError(f'x array {np.shape(x)} must contain at least 2 spline points to linearly extrapolate.')

    cdef double evaluate(self, double px, int index) except? -1e999:
        # The index returned from find_index is -1 at the array start or the length of the array at the end of array
        if index == -1:
            index += 1
        elif index == self._last_index:
            index -= 1
        else:
            raise ValueError('Invalid extrapolator index. Must be -1 for lower and shape-1 for upper extrapolation.')
        # Use a linear interpolator function to extrapolate instead
        return lerp(self._x[index], self._x[index + 1], self._f[index], self._f[index + 1], px)

    cdef double _analytic_gradient(self, double px, int index, int order):
        cdef double grad
        if index == -1:
            index += 1
        elif index == self._last_index:
            index -= 1
        else:
            raise ValueError('Invalid extrapolator index. Must be -1 for lower and shape-1 for upper extrapolation.')

        if order == 1:
            grad = (self._f[index + 1] - self._f[index]) / (self._x[index + 1] - self._x[index])
        elif order > 1:
            grad = 0
        else:
            raise ValueError('order must be an integer greater than or equal to 1.')
        return grad


cdef class _Extrapolator1DQuadratic(_Extrapolator1D):
    """
    Extrapolator that extrapolates quadratically.

    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    """

    ID = 'quadratic'

    def __init__(self, double [::1] x, double[::1] f):
        cdef double[2] dfdx_start, dfdx_end

        super().__init__(x, f)
        self._last_index = self._x.shape[0] - 1
        array_derivative = _ArrayDerivative1D(self._x, self._f)
        dfdx_start[0] = array_derivative.evaluate(0, derivative_order_x=1, rescale_norm_x=False)
        # Need to have the first derivatives normalised to the distance between spline knot 0->1 (not 1->2),
        # So un-normalise then re-normalise.
        dfdx_start[1] = array_derivative.evaluate(1, derivative_order_x=1, rescale_norm_x=True)#/(x[2] - x[1]))*(x[1] - x[0])

        dfdx_end[0] = array_derivative.evaluate(self._last_index - 1, derivative_order_x=1, rescale_norm_x=False)
        dfdx_end[1] = array_derivative.evaluate(self._last_index, derivative_order_x=1, rescale_norm_x=True)

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
        Calculate the coefficients for a quadratic spline where 2 spline knots are normalised to between 0 and 1. 


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

    cdef double _analytic_gradient(self, double px, int index, int order):
        # The index returned from find_index is -1 at the array start or the length of the array at the end of array
        cdef double grad
        cdef double x_scal
        if index == -1:
            index += 1
            x_scal =  (px - self._x[index])/(self._x[index + 1] - self._x[index])
            if order == 1:
                grad = 2.*self._a_first[0]*x_scal + self._a_first[1]
                grad = grad/(self._x[index + 1] - self._x[index])
            elif order == 2:
                grad = 2.*self._a_first[0]
                grad = grad/(self._x[index + 1] - self._x[index])**2
            elif order > 2:
                grad = 0
            else:
                raise ValueError('order must be an integer greater than or equal to 1')
        elif index == self._last_index:
            index -= 1
            x_scal = (px - self._x[index])/(self._x[index + 1] - self._x[index])
            if order == 1:
                grad = 2.*self._a_last[0]*x_scal + self._a_last[1]
                grad = grad/(self._x[index + 1] - self._x[index])
            elif order == 2:
                grad = 2.*self._a_last[0]
                grad = grad/(self._x[index + 1] - self._x[index])**2
            elif order > 2:
                grad = 0
            else:
                raise ValueError('order must be an integer greater than or equal to 1')
        else:
            raise ValueError('Invalid extrapolator index. Must be -1 for lower and shape-1 for upper extrapolation')
        return grad


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
        self._last_index_x = self._x.shape[0] - 1

    cdef double evaluate(self, int index_x, int derivative_order_x, bint rescale_norm_x) except? -1e999:
        """
        Evaluate the derivative of specific order at a grid point.

        The array of spline knots is reduced to a 2 to 3 points for gradient evaluation depending on if the requested
        derivative is near the edge or not (respectively).

        :param index_x: The lower index of the x array cell to evaluate.
        :param derivative_order_x: An integer of the derivative order x. Only zero if derivative_order_y is nonzero.
        :param rescale_norm_x: A boolean as whether to rescale to the delta before x[index_x] or after (default).
        """
        # Find if at the edge of the grid, and in what direction. Then evaluate the gradient.
        cdef double dfdn

        if index_x == 0:
            dfdn = self._evaluate_edge_x(index_x, derivative_order_x)
        elif index_x == self._last_index_x:
            dfdn = self._evaluate_edge_x(index_x - 1, derivative_order_x)
        else:
            dfdn = self._evaluate_x(index_x, derivative_order_x)
        if rescale_norm_x:
            if not (index_x == 0 or index_x == self._last_index_x):
                for i in range(derivative_order_x):
                    dfdn = rescale_lower_normalisation(dfdn,  self._x[index_x - 1], self._x[index_x], self._x[index_x + 1])
        return dfdn
    #todo should these have an _?
    cdef double _evaluate_edge_x(self, int index_x, int derivative_order_x):
        """
        Calculate the 1st derivative on an unevenly spaced array as a 1st order approximation.
        
        A taylor expansion of f(x) with changes in x:
        f(x+dx0) = f(x) + dx0*fx(x) + O^2(dx0)
        Can be rearranged to find fx(x)
        fx(x) = (f(x+dx0) - f(x))/dx0
        On unit normalisation dx0 = 1, so this is te final equation. At either edge of the grid, the gradient is 
        normalised to the first or last array width to make sure this is always the case.
        """
        return self._f[index_x + 1] - self._f[index_x]

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef double _evaluate_x(self, int index_x, int derivative_order_x):
        """
        Calculate the 1st derivative on an unevenly spaced array as a 2nd order approximation.

        A taylor expansion of f(x) with changes in x:
        f(x+dx0) = f(x) + dx0*fx(x) + dx0^2*fxx(x)/2 + O^3(dx0)
        f(x-dx1) = f(x) - dx1*fx(x) + dx1^2*fxx(x)/2 + O^3(dx1)
        Can be multiplied by dx1^2 or dx0^2 respectively then taken away to rearrange for the derivative fx(x) as
        second order terms cancel, this is a second order approximation.
        f(x+dx0)*dx1^2 - f(x-dx1)*dx0^2 = f(x)*(dx1^2 - dx0^2) +dx0*fx(x)*dx1^2 +dx1*fx(x)*dx0^2
        fx(x) = [f(x+dx0)*dx1^2 - f(x-dx1)*dx0^2 - f(x)*(dx1^2 - dx0^2)]/(dx0*dx1^2 +dx1*dx0^2)
        Which simplifies in the unit normalisation (dx0 = 1) to :
        fx(x) = [f(x+dx0)*dx1^2 - f(x-dx1) - f(x)*(dx1^2 - 1)]/(dx1^2 +dx1)

        If dx0 = dx1 the central difference approximation is recovered.
        
        .. WARNING:: For speed, this function does not perform any zero division, type or bounds
          checking. Supplying malformed data may result in data corruption or a
          segmentation fault.
        
        """
        cdef double x1_n, x1_n2
        x1_n = (self._x[index_x] - self._x[index_x - 1])/(self._x[index_x + 1] - self._x[index_x])
        x1_n2 = x1_n**2
        return (self._f[index_x + 1]*x1_n2 - self._f[index_x - 1] - self._f[index_x]*(x1_n2 - 1.))/(x1_n + x1_n2)


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
