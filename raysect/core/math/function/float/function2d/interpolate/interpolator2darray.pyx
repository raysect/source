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

import numpy as np
cimport cython
from raysect.core.math.cython.utility cimport find_index
from raysect.core.math.cython.interpolation.linear cimport linear2d
from raysect.core.math.cython.interpolation.cubic cimport calc_coefficients_2d, evaluate_cubic_2d


cdef double[4] FACTORIAL_ARRAY
FACTORIAL_ARRAY[0] = 1.
FACTORIAL_ARRAY[1] = 1.
FACTORIAL_ARRAY[2] = 2.
FACTORIAL_ARRAY[3] = 6.

@cython.cdivision(True)
cdef double rescale_lower_normalisation(double dfdn, double x_lower, double x, double x_upper):
    """
    Derivatives that are normalised to the unit square (x_upper - x) = 1 are un-normalised, then re-normalised to
    (x - x_lower)
    
    Must be spaced so that x and x_upper are not the same values. Monotonically increasing checks should already provide
    this.
    """
    return dfdn * (x - x_lower)/(x_upper - x)


cdef int find_index_change(int index, int last_index):
    """
    Transforming the output of find_index to find the index lower index of a cell required for an extrapolator.

    Finding the left most index of a grid cell from the output of find_index. The output of find_index is -1 at the 
    lower index, which has a lower border at index 0, and index = last_index at the upper border which is changed to 
    index = last_index - 1.

    :param int index: the index of the lower side of a unit cell.
    :param int last_index: the index of the final point.
    :return: the index of the lower cell at the border of the interpolator spline knots.
    """

    cdef int lower_index
    if index == -1:
        lower_index = 0

    elif index == last_index:
        lower_index = last_index - 1

    else:
        lower_index = index
    return lower_index


cdef int find_edge_index(int index, int last_index):
    """
    Transforming the output of find_index to find the index of the array border required for an extrapolator.

    Instead of finding the left most index of a grid cell, the index of the border is found from the output of 
    find_index. The output of find_index is -1 at the lower index, which has a border at index 0, and index = 
    last_index at the upper border. The difference from extrapolator_index_change is at the upper border.

    :param int index: the index of the lower side of a unit cell.
    :param int last_index: the index of the final point.
    :return: the index of the border of the interpolator spline knots.
    """

    cdef int edge_index
    if index == -1:
        edge_index = 0

    elif index == last_index:
        edge_index = last_index

    else:
        edge_index = index

    return edge_index


cdef class Interpolator2DArray(Function2D):
    """
    A configurable interpolator for 2D arrays.

    Coordinate array (x), array (y) and data array (f) are sorted and transformed into Numpy arrays.
    The resulting Numpy arrays are stored as read only. I.e. `writeable` flag of self.x, self.y and self.f
    is set to False. Alteration of the flag may result in unwanted behaviour.

    :param object x: 1D array-like object of real values storing the x spline knot positions.
    :param object y: 1D array-like object of real values storing the y spline knot positions.
    :param object f: 2D array-like object of real values storing the spline knot function value at x, y.
    :param str interpolation_type: Type of interpolation to use. Options are:
        `linear`: Interpolates the data using piecewise bilinear interpolation.
        `cubic`: Interpolates the data using piecewise bicubic interpolation.
    :param str extrapolation_type: Type of extrapolation to use. Options are:
        `none`: Attempt to access data outside of x's and y's range will yield ValueError.
        `nearest`: Extrapolation results is the nearest position x and y value in the interpolation domain.
        `linear`: Extrapolate bilinearly the interpolation function.
    :param double extrapolation_range_x: Limits the range where extrapolation is permitted. Requesting data beyond the
        extrapolation range results in ValueError. Extrapolation range will be applied as padding symmetrically to both
        ends of the interpolation range (x).
    :param double extrapolation_range_y: Limits the range where extrapolation is permitted. Requesting data beyond the
        extrapolation range results in ValueError. Extrapolation range will be applied as padding symmetrically to both
        ends of the interpolation range (y).

    .. code-block:: python

        >>> from raysect.core.math.function.float.function2d.interpolate.interpolator2darray import Interpolator2DArray
        >>>
        >>> x = np.linspace(-1., 1., 20)
        >>> y = np.linspace(-1., 1., 20)
        >>> x_array, y_array = np.meshgrid(x, y)
        >>> f = np.exp(-(x_array**2 + y_array**2))
        >>> interpolator2D = Interpolator2DArray(x, y, f, 'cubic', 'nearest', 1.0, 1.0)
        >>> # Interpolation
        >>> interpolator2D(1.0, 0.2)
        0.35345307120078995
        >>> # Extrapolation
        >>> interpolator2D(1.0, 1.1)
        0.1353352832366128
        >>> # Extrapolation out of bounds
        >>> interpolator2D(1.0, 2.1)
        ValueError: The specified value (y=2.1) is outside of extrapolation range.

    :note: All input derivatives used in calculations use the previous and next indices in the spline knot arrays.
        At the edge of the spline knot arrays the index of the edge of the array is is used instead.
    :note: x, y arrays must be equal in shape to f in the first and second dimension respectively.
    :note: x and y must be monotonically increasing arrays.

    """

    def __init__(self, object x, object y, object f, str interpolation_type, str extrapolation_type,
                 double extrapolation_range_x, double extrapolation_range_y):

        x = np.array(x, dtype=np.float64, order='c')
        y = np.array(y, dtype=np.float64, order='c')
        f = np.array(f, dtype=np.float64, order='c')

        # extrapolation_ranges must be greater than or equal to 0.
        if extrapolation_range_x < 0:
            raise ValueError('extrapolation_range_x must be greater than or equal to 0.')

        if extrapolation_range_y < 0:
            raise ValueError('extrapolation_range_y must be greater than or equal to 0.')

        # Dimensions checks.
        if x.ndim != 1:
            raise ValueError(f'The x array must be 1D. Got {x.shape}.')

        if y.ndim != 1:
            raise ValueError(f'The y array must be 1D. Got {y.shape}.')

        if f.ndim != 2:
            raise ValueError(f'The f array must be 2D. Got {f.shape}.')

        if x.shape[0] != f.shape[0]:
            raise ValueError(f'Shape mismatch between x array ({x.shape}) and f array ({f.shape}).')

        if y.shape[0] != f.shape[1]:
            raise ValueError(f'Shape mismatch between y array ({y.shape}) and f array ({f.shape}).')

        # Test monotonicity.
        if (np.diff(x) <= 0).any():
            raise ValueError('The x array must be monotonically increasing.')
        if (np.diff(y) <= 0).any():
            raise ValueError('The y array must be monotonically increasing.')

        self.x = x
        self.y = y
        self._x_mv = x
        self._y_mv = y
        self._f_mv = f
        x.flags.writeable = False
        y.flags.writeable = False
        f.flags.writeable = False

        self._last_index_x = self.x.shape[0] - 1
        self._last_index_y = self.y.shape[0] - 1
        self._extrapolation_range_x = extrapolation_range_x
        self._extrapolation_range_y = extrapolation_range_y

        # Check the requested interpolation type exists.
        interpolation_type = interpolation_type.lower()
        if interpolation_type not in id_to_interpolator:
            raise ValueError(f'Interpolation type {interpolation_type} not found. Options are {id_to_interpolator.keys()}.')

        # Check the requested extrapolation type exists.
        extrapolation_type = extrapolation_type.lower()
        if extrapolation_type not in id_to_extrapolator:
            raise ValueError(f'Extrapolation type {extrapolation_type} not found. Options are {id_to_extrapolator.keys()}.')

        # Permit combinations of interpolator and extrapolator where the order of extrapolator is higher than interpolator.
        if extrapolation_type not in permitted_interpolation_combinations[interpolation_type]:
            raise ValueError(f'Extrapolation type {extrapolation_type} not compatible with interpolation type {interpolation_type}.')

        # Create the interpolator and extrapolator objects.
        self._interpolator = id_to_interpolator[interpolation_type](self._x_mv, self._y_mv, self._f_mv)

        self._extrapolator = id_to_extrapolator[extrapolation_type](
            self._x_mv, self._y_mv, self._f_mv, self._interpolator, extrapolation_range_x, extrapolation_range_y
        )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double px, double py) except? -1e999:
        """
        Evaluates the interpolating function.
        
        Passes the evaluation to the _Interpolator2D object if within the bounds of the spline knots in both the x and 
        the y direction. If outside the bounds in x or y, within the _Extrapolator2D object an extrapolation method is 
        called depending on whether the requested point is out of bounds of the spline knots in the x, y or xy 
        direction. Because the return value of find_index returns the lower index of the first or last unit cell, 
        extrapolation at the upper or lower index requires the bordering index to evaluate.

        :param double px: the point for which an interpolated value is required.
        :param double py: the point for which an interpolated value is required.
        :return: the interpolated value at point x, y.
        """

        # Find index assuming the grid is the same in x and y
        cdef int index_x = find_index(self._x_mv, px)
        cdef int index_y = find_index(self._y_mv, py)
        cdef int index_lower_x = find_index_change(index_x, self._last_index_x)
        cdef int index_lower_y = find_index_change(index_y, self._last_index_y)
        cdef bint outside_domain_x = index_x == -1 or (index_x == self._last_index_x and px != self._x_mv[self._last_index_x])
        cdef bint outside_domain_y = index_y == -1 or (index_y == self._last_index_y and py != self._y_mv[self._last_index_y])

        if outside_domain_x or outside_domain_y:
            return self._extrapolator.evaluate(px, py, index_x, index_y)

        else:
            return self._interpolator.evaluate(px, py, index_lower_x, index_lower_y)

    @property
    def domain(self):
        """
        Returns min/max interval of 'x' and 'y' arrays.
        Order: min(x), max(x), min(y), max(y).
        """
        return self._x_mv[0], self._x_mv[self._last_index_x], self._y_mv[0], self._y_mv[self._last_index_y]


cdef class _Interpolator2D:
    """
    Base class for 2D interpolators.

    :param x: 1D memory view of the spline point x positions.
    :param y: 1D memory view of the spline point y positions.
    :param f: 2D memory view of the function value at spline point x, y positions.
    """

    ID = None

    def __init__(self, double[::1] x, double[::1] y, double[:, ::1] f):
        self._x = x
        self._y = y
        self._f = f
        self._last_index_x = self._x.shape[0] - 1
        self._last_index_y = self._y.shape[0] - 1

    cdef double evaluate(self, double px, double py, int index_x, int index_y) except? -1e999:
        """
        Calculates interpolated value at a requested point.

        :param double px: the point for which an interpolated value is required.
        :param double py: the point for which an interpolated value is required.
        :param int index_x: the lower index of the bin containing point px. (Result of bisection search).   
        :param int index_y: the lower index of the bin containing point py. (Result of bisection search).   
        """
        raise NotImplementedError('_Interpolator is an abstract base class.')

    cdef double analytic_gradient(self, double px, double py, int index_x, int index_y, int order_x, int order_y):
        """
        Calculates the interpolator's derivative of a valid order at a requested point.

        :param double px: the x position of the point for which an interpolated value is required.
        :param double py: the y position of the point for which an interpolated value is required.
        :param int index_x: the lower index of the bin containing point px. (Result of bisection search).
        :param int index_y: the lower index of the bin containing point py. (Result of bisection search).  
        """

        raise NotImplementedError('_Interpolator is an abstract base class.')


cdef class _Interpolator2DLinear(_Interpolator2D):
    """
    Linear interpolation of 2D function.

    :param x: 1D memory view of the spline point x positions.
    :param y: 1D memory view of the spline point y positions.
    :param f: 2D memory view of the function value at spline point x, y positions.
    """

    ID = 'linear'

    def __init__(self, double[::1] x, double[::1] y, double[:, ::1] f):
        super().__init__(x, y, f)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double px, double py, int index_x, int index_y) except? -1e999:
        return linear2d(
            self._x[index_x], self._x[index_x + 1], self._y[index_y], self._y[index_y + 1],
            self._f[index_x:index_x + 2, index_y:index_y + 2], px, py
        )

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double analytic_gradient(self, double px, double py, int index_x, int index_y, int order_x, int order_y):
        """
        Calculate the normalised derivative of specified order in a unit square.
        
        The order of the derivative corresponds to order_x and order_y as the number of times differentiated. For 
        example order_x = 1 and order_y = 1 is d2f/dxdy. The normalised gradient is calculated of the bilinear 
        function f(x, y) = a0 + a1x + a2y + a3xy, which is the product of 2 linear functions. The derivatives are 
        therefore df/dx = a1 + a3y ; df/dy = a1 + a3x ; d2f/dxdy = a3. The derivatives are calculated on the normalised 
        unit square.

        :param double px: the point for which an interpolated value is required.
        :param double py: the point for which an interpolated value is required.
        :param int index_x: the lower index of the bin containing point px. (Result of bisection search).   
        :param int index_y: the lower index of the bin containing point py. (Result of bisection search).   
        :param int order_x: the derivative order in the x direction.
        :param int order_y: the derivative order in the y direction.
        """
        cdef double df_dn
        #TODO make this neater
        if order_x == 1 and order_y == 1:
            df_dn = self._calculate_coefficients(index_x, index_y, coefficient_index=3)

        elif order_x == 1:
            df_dn = self._calculate_coefficients(index_x, index_y, coefficient_index=1) + self._calculate_coefficients(index_x, index_y, coefficient_index=3) * (py - self._y[index_y]) / (self._y[index_y + 1] - self._y[index_y])

        elif order_y == 1:
            df_dn = self._calculate_coefficients(index_x, index_y, coefficient_index=2) + self._calculate_coefficients(index_x, index_y, coefficient_index=3) * (px - self._x[index_x]) / (self._x[index_x + 1] - self._x[index_x])

        else:
            raise ValueError('The derivative order for x and y (order_x and order_y) must be a combination of 1 and 0 for the linear interpolator (but 0, 0 should be handled by evaluating the interpolator).')

        return df_dn

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _calculate_coefficients(self, int index_x, int index_y, int coefficient_index):
        """
        Calculate the bilinear coefficients in a unit square.

        The bilinear function (which is the product of 2 linear functions) f(x, y) = a0 + a1x + a2y + a3xy. Coefficients 
        a0, a1, a2, a3 are calculated for one unit square. The coefficients are calculated from inverting the equation
        Xa = fv.
        Where:
        X = [[1, x1, y1, x1y1],         a = [a0,         fv = [f(0, 0),
            [1, x1, y2, x1y2],               a1,               f(0, 1),
            [1, x2, y1, x2y1],               a2,               f(1, 0),
            [1, x2, y2, x2y2]]               a3]               f(1, 1)]
        This simplifies where x1, y1 = 0, x2, y2 = 1 for the unit square to find a = X^{-1} fv
        where:
        a[0] = f[0][0]
        a[1] = f[1][0] - f[0][0]
        a[2] = f[0][1] - f[0][0]
        a[3] = f[0][0] - f[0][1] - f[1][0] + f[1][1]

        :param int index_x: the lower index of the bin containing point px. (Result of bisection search).   
        :param int index_y: the lower index of the bin containing point py. (Result of bisection search). 
        :param int coefficient_index: Which coefficient of the bilinear equation to return a0, a1, a2, a3.
        """

        # Calculate the coefficients of the requested spline point.
        if coefficient_index == 0:
            return self._f[index_x, index_y]

        elif coefficient_index == 1:
            return self._f[index_x + 1, index_y] - self._f[index_x, index_y]

        elif coefficient_index == 2:
            return self._f[index_x, index_y + 1] - self._f[index_x, index_y]

        elif coefficient_index == 3:
            return self._f[index_x, index_y] - self._f[index_x, index_y + 1] - self._f[index_x + 1, index_y] + self._f[index_x + 1, index_y + 1]

        else:
            raise ValueError(f'There are only 4 bilinear coefficients, the index requested:{coefficient_index} is out of range.')

cdef class _Interpolator2DCubic(_Interpolator2D):
    """
    Cubic interpolation of a 2D function.

    When called, stores cubic polynomial coefficients from the value of the function, df/dx, df/dy  and d2f/dxdy at the
    neighbouring spline knots using _ArrayDerivative2D object. The polynomial coefficients and gradients are calculated
    between each spline knots in a unit square.

    :param x: 1D memory view of the spline point x positions.
    :param y: 1D memory view of the spline point y positions.
    :param f: 2D memory view of the function value at spline point x, y positions.
    """

    ID = 'cubic'

    def __init__(self, double[::1] x, double[::1] y, double[:, ::1] f):
        super().__init__(x, y, f)

        # Where 'a' has been calculated the mask value = 1.
        self._mask_a = np.zeros((self._last_index_x, self._last_index_y), dtype=np.float64)

        # Store the cubic spline coefficients, where increasing index values are the coefficients for the coefficients of higher powers of x, y in the last 2 dimensions.
        self._a = np.zeros((self._last_index_x, self._last_index_y, 4, 4), dtype=np.float64)
        self._a_mv = self._a

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double px, double py, int index_x, int index_y) except? -1e999:

        # rescale x between 0 and 1
        cdef double x_scal
        cdef double y_scal
        cdef double x_bound, y_bound
        cdef double[4][4] a

        x_bound = self._x[index_x + 1] - self._x[index_x]
        x_scal = (px - self._x[index_x]) / x_bound

        y_bound = self._y[index_y + 1] - self._y[index_y]
        y_scal = (py - self._y[index_y]) / y_bound

        # Calculate the coefficients (and gradients at each spline point) if they dont exist.
        self._cache_coefficients(index_x, index_y, a)

        return evaluate_cubic_2d(a, x_scal, y_scal)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef _cache_coefficients(self, int index_x, int index_y, double[4][4] a):
        """
        Calculates and stores, or loads previously stored cubic coefficients.
        
        :param int index_x: the lower index of the bin containing point px. (Result of bisection search).   
        :param int index_y: the lower index of the bin containing point py. (Result of bisection search). 
        :param double[4][4] a: The coefficients of the bicubic equation.
        """
        cdef double[2][2] f, dfdx, dfdy, d2fdxdy
        cdef int i, j
        cdef _ArrayDerivative2D array_derivative

        # Calculate the coefficients (and gradients at each spline point) if they dont exist
        if not self._mask_a[index_x, index_y]:
            f[0][0] = self._f[index_x, index_y]
            f[1][0] = self._f[index_x + 1, index_y]
            f[0][1] = self._f[index_x, index_y + 1]
            f[1][1] = self._f[index_x + 1, index_y + 1]

            array_derivative = _ArrayDerivative2D(self._x, self._y, self._f)
            dfdx[0][0] = array_derivative.evaluate(index_x, index_y, 1, 0, False, False)
            dfdx[0][1] = array_derivative.evaluate(index_x, index_y + 1, 1, 0, False, True)
            dfdx[1][0] = array_derivative.evaluate(index_x + 1, index_y, 1, 0, True, False)
            dfdx[1][1] = array_derivative.evaluate(index_x + 1, index_y + 1, 1, 0, True, True)

            dfdy[0][0] = array_derivative.evaluate(index_x, index_y, 0, 1, False, False)
            dfdy[0][1] = array_derivative.evaluate(index_x, index_y + 1, 0, 1, False, True)
            dfdy[1][0] = array_derivative.evaluate(index_x + 1, index_y, 0, 1, True, False)
            dfdy[1][1] = array_derivative.evaluate(index_x + 1, index_y + 1, 0, 1, True, True)

            d2fdxdy[0][0] = array_derivative.evaluate(index_x, index_y, 1, 1, False, False)
            d2fdxdy[0][1] = array_derivative.evaluate(index_x, index_y + 1, 1, 1, False, True)
            d2fdxdy[1][0] = array_derivative.evaluate(index_x + 1, index_y, 1, 1, True, False)
            d2fdxdy[1][1] = array_derivative.evaluate(index_x + 1, index_y + 1, 1, 1, True, True)

            calc_coefficients_2d(f, dfdx, dfdy, d2fdxdy, a)
            for i in range(4):
                for j in range(4):
                    self._a[index_x, index_y, i, j] = a[i][j]
            self._mask_a[index_x, index_y] = 1

        else:
            for i in range(4):
                for j in range(4):
                    a[i][j] = self._a[index_x, index_y, i, j]

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double analytic_gradient(self, double px, double py, int index_x, int index_y, int order_x, int order_y):
        """
        Calculate the normalised gradient of specified order in a unit square.

        The order of the derivative corresponds to order_x and order_y as the number of times differentiated. For 
        example order_x = 1 and order_y = 1 is d2f/dxdy. The normalised gradient is calculated for the bicubic by 
        generalising each orders derivative coefficient to n!/(n-order)! . e.g. for n = [1, 2, 3], for order 1
        these are [1, 2, 3] for the derivative dfdx = a1 + 2*a2*x + 3*a3*x^2, order 2 has [2, 6] for 
        d2fdx2 = 2*a2 + 6*a3*x. These combine in x and y by selecting elements of the matrix 'a' and the x^n and y^n 
        that remains after differentiation.
        
        :param double px: the point for which an interpolated value is required.
        :param double py: the point for which an interpolated value is required.
        :param int index_x: the lower index of the bin containing point px. (Result of bisection search).   
        :param int index_y: the lower index of the bin containing point py. (Result of bisection search).   
        :param int order_x: the derivative order in the x direction.
        :param int order_y: the derivative order in the y direction.
        """

        # rescale x between 0 and 1
        cdef double x_scal
        cdef double y_scal
        cdef double x_bound, y_bound
        cdef double[4][4] a
        cdef double[4] x_powers, y_powers
        cdef double df_dn = 0.
        cdef int i, j

        if order_x > 3:
            raise ValueError('Can\'t get a gradient of order 4 or more in cubic.')

        x_bound = self._x[index_x + 1] - self._x[index_x]
        x_scal = (px - self._x[index_x]) / x_bound

        y_bound = self._y[index_y + 1] - self._y[index_y]
        y_scal = (py - self._y[index_y]) / y_bound

        # Calculate the coefficients (and gradients at each spline point) if they dont exist
        self._cache_coefficients(index_x, index_y, a)
        x_powers[0] = 1
        x_powers[1] = x_scal
        x_powers[2] = x_scal * x_scal
        x_powers[3] = x_scal * x_scal * x_scal
        y_powers[0] = 1
        y_powers[1] = y_scal
        y_powers[2] = y_scal * y_scal
        y_powers[3] = y_scal * y_scal * y_scal

        for i in range(order_x, 4):
            for j in range(order_y, 4):
                df_dn += (a[i][j] * (FACTORIAL_ARRAY[i]/FACTORIAL_ARRAY[i-order_x]) * (FACTORIAL_ARRAY[j]/FACTORIAL_ARRAY[j-order_y]) *
                          x_powers[i-order_x] * y_powers[j-order_y])
        return  df_dn


cdef class _Extrapolator2D:
    """
    Base class for Function2D extrapolators.

    :param x: 1D memory view of the spline point x positions.
    :param y: 1D memory view of the spline point y positions.
    :param f: 2D memory view of the function value at spline point x, y positions.
    :param interpolator: stored _Interpolator2D object that is being used.
    """

    ID = NotImplemented

    def __init__(self, double[::1] x, double[::1] y, double[:, ::1] f, _Interpolator2D interpolator, double extrapolation_range_x, double extrapolation_range_y):
        self._x = x
        self._y = y
        self._f = f
        self._last_index_x = self._x.shape[0] - 1
        self._last_index_y = self._y.shape[0] - 1
        self._interpolator = interpolator
        self._extrapolation_range_x = extrapolation_range_x
        self._extrapolation_range_y = extrapolation_range_y

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double px, double py, int index_x, int index_y) except? -1e999:

        cdef int index_lower_x = find_index_change(index_x, self._last_index_x)
        cdef int index_lower_y = find_index_change(index_y, self._last_index_y)
        cdef int edge_x_index = find_edge_index(index_x, self._last_index_x)
        cdef int edge_y_index = find_edge_index(index_y, self._last_index_y)
        if (index_x == -1 or index_x == self._last_index_x) and (index_y == -1 or index_y == self._last_index_y):

            if np.abs(px - self._x[edge_x_index]) > self._extrapolation_range_x:
                raise ValueError(f'The specified value (x={px}) is outside of extrapolation range.')

            if np.abs(py - self._y[edge_y_index]) > self._extrapolation_range_y:
                raise ValueError(f'The specified value (y={py}) is outside of extrapolation range.')

            return self._evaluate_edge_xy(px, py, index_lower_x, index_lower_y, edge_x_index, edge_y_index)

        elif index_x == -1 or index_x == self._last_index_x:

            if np.abs(px - self._x[edge_x_index]) > self._extrapolation_range_x:
                raise ValueError(f'The specified value (x={px}) is outside of extrapolation range.')

            return self._evaluate_edge_x(px, py, index_lower_x, index_lower_y, edge_x_index)

        elif index_y == -1 or index_y == self._last_index_y:

            if np.abs(py - self._y[edge_y_index]) > self._extrapolation_range_y:
                raise ValueError(f'The specified value (y={py}) is outside of extrapolation range.')

            return self._evaluate_edge_y(px, py, index_lower_x, index_lower_y, edge_y_index)

        else:
            raise ValueError('Interpolated index parsed to extrapolator.')

    cdef double _evaluate_edge_x(self, double px, double py, int index_x, int index_y, int edge_x_index) except? -1e999:
        raise NotImplementedError(f'{self.__class__} not implemented.')

    cdef double _evaluate_edge_y(self, double px, double py, int index_x, int index_y, int edge_y_index) except? -1e999:
        raise NotImplementedError(f'{self.__class__} not implemented.')

    cdef double _evaluate_edge_xy(self, double px, double py, int index_x, int index_y, int edge_x_index, int edge_y_index) except? -1e999:
        raise NotImplementedError(f'{self.__class__} not implemented.')


cdef class _Extrapolator2DNone(_Extrapolator2D):
    """
    Extrapolator that does nothing.

    :param x: 1D memory view of the spline point x positions.
    :param y: 1D memory view of the spline point y positions.
    :param f: 2D memory view of the function value at spline point x, y positions.
    :param interpolator: stored _Interpolator2D object that is being used.
    """

    ID = 'none'

    def __init__(self, double[::1] x, double[::1] y, double[:, ::1] f, _Interpolator2D interpolator, double extrapolation_range_x, double extrapolation_range_y):
           super().__init__(x, y, f, interpolator, extrapolation_range_x, extrapolation_range_y)

    cdef double _evaluate_edge_x(self, double px, double py, int index_x, int index_y, int edge_x_index) except? -1e999:
        raise ValueError(f'Extrapolation not available. Interpolate within function range x {np.min(self._x)}-{np.max(self._x)}.')

    cdef double _evaluate_edge_y(self, double px, double py, int index_x, int index_y, int edge_y_index) except? -1e999:
        raise ValueError(f'Extrapolation not available. Interpolate within function range y {np.min(self._y)}-{np.max(self._y)}.')

    cdef double _evaluate_edge_xy(self, double px, double py, int index_x, int index_y, int edge_x_index, int edge_y_index) except? -1e999:
        raise ValueError(f'Extrapolation not available. Interpolate within function range x {np.min(self._x)}-{np.max(self._x)} and y {np.min(self._y)}-{np.max(self._y)}.')


cdef class _Extrapolator2DNearest(_Extrapolator2D):
    """
    Extrapolator that returns nearest input value.

    :param x: 1D memory view of the spline point x positions.
    :param y: 1D memory view of the spline point y positions.
    :param f: 2D memory view of the function value at spline point x, y positions.
    :param interpolator: stored _Interpolator2D object that is being used.
    """

    ID = 'nearest'

    def __init__(self, double[::1] x, double[::1] y, double[:, ::1] f, _Interpolator2D interpolator, double extrapolation_range_x, double extrapolation_range_y):
           super().__init__(x, y, f, interpolator, extrapolation_range_x, extrapolation_range_y)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate_edge_x(self, double px, double py, int index_x, int index_y, int edge_x_index) except? -1e999:
        """
        Extrapolate beyond the spline knot domain in the y direction, but within the spline knot domain in the x 
        direction to find the nearest neighbour.
        
        :param double px: the point for which an interpolated value is required.
        :param double py: the point for which an interpolated value is required.
        :param int index_x: the lower index of the bin containing point px. (Result of bisection search).   
        :param int index_y: the lower index of the bin containing point py. (Result of bisection search).   
        :param int edge_x_index: the index of the closest edge spline knot in the x direction.
        """
        return self._interpolator.evaluate(self._x[edge_x_index], py, index_x, index_y)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate_edge_y(self, double px, double py, int index_x, int index_y, int edge_y_index) except? -1e999:
        """
        Extrapolate beyond the spline knot domain in the x direction, but within the spline knot domain in the y 
        direction to find the nearest neighbour.

        :param double px: the point for which an interpolated value is required.
        :param double py: the point for which an interpolated value is required.
        :param int index_x: the lower index of the bin containing point px. (Result of bisection search).   
        :param int index_y: the lower index of the bin containing point py. (Result of bisection search).   
        :param int edge_y_index: the index of the closest edge spline knot in the y direction.
        """
        return self._interpolator.evaluate(px, self._y[edge_y_index], index_x, index_y)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate_edge_xy(self, double px, double py, int index_x, int index_y, int edge_x_index, int edge_y_index) except? -1e999:
        """
        Extrapolate beyond the spline knot domain in the x and y directions to find the nearest neighbour.

        :param double px: the point for which an interpolated value is required.
        :param double py: the point for which an interpolated value is required.
        :param int index_x: the lower index of the bin containing point px. (Result of bisection search).   
        :param int index_y: the lower index of the bin containing point py. (Result of bisection search).
        :param int edge_x_index: the index of the closest edge spline knot in the x direction.
        :param int edge_y_index: the index of the closest edge spline knot in the y direction.
        """
        return self._interpolator.evaluate(self._x[edge_x_index], self._y[edge_y_index], index_x, index_y)


cdef class _Extrapolator2DLinear(_Extrapolator2D):
    """
    Extrapolator that returns linearly extrapolated input value.

    :param x: 1D memory view of the spline point x positions.
    :param y: 1D memory view of the spline point y positions.
    :param f: 2D memory view of the function value at spline point x, y positions.
    :param interpolator: stored _Interpolator2D object that is being used.
    """

    ID = 'linear'

    def __init__(self, double[::1] x, double[::1] y, double[:, ::1] f, _Interpolator2D interpolator, double extrapolation_range_x, double extrapolation_range_y):
           super().__init__(x, y, f, interpolator, extrapolation_range_x, extrapolation_range_y)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate_edge_x(self, double px, double py, int index_x, int index_y, int edge_x_index) except? -1e999:
        """
        Extrapolate beyond the spline knot domain in the x direction, but within the spline knot domain in the y 
        direction.
        
        The extrapolated value uses the closest value of the function at the edge (at the point py on the edge in x)
        and df/dx of the interpolator at that same point to extrapolate as f_extrap = f(edge) + Dx * df(edge)/dx where 
        Dx = px - edge_x, and df(edge)/dx has to be unnormalised. In this scenario Dy = 0, so other terms in the 
        bilinear equation are 0.
        
        :param double px: the point for which an interpolated value is required.
        :param double py: the point for which an interpolated value is required.
        :param int index_x: the lower index of the bin containing point px. (Result of bisection search).   
        :param int index_y: the lower index of the bin containing point py. (Result of bisection search).   
        :param int edge_x_index: the index of the closest edge spline knot in the x direction.
        """
        cdef double f, df_dx

        f = self._interpolator.evaluate(self._x[edge_x_index], py, index_x, index_y)
        df_dx = self._interpolator.analytic_gradient(self._x[edge_x_index], py, index_x, index_y, 1, 0) / \
                (self._x[index_x + 1] - self._x[index_x])
        return f + df_dx*(px - self._x[edge_x_index])

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate_edge_y(self, double px, double py, int index_x, int index_y, int edge_y_index) except? -1e999:
        """
        Extrapolate beyond the spline knot domain in the y direction, but within the spline knot domain in the x 
        direction.

        The extrapolated value uses the closest value of the function at the edge (at the point px on the edge in y)
        and df/dy of the interpolator at that same point to extrapolate as f_extrap = f(edge) + Dy * df(edge)/dy where 
        Dy = py - edge_y, and df(edge)/dy has to be unnormalised. In this scenario Dx = 0, so other terms in the 
        bilinear equation are 0.

        :param double px: the point for which an interpolated value is required.
        :param double py: the point for which an interpolated value is required.
        :param int index_x: the lower index of the bin containing point px. (Result of bisection search).   
        :param int index_y: the lower index of the bin containing point py. (Result of bisection search).   
        :param int edge_y_index: the index of the closest edge spline knot in the y direction.
        """
        cdef double f, df_dy

        f = self._interpolator.evaluate(px, self._y[edge_y_index], index_x, index_y)
        df_dy = self._interpolator.analytic_gradient(px, self._y[edge_y_index], index_x, index_y, 0, 1) / \
                   (self._y[index_y + 1] - self._y[index_y])
        return f + df_dy * (py - self._y[edge_y_index])

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate_edge_xy(self, double px, double py, int index_x, int index_y, int edge_x_index, int edge_y_index) except? -1e999:
        """
        Extrapolate beyond the spline knot domain in the x and y direction.

        The extrapolated value uses the closest value of the function at the edge in x and y, df/dx, df/dy and d2f/dxdy  
        of the interpolator at the edge of the spline knot grid to extrapolate as 
        f_extrap = f(edge) + Dx*df(edge)/dx + Dy*df(edge)/dy + Dx*Dy*d2f(edge)/dxdy where 
        Dx = px - edge_x, Dy = py - edge_y. This is because the bilinear equation f(x, y) as a taylor expansion only 
        has terms with x and y to a maximum power of 1 in every term. All 3 derivatives are  
        un-normalised before extrapolation.
        
        :param double px: the point for which an interpolated value is required.
        :param double py: the point for which an interpolated value is required.
        :param int index_x: the lower index of the bin containing point px. (Result of bisection search).   
        :param int index_y: the lower index of the bin containing point py. (Result of bisection search).   
        :param int edge_x_index: the index of the closest edge spline knot in the x direction.
        :param int edge_y_index: the index of the closest edge spline knot in the y direction.

        """
        cdef double f, df_dx, df_dy, d2f_dxdy

        f = self._interpolator.evaluate(self._x[edge_x_index], self._y[edge_y_index], index_x, index_y)

        df_dx = self._interpolator.analytic_gradient(
            self._x[edge_x_index], self._y[edge_y_index], index_x, index_y, 1, 0) / (self._x[index_x + 1] - self._x[index_x])

        df_dy = self._interpolator.analytic_gradient(
            self._x[edge_x_index], self._y[edge_y_index], index_x, index_y, 0, 1) / (self._y[index_y + 1] - self._y[index_y])

        d2f_dxdy = self._interpolator.analytic_gradient(
            self._x[edge_x_index], self._y[edge_y_index], index_x, index_y, 1, 1) / ((self._x[index_x + 1] - self._x[index_x]) * (self._y[index_y + 1] - self._y[index_y]))

        return f + df_dx * (px - self._x[edge_x_index]) + df_dy * (py - self._y[edge_y_index]) + d2f_dxdy * (py - self._y[edge_y_index])* (px - self._x[edge_x_index])


cdef class _ArrayDerivative2D:
    """
    Gradient method that returns the approximate derivative of a desired order at a specified grid point.

    These methods of finding derivatives are only valid on a 2D grid of points, at the values at the points. Other
    derivative methods would be dependent on the interpolator types.

    :param x: 1D memory view of the spline point x positions.
    :param y: 1D memory view of the spline point y positions.
    :param f: 2D memory view of the function value at spline point x, y positions.
    """
    def __init__(self, double[::1] x, double[::1] y, double[:, ::1] f):

        self._x = x
        self._y = y
        self._f = f
        self._last_index_x = self._x.shape[0] - 1
        self._last_index_y = self._y.shape[0] - 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, int index_x, int index_y, int derivative_order_x, int derivative_order_y, bint rescale_norm_x, bint rescale_norm_y) except? -1e999:
        """
        Evaluate the derivative of specific order at a grid point.
        
        The grid of spline knots is reduced to a 2X2 to 3X3 grid for gradient evaluation depending on if the requested
        derivative is near the edge or not. If near the edge in 1 dimension, grid size is 2X3 or 3X2.

        :param index_x: The lower index of the x grid cell to evaluate.
        :param index_y: The lower index of the y grid cell to evaluate.
        :param derivative_order_x: An integer of the derivative order x. Only zero if derivative_order_y is nonzero.
        :param derivative_order_y: An integer of the derivative order y. Only zero if derivative_order_x is nonzero.
        :param rescale_norm_x: A boolean as whether to rescale to the delta before x[index_x] or after (default).
        :param rescale_norm_y: A boolean as whether to rescale to the delta before y[index_y] or after (default).
        """
        # Find if at the edge of the grid, and in what direction. Then evaluate the gradient.
        cdef double dfdn = 0.

        if index_x == 0:

            if index_y == 0:
                dfdn = self._eval_edge_xy(index_x, index_y, derivative_order_x, derivative_order_y, x_centre_add=0, y_centre_add=0)

            elif index_y == self._last_index_y:
                dfdn = self._eval_edge_xy(index_x, index_y - 1, derivative_order_x, derivative_order_y, x_centre_add=0, y_centre_add=1)

            else:
                dfdn = self._eval_edge_x(index_x, index_y, derivative_order_x, derivative_order_y, x_centre_add=0, y_centre_add=0)

        elif index_x == self._last_index_x:

            if index_y == 0:
                dfdn = self._eval_edge_xy(index_x - 1, index_y, derivative_order_x, derivative_order_y, x_centre_add=1, y_centre_add=0)

            elif index_y == self._last_index_y:
                dfdn = self._eval_edge_xy(index_x - 1, index_y - 1, derivative_order_x, derivative_order_y, x_centre_add=1, y_centre_add=1)

            else:
                dfdn = self._eval_edge_x(index_x - 1, index_y, derivative_order_x, derivative_order_y, x_centre_add=1, y_centre_add=0)

        else:

            if index_y == 0:
                dfdn = self._eval_edge_y(index_x, index_y, derivative_order_x, derivative_order_y, x_centre_add=0, y_centre_add=0)

            elif index_y == self._last_index_y:
                dfdn = self._eval_edge_y(index_x, index_y - 1, derivative_order_x, derivative_order_y, x_centre_add=0, y_centre_add=1)

            else:
                dfdn = self._eval_xy(index_x, index_y, derivative_order_x, derivative_order_y)

        if rescale_norm_x:

            if not (index_x == 0 or index_x == self._last_index_x):
                for i in range(derivative_order_x):
                    dfdn = rescale_lower_normalisation(dfdn,  self._x[index_x - 1], self._x[index_x], self._x[index_x + 1])

        if rescale_norm_y:

            if not (index_y == 0 or index_y == self._last_index_y):
                for i in range(derivative_order_y):
                    dfdn = rescale_lower_normalisation(dfdn,  self._y[index_y - 1], self._y[index_y], self._y[index_y + 1])

        return dfdn

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _eval_edge_x(self, int index_x, int index_y, int derivative_order_x, int derivative_order_y, int x_centre_add, int y_centre_add):
        cdef double dfdn = 0.
        cdef double[::1] x_range, y_range
        cdef double[:, ::1] f_range
        cdef int x_centre = 0, y_centre = 1

        x_range = self._x[index_x:index_x + 2]
        y_range = self._y[index_y - 1:index_y + 2]
        f_range = self._f[index_x:index_x + 2, index_y - 1:index_y + 2]

        if derivative_order_x == 1 and derivative_order_y == 0:
            dfdn = self._derivitive_dfdx_edge(f_range[:, y_centre + y_centre_add])

        elif derivative_order_x == 0 and derivative_order_y == 1:
            dfdn = self._derivitive_dfdx(y_range, f_range[x_centre + x_centre_add, :])

        elif derivative_order_x == 1 and derivative_order_y == 1:
            dfdn = self._derivitive_d2fdxdy_edge_x(y_range, f_range)

        else:
            raise ValueError('No higher order derivatives implemented.')

        return dfdn

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _eval_edge_y(self, int index_x, int index_y, int derivative_order_x, int derivative_order_y, int x_centre_add, int y_centre_add):
        cdef double dfdn = 0.
        cdef double[::1] x_range, y_range
        cdef double[:, ::1] f_range
        cdef int x_centre = 1, y_centre = 0

        x_range = self._x[index_x - 1:index_x + 2]
        y_range = self._y[index_y:index_y + 2]
        f_range = self._f[index_x - 1:index_x + 2, index_y:index_y + 2]

        if derivative_order_x == 1 and derivative_order_y == 0:
            dfdn = self._derivitive_dfdx(x_range, f_range[:, y_centre + y_centre_add])

        elif derivative_order_x == 0 and derivative_order_y == 1:
            dfdn = self._derivitive_dfdx_edge(f_range[x_centre + x_centre_add, :])

        elif derivative_order_x == 1 and derivative_order_y == 1:
            dfdn = self._derivitive_d2fdxdy_edge_y(x_range, f_range)

        else:
            raise ValueError('No higher order derivatives implemented.')

        return dfdn

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _eval_edge_xy(self, int index_x, int index_y, int derivative_order_x, int derivative_order_y, int x_centre_add, int y_centre_add) except? -1e999:
        cdef double dfdn = 0.
        cdef double[::1] x_range, y_range
        cdef double[:, ::1] f_range
        cdef int x_centre = 0, y_centre = 0

        x_range = self._x[index_x:index_x + 2]
        y_range = self._y[index_y:index_y + 2]
        f_range = self._f[index_x:index_x + 2, index_y:index_y + 2]

        if derivative_order_x == 1 and derivative_order_y == 0:
            dfdn = self._derivitive_dfdx_edge(f_range[:, y_centre + y_centre_add])

        elif derivative_order_x == 0 and derivative_order_y == 1:
            dfdn = self._derivitive_dfdx_edge(f_range[x_centre + x_centre_add, :])

        elif derivative_order_x == 1 and derivative_order_y == 1:
            dfdn = self._derivitive_d2fdxdy_edge_xy(f_range[x_centre:x_centre + 2, y_centre:y_centre + 2])

        else:
            raise ValueError('No higher order derivatives implemented.')

        return dfdn

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _eval_xy(self, int index_x, int index_y, int derivative_order_x, int derivative_order_y):
        cdef double dfdn = 0.
        cdef double[::1] x_range, y_range
        cdef double[:, ::1] f_range
        cdef int x_centre = 1, y_centre = 1

        x_range = self._x[index_x - 1:index_x + 2]
        y_range = self._y[index_y - 1:index_y + 2]
        f_range = self._f[index_x - 1:index_x + 2, index_y - 1:index_y + 2]

        if derivative_order_x == 1 and derivative_order_y == 0:
            dfdn = self._derivitive_dfdx(x_range, f_range[:, y_centre])

        elif derivative_order_x == 0 and derivative_order_y == 1:
            dfdn = self._derivitive_dfdx(y_range, f_range[x_centre, :])

        elif derivative_order_x == 1 and derivative_order_y == 1:
            dfdn = self._derivitive_d2fdxdy(
                x_range[x_centre - 1:x_centre + 2], y_range[y_centre - 1:y_centre + 2],
                f_range[x_centre - 1:x_centre + 2, y_centre - 1:y_centre + 2]
            )

        else:
            raise ValueError('No higher order derivatives implemented.')

        return dfdn

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivitive_dfdx_edge(self, double[:] f):
        """
        Calculate the 1st derivative on an unevenly spaced grid as a 1st order approximation.

        A taylor expansion of f(x, y) with changes in x only:
        f(x+dx0, y) = f(x, y) + dx0*fx(x, y) + O^2(dx0)
        Can simply be rearranged to fx(x, y) = [f(x+dx0, y) - f(x, y)]/dx0, and using the normalisation dx0 = 1
        simply fx(x, y) = [f(x+dx0, y) - f(x, y)] recovers the forward difference without need to account for uneven 
        grid spacing.
        The input x and f are 2 long, where [0] is the central value, [1] is the forward value (where x[1]-x[0] = 1 
        when normalised).
        """
        return f[1] - f[0]

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivitive_dfdx(self, double[:] x, double[:] f) except? -1e999:
        """
        Calculate the 1st derivative on an unevenly spaced grid as a 2nd order approximation.
        
        A taylor expansion of f(x, y) with changes in x only:
        f(x+dx0, y) = f(x, y) + dx0*fx(x, y) + dx0^2*fxx(x, y)/2 + O^3(dx0)
        f(x-dx1, y) = f(x, y) - dx1*fx(x, y) + dx1^2*fxx(x, y)/2 + O^3(dx1)
        Can be multiplied by dx1^2 or dx0^2 respectively then taken away to rearrange for the derivative fx(x, y) as
        second order terms cancel, this is a second order approximation.
        f(x+dx0, y)*dx1^2 - f(x-dx1, y)*dx0^2 = f(x, y)*(dx1^2 - dx0^2) +dx0*fx(x, y)*dx1^2 +dx1*fx(x, y)*dx0^2
        fx(x, y) = [f(x+dx0, y)*dx1^2 - f(x-dx1, y)*dx0^2 - f(x, y)*(dx1^2 - dx0^2)]/(dx0*dx1^2 +dx1*dx0^2)
        Which simplifies in the unit normalisation (dx0 = 1) to :
        fx(x, y) = [f(x+dx0, y)*dx1^2 - f(x-dx1, y) - f(x, y)*(dx1^2 - 1)]/(dx1^2 +dx1)
        The same can be applied to calculate the first derivative in y (fy(x, y))
        
        The input x and f are 3 long, where [1] is the central value, [2] is the forward value (where x[2]-x[1] = 1 
        when normalised). dx1 = x[1]-x[0] has to be normalised to the same distance (x[2]-x[1] = 1).
        
        If dx0 = dx1 the central difference approximation is recovered.
        """
        cdef double x1_n, x1_n2

        x1_n = (x[1] - x[0])/(x[2] - x[1])
        x1_n2 = x1_n**2
        return (f[2]*x1_n2 - f[0] - f[1]*(x1_n2 - 1.))/(x1_n + x1_n2)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivitive_d2fdxdy_edge_xy(self, double[:, ::1] f) except? -1e999:
        """
        Calculate d2f/dxdy on an unevenly spaced grid as a 2nd order approximation. Valid at the edges of the grid 
        where higher/lower spline knots don't exist in both x and y.

        For the case where there are no lower spline points than x and y in a 2D array:
        A taylor expansion of f(x, y) with changes in x and y:
        1) f(x+dx0, y+dy0) = f(x, y) + dx0*fx(x, y) + dy0*fy(x, y) + dx0^2*fxx(x, y)/2 + dx0*dy0*fxy(x, y) + dy0^2*fyy(x, y)/2 + O^3(dx0, dy0)
        2) f(x+dx0, y) = f(x, y) + dx0*fx(x, y) + dx0^2*fxx(x, y)/2 + O^3(dx0, dy0)
        3) f(x, y+dy0) = f(x, y) + dy0*fy(x, y) + dy0^2*fyy(x, y)/2 + O^3(dx0, dy0)

        1) - 2) - 3)=>
        4) f(x+dx0, y+dy0) - f(x+dx0, y) - f(x, y+dy0) = -f(x, y) + dx0*dy0*fxy(x, y)
        =>
        fxy(x, y) = [f(x+dx0, y+dy0) - f(x+dx0, y) - f(x, y+dy0) + f(x, y)]/(dx0*dy0)

        For unit square normalisation, dy0 = 1 and dx0 = 1.
        fxy(x, y) = f(x+dx0, y+dy0) - f(x+dx0, y) - f(x, y+dy0) + f(x, y)
        """
        return f[1, 1] - f[0, 1] - f[1, 0] + f[0, 0]

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivitive_d2fdxdy_edge_x(self, double[:] y, double[:, ::1] f) except? -1e999:
        """
        Calculate d2f/dxdy on an unevenly spaced grid as a 2nd order approximation. Valid at the edges of the grid 
        where higher/lower spline knots don't exist in x.

        For the case where there are no lower spline points than x in a 2D array (x, y):
        A taylor expansion of f(x, y) with changes in x and y:
        1) f(x+dx0, y+dy0) = f(x, y) + dx0*fx(x, y) + dy0*fy(x, y) + dx0^2*fxx(x, y)/2 + dx0*dy0*fxy(x, y) + dy0^2*fyy(x, y)/2 + O^3(dx0, dy0)
        2) f(x+dx0, y-dy1) = f(x, y) + dx0*fx(x, y) - dy1*fy(x, y) + dx0^2*fxx(x, y)/2 - dx0*dy1*fxy(x, y) + dy1^2*fyy(x, y)/2 + O^3(dx0, dy1)
        3) f(x, y+dy0) = f(x, y) + dy0*fy(x, y) + dy0^2*fyy(x, y)/2 + O^3(dx0, dy0)
        4) f(x, y-dy1) = f(x, y) - dy1*fy(x, y) + dy1^2*fyy(x, y)/2 + O^3(dx0, dy1)
        
        1) - 2) =>
        5) f(x+dx0, y+dy0) - f(x+dx0, y-dy1) = (dy0 + dy1)*fy(x, y) + dx0*(dy0 + dy1)*fxy(x, y) + (dy0^2 - dy1^2)*fyy(x, y)/2

        3) - 4) =>
        6) f(x, y+dy0) - f(x, y-dy1) = (dy0 + dy1)*fy(x, y) + (dy0^2 - dy1^2)*fyy(x, y)/2 + O^3(dx0, dy0)
        5) - 6)
        f(x+dx0, y+dy0) - f(x+dx0, y-dy1) - f(x, y+dy0) + f(x, y-dy1) = dx0*(dy0 + dy1)*fxy(x, y)
        =>
        fxy(x, y) = [f(x+dx0, y+dy0) - f(x+dx0, y-dy1) - f(x, y+dy0) + f(x, y-dy1)]/(dx0*(dy0 + dy1))

        For unit square normalisation, dy0 = 1 and dx0 = 1.
        fxy(x, y) = [f(x+dx0, y+dy0) - f(x+dx0, y-dy1) - f(x, y+dy0) + f(x, y-dy1)]/(1 + dy1)

        Simplifies to (f[1, 2] - f[0, 2] - f[1, 0] + f[0, 0])/2 if dx0=dx1 and dy0=dy1.
        """
        cdef double dy1
        dy1 = (y[1] - y[0])/(y[2] - y[1])

        return (f[1, 2] - f[0, 2] - f[1, 0] + f[0, 0])/(1. + dy1)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivitive_d2fdxdy_edge_y(self, double[:] x, double[:, ::1] f) except? -1e999:
        """
        Calculate d2f/dxdy on an unevenly spaced grid as a 2nd order approximation. Valid at the edges of the grid 
        where higher/lower spline knots don't exist in y.
        
        The same derivation can be made at if at the edge of the grid in x or y only. x and y are not specifically
        defined in a unique way so between these derivations f(x=x, y=y) => f(x=y, y=x) without any problems.
        The equation for d2f/dxdy becomes:
        =>
        fxy(x, y) = [f(x+dx0, y+dy0) - f(x-dx1, y+dy0) - f(x+dx0, y) + f(x-dx1, y)]/(dy0*(dx0 + dx1))

        For unit square normalisation, dy0 = 1 and dx0 = 1.
        fxy(x, y) = [f(x+dx0, y+dy0) - f(x-dx1, y+dy0) - f(x+dx0, y) + f(x-dx1, y)]/(1 + dx1)

        Simplifies to (f[2, 1] - f[2, 0] - f[0, 1] + f[0, 0])/2 if dx0=dx1 and dy0=dy1.
        """
        cdef double dx1
        dx1 = (x[1] - x[0])/(x[2] - x[1])

        return (f[2, 1] - f[0, 1] - f[2, 0] + f[0, 0])/(1. + dx1)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivitive_d2fdxdy(self, double[:] x, double[:] y, double[:, ::1] f) except? -1e999:
        """
        Calculate d2f/dxdy on an unevenly spaced grid as a 2nd order approximation.
        
        A taylor expansion of f(x, y) with changes in x and y:
        1) f(x+dx0, y+dy0) = f(x, y) + dx0*fx(x, y) + dy0*fy(x, y) + dx0^2*fxx(x, y)/2 + dx0*dy0*fxy(x, y) + dy0^2*fyy(x, y)/2 + O^3(dx0, dy0)
        2) f(x-dx1, y-dy1) = f(x, y) - dx1*fx(x, y) - dy1*fy(x, y) + dx1^2*fxx(x, y)/2 + dx1*dy1*fxy(x, y) + dy1^2*fyy(x, y)/2 + O^3(dx1, dy1)
        3) f(x+dx0, y-dy1) = f(x, y) + dx0*fx(x, y) - dy1*fy(x, y) + dx0^2*fxx(x, y)/2 - dx0*dy1*fxy(x, y) + dy1^2*fyy(x, y)/2 + O^3(dx0, dy1)
        4) f(x-dx1, y+dy0) = f(x, y) - dx1*fx(x, y) + dy0*fy(x, y) + dx1^2*fxx(x, y)/2 - dx1*dy0*fxy(x, y) + dy0^2*fyy(x, y)/2 + O^3(dx0, dy0)
        
        1) - 4) =>
        5) f(x+dx0, y+dy0) - f(x-dx1, y+dy0) = fx(x, y)*(dx0 + dx1) + [fxx(x, y)/2]*(dx0^2 - dx1^2) + fxy(x, y)*(dx0*dy0 + dx1*dy0)
        
        3) - 2) =>
        6) f(x+dx0, y-dy1) - f(x-dx1, y-dy1) = fx(x, y)*(dx0 + dx1) + [fxx(x, y)/2]*(dx0^2 - dx1^2) - fxy(x, y)*(dx0*dy1 + dx1*dy1)
        5) - 6)
        f(x+dx0, y+dy0) - f(x-dx1, y+dy0) - f(x+dx0, y-dy1) + f(x-dx1, y-dy1) = fxy(x, y)*(dx0*dy0 + dx1*dy0 + dx0*dy1 + dx1*dy1)
        =>
        fxy(x, y) = [f(x+dx0, y+dy0) - f(x-dx1, y+dy0) - f(x+dx0, y-dy1) + f(x-dx1, y-dy1)]/(dx0*dy0 + dx1*dy0 + dx0*dy1 + dx1*dy1)
        
        For unit square normalisation, dy0 = 1 and dx0 = 1.
        fxy(x, y) = [f(x+dx0, y+dy0) - f(x-dx1, y+dy0) - f(x+dx0, y-dy1) + f(x-dx1, y-dy1)]/(1 + dx1 + dy1 + dx1*dy1)

        Simplifies to (f[2, 2] - f[0, 2] - f[2, 0] + f[0, 0])/4 if dx0=dx1 and dy0=dy1.
        """
        cdef double dx1, dy1
        dx1 = (x[1] - x[0])/(x[2] - x[1])
        dy1 = (y[1] - y[0])/(y[2] - y[1])

        return (f[2, 2] - f[0, 2] - f[2, 0] + f[0, 0])/(1. + dx1 + dy1 + dx1*dy1)

    def __call__(self, index_x, index_y, derivative_order_x, derivative_order_y, rescale_norm_x, rescale_norm_y):
        return self.evaluate(index_x, index_y, derivative_order_x, derivative_order_y, rescale_norm_x, rescale_norm_y)


id_to_interpolator = {
    _Interpolator2DLinear.ID: _Interpolator2DLinear,
    _Interpolator2DCubic.ID: _Interpolator2DCubic
}


id_to_extrapolator = {
    _Extrapolator2DNone.ID: _Extrapolator2DNone,
    _Extrapolator2DNearest.ID: _Extrapolator2DNearest,
    _Extrapolator2DLinear.ID: _Extrapolator2DLinear,
}


permitted_interpolation_combinations = {
    _Interpolator2DLinear.ID: [_Extrapolator2DNone.ID, _Extrapolator2DNearest.ID, _Extrapolator2DLinear.ID],
    _Interpolator2DCubic.ID: [_Extrapolator2DNone.ID, _Extrapolator2DNearest.ID, _Extrapolator2DLinear.ID]
}
