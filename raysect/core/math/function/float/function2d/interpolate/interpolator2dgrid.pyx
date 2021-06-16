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


cdef class Interpolator2DGrid(Function2D):
    """
    Interface class for Function2D interpolators.

    Coordinate array (x), array (y) and data array (f) are sorted and transformed into Numpy arrays.
    The resulting Numpy arrays are stored as read only. I.e. `writeable` flag of self.x, self.y and self.f
    is set to False. Alteration of the flag may result in unwanted behaviour.

    :note: x, y and f arrays must be of equal shape.

    :param object x: 2D array-like object of real values.
    :param object y: 2D array-like object of real values.
    :param object f: 2D array-like object of real values.

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

    def __init__(self, object x, object y, object f, str interpolation_type,
                 str extrapolation_type, double extrapolation_range):

        # Todo test extrapolation range is positive
        self.x = np.array(x, dtype=np.float64)
        self.x.flags.writeable = False
        self.y = np.array(y, dtype=np.float64)
        self.y.flags.writeable = False
        self.f = np.array(f, dtype=np.float64)
        self.f.flags.writeable = False

        self._x_mv = x
        self._y_mv = y
        self._f_mv = f
        self._last_index_x = self.x.shape[0] - 1
        self._last_index_y = self.y.shape[0] - 1
        self._extrapolation_range = extrapolation_range

        # dimensions checks
        if x.ndim != 1:
            raise ValueError(f'The x array must be 1D. Got {x.shape}.')

        if y.ndim != 1:
            raise ValueError(f'The y array must be 1D. Got {y.shape}.')

        if f.ndim != 2:
            raise ValueError(f'The f array must be 2D. Got {f.shape}.')

        if np.shape(x)[0] != np.shape(f)[0]:
            raise ValueError(f'Shape mismatch between x array ({x.shape}) and f array ({f.shape}).')

        if np.shape(y)[0] != np.shape(f)[1]:
            raise ValueError(f'Shape mismatch between y array ({y.shape}) and f array ({f.shape}).')

        # test monotonicity
        if (np.diff(x) <= 0).any():
            raise ValueError('The x array must be monotonically increasing.')
        if (np.diff(y) <= 0).any():
            raise ValueError('The y array must be monotonically increasing.')

        # create interpolator per interapolation_type argument
        interpolation_type = interpolation_type.lower()
        if interpolation_type not in id_to_interpolator:
            raise ValueError(f'Interpolation type {interpolation_type} not found. options are {id_to_interpolator.keys()}')


        self._interpolator = id_to_interpolator[interpolation_type](self._x_mv, self._y_mv, self._f_mv)

        # create extrapolator per extrapolation_type argument
        extrapolation_type = extrapolation_type.lower()
        if extrapolation_type not in id_to_extrapolator:
            raise ValueError(f'Extrapolation type {interpolation_type} not found. options are {id_to_extrapolator.keys()}')

        self._extrapolator = id_to_extrapolator[extrapolation_type](
            self._x_mv, self._y_mv, self._f_mv, extrapolation_range, self._interpolator
        )

    cdef int extrapolator_index_change(self, int index, int last_index):
        if index == -1:
            index += 1
        elif index == last_index:
            index -= 1
        else:
            raise ValueError('Invalid extrapolator index. Must be -1 for lower and shape-1 for upper extrapolation')
        return index

    cdef int extrapolator_get_edge_index(self, int index, int last_index):
        cdef int edge_index
        if index == -1:
            edge_index = index + 1
        elif index == last_index:
            edge_index = index
        else:
            raise ValueError('Invalid extrapolator index. Must be -1 for lower and shape-1 for upper extrapolation')
        return edge_index

    cdef double evaluate(self, double px, double py) except? -1e999:
        """
        Evaluates the interpolating function.

        :param double px: the point for which an interpolated value is required
        :param double py: the point for which an interpolated value is required
        :return: the interpolated value at point x, y.
        """
        # Find index assuming the grid is the same in x and y
        cdef int edge_x_index, edge_y_index

        cdef int index_x = find_index(self._x_mv, px)
        cdef int index_y = find_index(self._y_mv, py)
        if (index_x == -1 or index_x == self._last_index_x) and (index_y == -1 or index_y == self._last_index_y):
            edge_x_index = self.extrapolator_get_edge_index(index_x, self._last_index_x)
            edge_y_index = self.extrapolator_get_edge_index(index_y, self._last_index_y)
            index_x = self.extrapolator_index_change(index_x, self._last_index_x)
            index_y = self.extrapolator_index_change(index_y, self._last_index_y)

            if np.abs(px - self._x_mv[edge_x_index]) > self._extrapolation_range:
                raise ValueError(
                    f'The specified value (x={px}) is outside of extrapolation range.')
            if np.abs(py - self._y_mv[edge_y_index]) > self._extrapolation_range:
                raise ValueError(
                    f'The specified value (y={py}) is outside of extrapolation range.')
            return self._extrapolator.evaluate_edge_xy(px, py, index_x, index_y, edge_x_index, edge_y_index)

        if index_x == -1 or index_x == self._last_index_x:
            edge_x_index = self.extrapolator_get_edge_index(index_x, self._last_index_x)
            index_x = self.extrapolator_index_change(index_x, self._last_index_x)
            if np.abs(px - self._x_mv[edge_x_index]) > self._extrapolation_range:
                raise ValueError(
                    f'The specified value (x={px}) is outside of extrapolation range.')
            return self._extrapolator.evaluate_edge_x(px, py, index_x, index_y, edge_x_index)

        elif index_y == -1 or index_y == self._last_index_y:
            edge_y_index = self.extrapolator_get_edge_index(index_y, self._last_index_y)
            index_y = self.extrapolator_index_change(index_y, self._last_index_y)
            if np.abs(py - self._y_mv[edge_y_index]) > self._extrapolation_range:
                raise ValueError(
                    f'The specified value (y={py}) is outside of extrapolation range.')
            return self._extrapolator.evaluate_edge_y(px, py, index_x, index_y, edge_y_index)

        else:
            return self._interpolator.evaluate(px, py, index_x, index_y)

    @property
    def domain(self):
        """
        Returns min/max interval of 'x' array.
        Order: min(x), max(x)
        """
        return np.min(self._x_mv), np.max(self._x_mv), np.min(self._y_mv), np.max(self._y_mv)


cdef class _Interpolator2D:
    """
    Base class for 2D interpolators.

    :param x: 2D memory view of the spline point x positions.
    :param y: 2D memory view of the spline point y positions.
    :param f: 2D memory view of the function value at spline point x, y positions.
    """

    ID = NotImplemented
    def __init__(self, double[::1] x, double[::1] y, double[:, ::1] f):
        self._x = x
        self._y = y
        self._f = f
        self._last_index_x = self._x.shape[0] - 1
        self._last_index_y = self._y.shape[0] - 1

    cdef double evaluate(self, double px, double py, int index_x, int index_y) except? -1e999:
        """
        Calculates interpolated value at given point. 

        :param double px: the point for which an interpolated value is required
        :param double py: the point for which an interpolated value is required
        :param int index_x: the lower index of the bin containing point px. (Result of bisection search).   
        :param int index_y: the lower index of the bin containing point py. (Result of bisection search).   
        """
        raise NotImplementedError('_Interpolator is an abstract base class.')


cdef class _Interpolator2DLinear(_Interpolator2D):
    """
    Linear interpolation of 2D function.

    :param x: 2D memory view of the spline point x positions.
    :param y: 2D memory view of the spline point y positions.
    :param f: 2D memory view of the function value at spline point x, y positions.
    """

    ID = 'linear'

    def __init__(self, double[::1] x, double[::1] y, double[:, ::1] f):
        super().__init__(x, y, f)

    cdef double evaluate(self, double px, double py, int index_x, int index_y) except? -1e999:
        return linear2d(
            self._x[index_x], self._x[index_x + 1], self._y[index_y], self._y[index_y + 1],
            self._f[index_x:index_x + 1, index_y:index_y + 1], px, py
        )


cdef class _Interpolator2DCubic(_Interpolator2D):
    """
    Linear interpolation of 2D function.

    :param x: 2D memory view of the spline point x positions.
    :param y: 2D memory view of the spline point y positions.
    :param f: 2D memory view of the function value at spline point x, y positions.
    """

    ID = 'cubic'

    def __init__(self, double[::1] x, double[::1] y, double[:, ::1] f):
        super().__init__(x, y, f)

        # Where 'a' has been calculated already the mask value = 1
        self._mask_a = np.zeros((self._last_index_x, self._last_index_y), dtype=np.float64)
        self._a = np.zeros((self._last_index_x, self._last_index_y, 4, 4), dtype=np.float64)
        self._a_mv = self._a

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double px, double py, int index_x, int index_y) except? -1e999:
        # rescale x between 0 and 1
        cdef double x_scal
        cdef double y_scal
        cdef double[2][2] f, dfdx, dfdy, d2fdxdy
        cdef double x_bound
        cdef double[4][4] a

        x_bound = self._x[index_x + 1] - self._x[index_x]
        if x_bound != 0:
            x_scal = (px - self._x[index_x]) / x_bound
        else:
            raise ZeroDivisionError('Two adjacent spline points have the same x value!')
        y_bound = self._y[index_y + 1] - self._y[index_y]
        if y_bound != 0:
            y_scal = (py - self._y[index_y]) / y_bound
        else:
            raise ZeroDivisionError('Two adjacent spline points have the same y value!')

        # Calculate the coefficients (and gradients at each spline point) if they dont exist
        if not self._mask_a[index_x, index_y]:
            f[0][0] = self._f[index_x, index_y]
            f[1][0] = self._f[index_x + 1, index_y]
            f[0][1] = self._f[index_x, index_y + 1]
            f[1][1] = self._f[index_x + 1, index_y + 1]
            grid_grad = _GridGradients2D(self._x, self._y, self._f)
            dfdx[0][0] = grid_grad(index_x, index_y, 1, 0)
            dfdx[0][1] = grid_grad(index_x, index_y + 1, 1, 0)
            dfdx[1][0] = grid_grad(index_x + 1, index_y, 1, 0)
            dfdx[1][1] = grid_grad(index_x + 1, index_y + 1, 1, 0)

            dfdy[0][0] = grid_grad(index_x, index_y, 0, 1)
            dfdy[0][1] = grid_grad(index_x, index_y + 1, 0, 1)
            dfdy[1][0] = grid_grad(index_x + 1, index_y, 0, 1)
            dfdy[1][1] = grid_grad(index_x + 1, index_y + 1, 0, 1)

            d2fdxdy[0][0] = grid_grad(index_x, index_y, 1, 1)
            d2fdxdy[0][1] = grid_grad(index_x, index_y + 1, 1, 1)
            d2fdxdy[1][0] = grid_grad(index_x + 1, index_y, 1, 1)
            d2fdxdy[1][1] = grid_grad(index_x + 1, index_y + 1, 1, 1)

            calc_coefficients_2d(f, dfdx, dfdy, d2fdxdy, a)
            self._a_mv[index_x, index_y] = a
            self._mask_a[index_x, index_y] = 1
        else:
            a = self._a[index_x, index_y, :4, :4]

        return evaluate_cubic_2d(a, x_scal, y_scal)


cdef class _Extrapolator2D:
    """
    Base class for Function1D extrapolators.

    :param x: 2D memory view of the spline point x positions.
    :param y: 2D memory view of the spline point y positions.
    :param f: 2D memory view of the function value at spline point x, y positions.
    :param extrapolation_range: Range covered by the extrapolator. Padded symmetrically to both ends of the input.
    :param external_interpolator: stored _Interpolator2D object that is being used.
    """

    ID = NotImplemented

    def __init__(self, double[::1] x, double[::1] y, double[:, ::1] f, double extrapolation_range, _Interpolator2D external_interpolator):
        self._range = extrapolation_range
        self._x = x
        self._y = y
        self._f = f
        self._last_index_x = self._x.shape[0] - 1
        self._last_index_y = self._y.shape[0] - 1
        self._external_interpolator = external_interpolator

    cdef double evaluate_edge_x(self, double px, double py, int index_x, int index_y, int edge_x_index) except? -1e999:
        raise NotImplementedError(f'{self.__class__} not implemented.')

    cdef double evaluate_edge_y(self, double px, double py, int index_x, int index_y, int edge_y_index) except? -1e999:
        raise NotImplementedError(f'{self.__class__} not implemented.')

    cdef double evaluate_edge_xy(self, double px, double py, int index_x, int index_y, int edge_x_index, int edge_y_index) except? -1e999:
        raise NotImplementedError(f'{self.__class__} not implemented.')


cdef class _Extrapolator2DNone(_Extrapolator2D):
    """
    Extrapolator that does nothing.

    :param x: 2D memory view of the spline point x positions.
    :param y: 2D memory view of the spline point y positions.
    :param f: 2D memory view of the function value at spline point x, y positions.
    :param extrapolation_range: Range covered by the extrapolator. Padded symmetrically to both ends of the input.
    :param external_interpolator: stored _Interpolator2D object that is being used.
    """

    ID = 'none'

    def __init__(self, double[::1] x, double[::1] y, double[:, ::1] f, double extrapolation_range, _Interpolator2D external_interpolator):
           super().__init__(x, y, f, extrapolation_range, external_interpolator)

    cdef double evaluate_edge_x(self, double px, double py, int index_x, int index_y, int edge_x_index) except? -1e999:
        if px != self._x[-1]:
            raise ValueError(
                f'Extrapolation not available. Interpolate within function range x '
                f'{np.min(self._x)}-{np.max(self._x)}.'
            )
        return self._external_interpolator.evaluate(self._x[-1], py, index_x, index_y)

    cdef double evaluate_edge_y(self, double px, double py, int index_x, int index_y, int edge_y_index) except? -1e999:
        if py != self._y[-1]:
            raise ValueError(
                f'Extrapolation not available. Interpolate within function range y '
                f'{np.min(self._y)}-{np.max(self._y)}.'
            )
        return self._external_interpolator.evaluate(px, self._y[-1], index_x, index_y)

    cdef double evaluate_edge_xy(self, double px, double py, int index_x, int index_y, int edge_x_index, int edge_y_index) except? -1e999:
        if px != self._x[-1] and py != self._y[-1]:
            raise ValueError(
                f'Extrapolation not available. Interpolate within function range x '
                f'{np.min(self._x)}-{np.max(self._x)} '
                f'and y {np.min(self._y)}-{np.max(self._y)}.'
            )
        return self._external_interpolator.evaluate(self._x[-1], self._y[-1], index_x, index_y)


cdef class _Extrapolator2DNearest(_Extrapolator2D):
    """
    Extrapolator that returns nearest input value.

    :param x: 2D memory view of the spline point x positions.
    :param y: 2D memory view of the spline point y positions.
    :param f: 2D memory view of the function value at spline point x, y positions.
    :param extrapolation_range: Range covered by the extrapolator. Padded symmetrically to both ends of the input.
    :param external_interpolator: stored _Interpolator2D object that is being used.
    """

    ID = 'nearest'

    def __init__(self, double[::1] x, double[::1] y, double[:, ::1] f, double extrapolation_range, _Interpolator2D external_interpolator):
           super().__init__(x, y, f, extrapolation_range, external_interpolator)

    cdef double evaluate_edge_x(self, double px, double py, int index_x, int index_y, int edge_x_index) except? -1e999:
        return self._external_interpolator.evaluate(self._x[edge_x_index], py, index_x, index_y)

    cdef double evaluate_edge_y(self, double px, double py, int index_x, int index_y, int edge_y_index) except? -1e999:
        return self._external_interpolator.evaluate(px, self._y[edge_y_index], index_x, index_y)

    cdef double evaluate_edge_xy(self, double px, double py, int index_x, int index_y, int edge_x_index, int edge_y_index) except? -1e999:
        return self._external_interpolator.evaluate(self._x[edge_x_index], self._y[edge_y_index], index_x, index_y)


cdef class _GridGradients2D:
    """
    Gradient method that returns the approximate derivative of a desired order at a specified grid point.

    These methods of finding derivatives are only valid on a 2D grid of points, at the values at the points. Other
    derivative method would be dependent on the interpolator types.

    :param x: 2D memory view of the spline point x positions.
    :param y: 2D memory view of the spline point y positions.
    :param f: 2D memory view of the function value at spline point x, y positions.
    """
    def __init__(self, double[::1] x, double[::1] y, double[:, ::1] f):

        self._x = x
        self._y = y
        self._f = f
        self._last_index_x = self._x.shape[0] - 1
        self._last_index_y = self._y.shape[0] - 1

    cdef double evaluate(self, int index_x, int index_y, int derivative_order_x, int derivative_order_y) except? -1e999:
        """
        Evaluate the derivative of specific order at a grid point.

        :param index_x: The lower index of the x grid cell to evaluate.
        :param index_y: The lower index of the y grid cell to evaluate.
        :param derivative_order_x: An integer of the derivative order x. Only zero if derivative_order_y is nonzero
        :param derivative_order_y: An integer of the derivative order y. Only zero if derivative_order_x is nonzero
        """
        # Find if at the edge of the grid, and in what direction. Then evaluate the gradient.
        cdef double dfdn

        if index_x == 0:
            if index_y == 0:
                dfdn = self.eval_edge_xy(index_x, index_y, derivative_order_x, derivative_order_y)
            elif index_y == self._last_index_y:
                dfdn = self.eval_edge_xy(index_x, index_y - 1, derivative_order_x, derivative_order_y)
            else:
                dfdn = self.eval_edge_x(index_x, index_y, derivative_order_x, derivative_order_y)
        elif index_x == self._last_index_x:
            if index_y == 0:
                dfdn = self.eval_edge_xy(index_x - 1, index_y, derivative_order_x, derivative_order_y)
            elif index_y == self._last_index_y:
                dfdn = self.eval_edge_xy(index_x - 1, index_y - 1, derivative_order_x, derivative_order_y)
            else:
                dfdn = self.eval_edge_x(index_x - 1, index_y, derivative_order_x, derivative_order_y)
        else:
            if index_y == 0:
                dfdn = self.eval_edge_y(index_x, index_y, derivative_order_x, derivative_order_y)
            elif index_y == self._last_index_y:
                dfdn = self.eval_edge_y(index_x, index_y - 1, derivative_order_x, derivative_order_y)
            else:
                dfdn = self.eval_xy(index_x, index_y, derivative_order_x, derivative_order_y)

        return dfdn

    cdef double eval_edge_x(self, int index_x, int index_y, int derivative_order_x, int derivative_order_y):
        cdef double dfdn
        cdef double[::1] x_range, y_range
        cdef double[:, ::1] f_range
        cdef int x_centre = 0, y_centre = 1
        x_range = self._x[index_x:index_x + 2]
        y_range = self._y[index_y - 1:index_y + 2]
        f_range = self._f[index_x:index_x + 2, index_y - 1:index_y + 2]
        if derivative_order_x == 1 and derivative_order_y == 0:
            dfdn = self.derivitive_dfdx_edge(f_range[:, y_centre])
        elif derivative_order_x == 0 and derivative_order_y == 1:
            dfdn = self.derivitive_dfdx(y_range, f_range[x_centre, :])
        elif derivative_order_x == 1 and derivative_order_y == 1:
            dfdn = self.derivitive_d2fdxdy_edge_1(f_range[x_centre:x_centre + 2, y_centre:y_centre + 2])
        return dfdn

    cdef double eval_edge_y(self, int index_x, int index_y, int derivative_order_x, int derivative_order_y):
        cdef double dfdn
        cdef double[::1] x_range, y_range
        cdef double[:, ::1] f_range
        cdef int x_centre = 1, y_centre = 0
        x_range = self._x[index_x - 1:index_x + 2]
        y_range = self._y[index_y:index_y + 2]
        f_range = self._f[index_x - 1:index_x + 2, index_y:index_y + 2]
        if derivative_order_x == 1 and derivative_order_y == 0:
            dfdn = self.derivitive_dfdx(x_range, f_range[:, y_centre])
        elif derivative_order_x == 0 and derivative_order_y == 1:
            dfdn = self.derivitive_dfdx_edge(f_range[x_centre, :])
        elif derivative_order_x == 1 and derivative_order_y == 1:
            dfdn = self.derivitive_d2fdxdy_edge_1(f_range[x_centre:x_centre + 2, y_centre:y_centre + 2])
        return dfdn

    cdef double eval_edge_xy(self, int index_x, int index_y, int derivative_order_x, int derivative_order_y) except? -1e999:
        cdef double dfdn
        cdef double[::1] x_range, y_range
        cdef double[:, ::1] f_range
        cdef int x_centre = 0, y_centre = 0
        x_range = self._x[index_x:index_x + 2]
        y_range = self._y[index_y:index_y + 2]
        f_range = self._f[index_x:index_x + 2, index_y:index_y + 2]

        if derivative_order_x == 1 and derivative_order_y == 0:
            dfdn = self.derivitive_dfdx_edge(f_range[:, y_centre])
        elif derivative_order_x == 0 and derivative_order_y == 1:
            dfdn = self.derivitive_dfdx_edge(f_range[x_centre, :])
        elif derivative_order_x == 1 and derivative_order_y == 1:
            dfdn = self.derivitive_d2fdxdy_edge_1(f_range[x_centre:x_centre + 2, y_centre:y_centre + 2])
        return dfdn

    cdef double eval_xy(self, int index_x, int index_y, int derivative_order_x, int derivative_order_y):
        cdef double dfdn
        cdef double[::1] x_range, y_range
        cdef double[:, ::1] f_range
        cdef int x_centre = 1, y_centre = 1
        x_range = self._x[index_x - 1:index_x + 2]
        y_range = self._y[index_y - 1:index_y + 2]
        f_range = self._f[index_x - 1:index_x + 2, index_y - 1:index_y + 2]

        if derivative_order_x == 1 and derivative_order_y == 0:
            dfdn = self.derivitive_dfdx(x_range, f_range[:, y_centre])
        elif derivative_order_x == 0 and derivative_order_y == 1:
            dfdn = self.derivitive_dfdx(y_range, f_range[x_centre, :])
        elif derivative_order_x == 1 and derivative_order_y == 1:
            dfdn = self.derivitive_d2fdxdy(f_range[x_centre - 1:x_centre + 2, y_centre - 1:y_centre + 2])
        return dfdn

    cdef double derivitive_dfdx_edge(self, double[:] f):
        return f[1] - f[0]

    cdef double derivitive_dfdx(self, double[:] x, double[:] f) except? -1e999:
        cdef double x1_n, x1_n2
        x1_n = (x[1] - x[0])/(x[2] - x[1])
        x1_n2 = x1_n**2
        return (f[2]*x1_n2 - f[0] - f[1]*(x1_n2 - 1.))/(x1_n + x1_n2)

    cdef double derivitive_d2fdxdy_edge_1(self, double[:, ::1] f) except? -1e999:
        return (f[1, 1] - f[0, 1] - f[1, 0] + f[0, 0])/4.

    cdef double derivitive_d2fdxdy(self, double[:, ::1] f) except? -1e999:
        return (f[2, 2] - f[0, 2] - f[2, 0] + f[0, 0])/4.

    def __call__(self, index_x, index_y, derivative_order_x, derivative_order_y):
        return self.evaluate(index_x, index_y, derivative_order_x, derivative_order_y)


id_to_interpolator = {
    _Interpolator2DLinear.ID: _Interpolator2DLinear,
    _Interpolator2DCubic.ID: _Interpolator2DCubic
}

id_to_extrapolator = {
    _Extrapolator2DNone.ID: _Extrapolator2DNone,
    _Extrapolator2DNearest.ID: _Extrapolator2DNearest,
    # _Extrapolator2DLinear.ID: _Extrapolator2DLinear,
    # _Extrapolator2DQuadratic.ID: _Extrapolator2DQuadratic
}

forbidden_interpolation_combinations = {
    # _Interpolator2DLinear.ID: [_Extrapolator2DQuadratic]
}