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
from raysect.core.math.cython.utility cimport find_index, factorial
from raysect.core.math.cython.interpolation.linear cimport linear3d
from raysect.core.math.cython.interpolation.cubic cimport calc_coefficients_3d, evaluate_cubic_3d

# TODO These functions are in 2D too. Move them somewhere common
cdef int find_index_change(int index, int last_index):
    """
    Transforming the output of find_index to find the index lower index of a cell required for an extrapolator.

    Finding the left most index of a grid cell from the output of find_index. The output of find_index is -1 at the 
    lower index, which has a lower border at index 0, and index = last_index at the upper border which is changed to 
    index = last_index - 1.

    :param int index: the index of the lower side of a unit cell.
    :param int last_index: the index of the final point.
    :return: the index of the lower cell at the border of the interpolator spline knots
    """
    cdef int lower_index
    if index == -1:
        lower_index = index + 1
    elif index == last_index:
        lower_index = index - 1
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
        edge_index = index + 1
    elif index == last_index:
        edge_index = index
    else:
        edge_index = index
    return edge_index


cdef class Interpolator3DArray(Function3D):
    """
    Interface class for Function3D interpolators.

    Coordinate array (x), array (y), array (z) and data array (f) are sorted and transformed into Numpy arrays.
    The resulting Numpy arrays are stored as read only. I.e. `writeable` flag of self.x, self.y, self.z and self.f
    is set to False. Alteration of the flag may result in unwanted behaviour.

    :note: x, y, z arrays must be equal in shape to f in the first, second and third dimension respectively.

    :param object x: 1D array-like object of real values.
    :param object y: 1D array-like object of real values.
    :param object z: 1D array-like object of real values.
    :param object f: 3D array-like object of real values.

    :param str interpolation_type: Type of interpolation to use. Options are:
    `linear`: Interpolates the data using linear interpolation.
    `cubic`: Interpolates the data using cubic interpolation.

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
    :param double extrapolation_range_z: Limits the range where extrapolation is permitted. Requesting data beyond the
    extrapolation range results in ValueError. Extrapolation range will be applied as padding symmetrically to both
    ends of the interpolation range (z).
    """

    def __init__(self, object x, object y, object z, object f, str interpolation_type, str extrapolation_type,
                 double extrapolation_range_x, double extrapolation_range_y, double extrapolation_range_z):

        # extrapolation_ranges must be greater than or equal to 0.
        if extrapolation_range_x < 0:
            raise ValueError('extrapolation_range_x must be greater than or equal to 0.')
        if extrapolation_range_y < 0:
            raise ValueError('extrapolation_range_y must be greater than or equal to 0.')
        if extrapolation_range_z < 0:
            raise ValueError('extrapolation_range_z must be greater than or equal to 0.')
        # dimensions checks.
        if x.ndim != 1:
            raise ValueError(f'The x array must be 1D. Got {x.shape}.')

        if y.ndim != 1:
            raise ValueError(f'The y array must be 1D. Got {y.shape}.')

        if z.ndim != 1:
            raise ValueError(f'The z array must be 1D. Got {z.shape}.')

        if f.ndim != 3:
            raise ValueError(f'The f array must be 3D. Got {f.shape}.')

        if np.shape(x)[0] != np.shape(f)[0]:
            raise ValueError(f'Shape mismatch between x array ({x.shape}) and f array ({f.shape}).')

        if np.shape(y)[0] != np.shape(f)[1]:
            raise ValueError(f'Shape mismatch between y array ({y.shape}) and f array ({f.shape}).')

        if np.shape(z)[0] != np.shape(f)[2]:
            raise ValueError(f'Shape mismatch between z array ({z.shape}) and f array ({f.shape}).')

        # test monotonicity
        if (np.diff(x) <= 0).any():
            raise ValueError('The x array must be monotonically increasing.')
        if (np.diff(y) <= 0).any():
            raise ValueError('The y array must be monotonically increasing.')
        if (np.diff(z) <= 0).any():
            raise ValueError('The z array must be monotonically increasing.')

        self.x = np.array(x, dtype=np.float64)
        self.x.flags.writeable = False
        self.y = np.array(y, dtype=np.float64)
        self.y.flags.writeable = False
        self.z = np.array(z, dtype=np.float64)
        self.z.flags.writeable = False
        self.f = np.array(f, dtype=np.float64)
        self.f.flags.writeable = False

        self._x_mv = x
        self._y_mv = y
        self._z_mv = z
        self._f_mv = f
        self._last_index_x = self.x.shape[0] - 1
        self._last_index_y = self.y.shape[0] - 1
        self._last_index_z = self.z.shape[0] - 1
        self._extrapolation_range_x = extrapolation_range_x
        self._extrapolation_range_y = extrapolation_range_y
        self._extrapolation_range_z = extrapolation_range_z

        # create interpolator per interapolation_type argument
        interpolation_type = interpolation_type.lower()
        if interpolation_type not in id_to_interpolator:
            raise ValueError(
                f'Interpolation type {interpolation_type} not found. options are {id_to_interpolator.keys()}')

        self._interpolator = id_to_interpolator[interpolation_type](self._x_mv, self._y_mv, self._z_mv, self._f_mv)

        # create extrapolator per extrapolation_type argument
        extrapolation_type = extrapolation_type.lower()
        if extrapolation_type not in id_to_extrapolator:
            raise ValueError(
                f'Extrapolation type {extrapolation_type} not found. options are {id_to_extrapolator.keys()}')

        self._extrapolator = id_to_extrapolator[extrapolation_type](
            self._x_mv, self._y_mv, self._z_mv, self._f_mv, self._interpolator, extrapolation_range_x,
            extrapolation_range_y, extrapolation_range_z
        )
        # Permit combinations of interpolator and extrapolator that the order of extrapolator is higher than interpolator
        if extrapolation_type not in permitted_interpolation_combinations[interpolation_type]:
            raise ValueError(
                f'Extrapolation type {extrapolation_type} not compatible with interpolation type {interpolation_type}')

    cdef double evaluate(self, double px, double py, double pz) except? -1e999:
        """
        Evaluates the interpolating function.

        Passes the evaluation to the _Interpolator2D object if within the bounds of the spline knots in both the x and 
        the y direction. If outside the bounds in x or y, within the _Extrapolator2D object an extrapolation method is 
        called depending on whether the requested point is out of bounds of the spline knots in the x, y , z or a 
        combination. Because the return value of find_index returns the lower index of the first or last unit cell, 
        extrapolation at the upper or lower index requires the bordering index to evaluate.

        :param double px: the point for which an interpolated value is required.
        :param double py: the point for which an interpolated value is required.
        :param double pz: the point for which an interpolated value is required.
        :return: the interpolated value at point x, y, z.
        """
        # Find index assuming the grid is the same in x, y and z
        cdef int index_x = find_index(self._x_mv, px)
        cdef int index_y = find_index(self._y_mv, py)
        cdef int index_z = find_index(self._z_mv, pz)
        cdef int index_lower_x = find_index_change(index_x, self._last_index_x)
        cdef int index_lower_y = find_index_change(index_y, self._last_index_y)
        cdef int index_lower_z = find_index_change(index_z, self._last_index_z)

        if (index_x == -1 or (index_x == self._last_index_x and px != self._x_mv[-1])) or (
                index_y == -1 or (index_y == self._last_index_y and py != self._y_mv[-1])) or (
                    index_z == -1 or (index_z == self._last_index_z and pz != self._z_mv[-1])):
            return self._extrapolator.evaluate(px, py, pz, index_x, index_y, index_z)
        else:
            return self._interpolator.evaluate(px, py, pz, index_lower_x, index_lower_y, index_lower_z)

    @property
    def domain(self):
        """
        Returns min/max interval of 'x' array.
        Order: min(x), max(x)
        """
        return np.min(self._x_mv), np.max(self._x_mv), np.min(self._y_mv), np.max(self._y_mv)



cdef class _Interpolator3D:
    """
    Base class for 3D interpolators.

    :param x: 1D memory view of the spline point x positions.
    :param y: 1D memory view of the spline point y positions.
    :param z: 1D memory view of the spline point y positions.
    :param f: 3D memory view of the function value at spline point x, y, z positions.
    """

    ID = NotImplemented
    def __init__(self, double[::1] x, double[::1] y, double[::1] z, double[:, :, ::1] f):
        self._x = x
        self._y = y
        self._z = z
        self._f = f
        self._last_index_x = self._x.shape[0] - 1
        self._last_index_y = self._y.shape[0] - 1
        self._last_index_z = self._z.shape[0] - 1

    cdef double evaluate(self, double px, double py, double pz, int index_x, int index_y, int index_z) except? -1e999:
        """
        Calculates interpolated value at given point. 

        :param double px: the point for which an interpolated value is required.
        :param double py: the point for which an interpolated value is required.
        :param double pz: the point for which an interpolated value is required.
        :param int index_x: the lower index of the bin containing point px. (Result of bisection search).   
        :param int index_y: the lower index of the bin containing point py. (Result of bisection search).   
        :param int index_z: the lower index of the bin containing point pz. (Result of bisection search).   
        """
        raise NotImplementedError('_Interpolator is an abstract base class.')

    cdef double _analytic_gradient(self, double px, double py, double pz, int index_x, int index_y, int index_z, int order_x, int order_y, int order_z):
        raise NotImplementedError('_Interpolator is an abstract base class.')



cdef class _Interpolator3DLinear(_Interpolator3D):
    """
    Linear interpolation of 3D function.

    :param x: 1D memory view of the spline point x positions.
    :param y: 1D memory view of the spline point y positions.
    :param z: 1D memory view of the spline point y positions.
    :param f: 3D memory view of the function value at spline point x, y, z positions.
    """

    ID = 'linear'

    def __init__(self, double[::1] x, double[::1] y, double[::1] z, double[:, :, ::1] f):
        super().__init__(x, y, z, f)

    cdef double evaluate(self, double px, double py, double pz, int index_x, int index_y, int index_z) except? -1e999:
        return linear3d(
            self._x[index_x], self._x[index_x + 1], self._y[index_y], self._y[index_y + 1], self._z[index_y], self._z[index_y + 1],
            self._f[index_x:index_x + 1, index_y:index_y + 1, index_z:index_z + 1], px, py, pz
        )

    cdef double _analytic_gradient(self, double px, double py, double pz, int index_x, int index_y, int index_z, int order_x, int order_y, int order_z):
        raise NotImplementedError('TODO.')


cdef class _Extrapolator3D:
    """
    Base class for Function3D extrapolators.

    :param x: 1D memory view of the spline point x positions.
    :param y: 1D memory view of the spline point y positions.
    :param z: 1D memory view of the spline point y positions.
    :param f: 3D memory view of the function value at spline point x, y, z positions.
    :param external_interpolator: stored _Interpolator2D object that is being used.
    """

    ID = NotImplemented

    def __init__(self, double[::1] x, double[::1] y, double[::1] z, double[:, :, ::1] f, _Interpolator3D external_interpolator, double extrapolation_range_x, double extrapolation_range_y, double extrapolation_range_z):
        self._x = x
        self._y = y
        self._z = z
        self._f = f
        self._last_index_x = self._x.shape[0] - 1
        self._last_index_y = self._y.shape[0] - 1
        self._last_index_z = self._z.shape[0] - 1
        self._external_interpolator = external_interpolator
        self._extrapolation_range_x = extrapolation_range_x
        self._extrapolation_range_y = extrapolation_range_y
        self._extrapolation_range_z = extrapolation_range_z

    cdef double evaluate(self, double px, double py, double pz, int index_x, int index_y, int index_z) except? -1e999:
        cdef int index_lower_x = find_index_change(index_x, self._last_index_x)
        cdef int index_lower_y = find_index_change(index_y, self._last_index_y)
        cdef int index_lower_z = find_index_change(index_z, self._last_index_z)
        cdef int edge_x_index = find_edge_index(index_x, self._last_index_x)
        cdef int edge_y_index = find_edge_index(index_y, self._last_index_y)
        cdef int edge_z_index = find_edge_index(index_z, self._last_index_z)

        # Corner in x, y, z
        if (index_x == -1 or index_x == self._last_index_x) and (index_y == -1 or index_y == self._last_index_y) and (index_z == -1 or index_z == self._last_index_z):
            if np.abs(px - self._x[edge_x_index]) > self._extrapolation_range_x:
                raise ValueError(
                    f'The specified value (x={px}) is outside of extrapolation range.')
            if np.abs(py - self._y[edge_y_index]) > self._extrapolation_range_y:
                raise ValueError(
                    f'The specified value (y={py}) is outside of extrapolation range.')
            if np.abs(pz - self._z[edge_z_index]) > self._extrapolation_range_z:
                raise ValueError(
                    f'The specified value (z={pz}) is outside of extrapolation range.')
            return self.evaluate_edge_xyz(px, py, pz, index_lower_x, index_lower_y, index_lower_z, edge_x_index, edge_y_index, edge_z_index)
        elif (index_x == -1 or index_x == self._last_index_x) and (index_y == -1 or index_y == self._last_index_y):
            if np.abs(px - self._x[edge_x_index]) > self._extrapolation_range_x:
                raise ValueError(
                    f'The specified value (x={px}) is outside of extrapolation range.')
            if np.abs(py - self._y[edge_y_index]) > self._extrapolation_range_y:
                raise ValueError(
                    f'The specified value (y={py}) is outside of extrapolation range.')
            return self.evaluate_edge_xy(px, py, pz, index_lower_x, index_lower_y, index_lower_z, edge_x_index, edge_y_index, edge_z_index)
        elif (index_x == -1 or index_x == self._last_index_x) and (index_z == -1 or index_z == self._last_index_z):
            if np.abs(px - self._x[edge_x_index]) > self._extrapolation_range_x:
                raise ValueError(
                    f'The specified value (x={px}) is outside of extrapolation range.')
            if np.abs(pz - self._z[edge_z_index]) > self._extrapolation_range_z:
                raise ValueError(
                    f'The specified value (z={pz}) is outside of extrapolation range.')
            return self.evaluate_edge_xz(px, py, pz, index_lower_x, index_lower_y, index_lower_z, edge_x_index, edge_y_index, edge_z_index)
        elif (index_y == -1 or index_y == self._last_index_y) and (index_z == -1 or index_z == self._last_index_z):
            if np.abs(py - self._y[edge_y_index]) > self._extrapolation_range_y:
                raise ValueError(
                    f'The specified value (y={py}) is outside of extrapolation range.')
            if np.abs(pz - self._z[edge_z_index]) > self._extrapolation_range_z:
                raise ValueError(
                    f'The specified value (z={pz}) is outside of extrapolation range.')
            return self.evaluate_edge_yz(px, py, pz, index_lower_x, index_lower_y, index_lower_z, edge_x_index, edge_y_index, edge_z_index)
        elif index_x == -1 or index_x == self._last_index_x:
            if np.abs(px - self._x[edge_x_index]) > self._extrapolation_range_x:
                raise ValueError(
                    f'The specified value (x={px}) is outside of extrapolation range.')
            return self.evaluate_edge_x(px, py, pz, index_lower_x, index_lower_y, index_lower_z, edge_x_index, edge_y_index, edge_z_index)
        elif index_y == -1 or index_y == self._last_index_y:
            if np.abs(py - self._y[edge_y_index]) > self._extrapolation_range_y:
                raise ValueError(
                    f'The specified value (y={py}) is outside of extrapolation range.')
            return self.evaluate_edge_y(px, py, pz, index_lower_x, index_lower_y, index_lower_z, edge_x_index, edge_y_index, edge_z_index)
        elif index_z == -1 or index_z == self._last_index_z:
            if np.abs(pz - self._z[edge_z_index]) > self._extrapolation_range_z:
                raise ValueError(
                    f'The specified value (z={pz}) is outside of extrapolation range.')
            return self.evaluate_edge_z(px, py, pz, index_lower_x, index_lower_y, index_lower_z, edge_x_index, edge_y_index, edge_z_index)
        else:
            raise ValueError('Interpolated index parsed to extrapolator')

    cdef double evaluate_edge_x(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise NotImplementedError(f'{self.__class__} not implemented.')

    cdef double evaluate_edge_y(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise NotImplementedError(f'{self.__class__} not implemented.')

    cdef double evaluate_edge_z(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise NotImplementedError(f'{self.__class__} not implemented.')

    cdef double evaluate_edge_xy(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise NotImplementedError(f'{self.__class__} not implemented.')

    cdef double evaluate_edge_xz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise NotImplementedError(f'{self.__class__} not implemented.')

    cdef double evaluate_edge_yz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise NotImplementedError(f'{self.__class__} not implemented.')

    cdef double evaluate_edge_xyz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise NotImplementedError(f'{self.__class__} not implemented.')


cdef class _Extrapolator3DNone(_Extrapolator3D):
    """
    Extrapolator that does nothing.

    :param x: 1D memory view of the spline point x positions.
    :param y: 1D memory view of the spline point y positions.
    :param z: 1D memory view of the spline point y positions.
    :param f: 3D memory view of the function value at spline point x, y, z positions.
    :param external_interpolator: stored _Interpolator2D object that is being used.
    """

    ID = 'none'

    def __init__(self, double[::1] x, double[::1] y, double[::1] z, double[:, :, ::1] f, _Interpolator3D external_interpolator, double extrapolation_range_x, double extrapolation_range_y, double extrapolation_range_z):
           super().__init__(x, y, z, f, external_interpolator, extrapolation_range_x, extrapolation_range_y, extrapolation_range_z)


    cdef double evaluate_edge_x(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise ValueError(
            f'Extrapolation not available. Interpolate within function range x '
            f'{np.min(self._x)}-{np.max(self._x)}.'
        )

    cdef double evaluate_edge_y(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise ValueError(
            f'Extrapolation not available. Interpolate within function range y '
            f'{np.min(self._y)}-{np.max(self._y)}.'
        )

    cdef double evaluate_edge_z(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise ValueError(
            f'Extrapolation not available. Interpolate within function range z '
            f'{np.min(self._z)}-{np.max(self._z)}.'
        )

    cdef double evaluate_edge_xy(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise ValueError(
            f'Extrapolation not available. Interpolate within function range x '
            f'{np.min(self._x)}-{np.max(self._x)} and y  {np.min(self._y)}-{np.max(self._y)}.'
        )

    cdef double evaluate_edge_xz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise ValueError(
            f'Extrapolation not available. Interpolate within function range x '
            f'{np.min(self._x)}-{np.max(self._x)} and z  {np.min(self._z)}-{np.max(self._z)}.'
        )

    cdef double evaluate_edge_yz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise ValueError(
            f'Extrapolation not available. Interpolate within function range y '
            f'{np.min(self._y)}-{np.max(self._y)} and z {np.min(self._z)}-{np.max(self._z)}.'
        )

    cdef double evaluate_edge_xyz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise ValueError(
            f'Extrapolation not available. Interpolate within function range x '
            f'{np.min(self._x)}-{np.max(self._x)}, y  {np.min(self._y)}-{np.max(self._y)} '
            f'and z {np.min(self._z)}-{np.max(self._z)}.'
        )

id_to_interpolator = {
    _Interpolator3DLinear.ID: _Interpolator3DLinear,
    # _Interpolator3DCubic.ID: _Interpolator3DCubic
}

id_to_extrapolator = {
    _Extrapolator3DNone.ID: _Extrapolator3DNone,
    # _Extrapolator3DNearest.ID: _Extrapolator3DNearest,
    # _Extrapolator3DLinear.ID: _Extrapolator3DLinear,
    # _Extrapolator3DQuadratic.ID: _Extrapolator3DQuadratic
}

permitted_interpolation_combinations = {
    _Interpolator3DLinear.ID: [_Extrapolator3DNone.ID],
    # _Interpolator3DCubic.ID: [_Extrapolator3DNone.ID, _Extrapolator3DNearest.ID, _Extrapolator3DLinear.ID]
}