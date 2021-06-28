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

cdef class Interpolator3DGrid(Function3D):
    """
    Interface class for Function2D interpolators.

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
            self._x_mv, self._y_mv, self._z_mv, self._f_mv, self._interpolator
        )

    cdef int extrapolator_index_change(self, int index, int last_index):
        """
        Transforming the output of find_index to find the index lower index of a cell required for an extrapolator.

        Finding the left most index of a grid cell from the output of find_index. The output of find_index is -1 at the 
        lower index, which has a lower border at index 0, and index = last_index at the upper border which is changed to 
        index = last_index - 1.

        :param int index: the index of the lower side of a unit cell.
        :param int last_index: the index of the final point.
        :return: the index of the lower cell at the border of the interpolator spline knots
        """
        if index == -1:
            index += 1
        elif index == last_index:
            index -= 1
        else:
            raise ValueError('Invalid extrapolator index. Must be -1 for lower and shape-1 for upper extrapolation')
        return index

    cdef int extrapolator_get_edge_index(self, int index, int last_index):
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
            raise ValueError('Invalid extrapolator index. Must be -1 for lower and shape-1 for upper extrapolation.')
        return edge_index

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
        cdef int edge_x_index, edge_y_index

        cdef int index_x = find_index(self._x_mv, px)
        cdef int index_y = find_index(self._y_mv, py)
        cdef int index_z = find_index(self._z_mv, pz)
        if (index_x == -1 or index_x == self._last_index_x) and (index_y == -1 or index_y == self._last_index_y):
            edge_x_index = self.extrapolator_get_edge_index(index_x, self._last_index_x)
            edge_y_index = self.extrapolator_get_edge_index(index_y, self._last_index_y)
            index_x = self.extrapolator_index_change(index_x, self._last_index_x)
            index_y = self.extrapolator_index_change(index_y, self._last_index_y)

            if np.abs(px - self._x_mv[edge_x_index]) > self._extrapolation_range_x:
                raise ValueError(
                    f'The specified value (x={px}) is outside of extrapolation range.')
            if np.abs(py - self._y_mv[edge_y_index]) > self._extrapolation_range_y:
                raise ValueError(
                    f'The specified value (y={py}) is outside of extrapolation range.')
            return self._extrapolator.evaluate_edge_xy(px, py, index_x, index_y, edge_x_index, edge_y_index)

        elif index_x == -1 or index_x == self._last_index_x:
            edge_x_index = self.extrapolator_get_edge_index(index_x, self._last_index_x)
            index_x = self.extrapolator_index_change(index_x, self._last_index_x)
            if np.abs(px - self._x_mv[edge_x_index]) > self._extrapolation_range_x:
                raise ValueError(
                    f'The specified value (x={px}) is outside of extrapolation range.')
            return self._extrapolator.evaluate_edge_x(px, py, index_x, index_y, edge_x_index)

        elif index_y == -1 or index_y == self._last_index_y:
            edge_y_index = self.extrapolator_get_edge_index(index_y, self._last_index_y)
            index_y = self.extrapolator_index_change(index_y, self._last_index_y)
            if np.abs(py - self._y_mv[edge_y_index]) > self._extrapolation_range_y:
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


id_to_interpolator = {
    # _Interpolator2DLinear.ID: _Interpolator2DLinear,
    # _Interpolator2DCubic.ID: _Interpolator2DCubic
}

id_to_extrapolator = {
    # _Extrapolator2DNone.ID: _Extrapolator2DNone,
    # _Extrapolator2DNearest.ID: _Extrapolator2DNearest,
    # _Extrapolator2DLinear.ID: _Extrapolator2DLinear,
    # _Extrapolator2DQuadratic.ID: _Extrapolator2DQuadratic
}

forbidden_interpolation_combinations = {
    # _Interpolator2DLinear.ID: [_Extrapolator2DQuadratic]
}