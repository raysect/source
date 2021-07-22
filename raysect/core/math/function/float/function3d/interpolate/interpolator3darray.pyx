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
from raysect.core.math.cython.interpolation.linear cimport linear3d
from raysect.core.math.cython.interpolation.cubic cimport calc_coefficients_3d, evaluate_cubic_3d

# TODO These functions are in 2D too. Move them somewhere common
cdef double rescale_lower_normalisation(dfdn, x_lower, x, x_upper):
    """
    Derivatives that are normalised to the unt square (x_upper - x) = 1 are un-normalised, then re-normalised to
    (x - x_lower)
    """
    return dfdn * (x - x_lower)/(x_upper - x)

cdef double lookup_factorial(int n):
    """
    A small lookup table for a factorial calculation.

    So far this is only required for cubic functions, so going up to 3!.
    """
    cdef double[4] factorial
    factorial[0] = 1.
    factorial[1] = 1.
    factorial[2] = 2.
    factorial[3] = 6.
    return factorial[n]


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

    :param object x: 1D array-like object of real values storing the x spline knot positions.
    :param object y: 1D array-like object of real values storing the y spline knot positions.
    :param object z: 1D array-like object of real values storing the z spline knot positions.
    :param object f: 3D array-like object of real values storing the spline knot function value at x, y, z.
    :param str interpolation_type: Type of interpolation to use. Options are:
        `linear`: Interpolates the data using piecewise trilinear interpolation.
        `cubic`: Interpolates the data using piecewise tricubic interpolation.
    :param str extrapolation_type: Type of extrapolation to use. Options are:
        `none`: Attempt to access data outside of x's and y's range will yield ValueError.
        `nearest`: Extrapolation results is the nearest position x and y value in the interpolation domain.
        `linear`: Extrapolate bilinearly the interpolation function.
    :param double extrapolation_range_x:    Limits the range where extrapolation is permitted. Requesting data beyond the
        extrapolation range results in ValueError. Extrapolation range will be applied as padding symmetrically to both
        ends of the interpolation range (x).
    :param double extrapolation_range_y: Limits the range where extrapolation is permitted. Requesting data beyond the
        extrapolation range results in ValueError. Extrapolation range will be applied as padding symmetrically to both
        ends of the interpolation range (y).
    :param double extrapolation_range_z: Limits the range where extrapolation is permitted. Requesting data beyond the
        extrapolation range results in ValueError. Extrapolation range will be applied as padding symmetrically to both
        ends of the interpolation range (z).

    .. code-block:: python

        >>> from raysect.core.math.function.float.function3d.interpolate.interpolator3darray import Interpolator3DArray
        >>>
        >>> x = np.linspace(-1., 1., 20)
        >>> y = np.linspace(-1., 1., 20)
        >>> z = np.linspace(-1., 1., 20)
        >>> x_array, y_array, z_array = np.meshgrid(x, y, z, indexing='ij')
        >>> f = np.exp(-(x_array**2 + y_array**2 + z_array**2))
        >>> interpolator3D = Interpolator3DArray(x, y, z, f, 'cubic', 'nearest', 1.0, 1.0, 1.0)
        >>> # Interpolation
        >>> interpolator3D(1.0, 1.0, 0.2)
        0.1300281183136766
        >>> # Extrapolation
        >>> interpolator3D(1.0, 1.0, 1.1)
        0.0497870683678659
        >>> # Extrapolation out of bounds
        >>> interpolator3D(1.0, 1.0, 2.1)
        ValueError: The specified value (z=2.1) is outside of extrapolation range.

    :note: All input derivatives used in calculations use the previous and next indices in the spline knot arrays.
        At the edge of the spline knot arrays the index of the edge of the array is is used instead.
    :note: x, y, z arrays must be equal in shape to f in the first, second and third dimension respectively.
    :note: x, y and z must be monotonically increasing arrays.
    :warning: x, y, z, f must all be c contiguous in memory. Avoid operations that break this condition when
        preparing data (e.g. don't transpose any data arrays).

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

        # Test that arrays are too short for any interpolation. Specific requirements for interpolation/extrapolation
        # objects should be in __init__ of the internal interpolator/extrapolator. Must check after size checks.
        if np.shape(x)[0] < 1:
            raise ValueError('The x, y, z arrays need more than 1 point to interpolate from.')

        # test monotonicity
        if (np.diff(x) <= 0).any():
            raise ValueError('The x array must be monotonically increasing.')
        if (np.diff(y) <= 0).any():
            raise ValueError('The y array must be monotonically increasing.')
        if (np.diff(z) <= 0).any():
            raise ValueError('The z array must be monotonically increasing.')

        self.x = np.array(x, dtype=np.float64, order='c')
        self.x.flags.writeable = False
        self.y = np.array(y, dtype=np.float64, order='c')
        self.y.flags.writeable = False
        self.z = np.array(z, dtype=np.float64, order='c')
        self.z.flags.writeable = False
        self.f = np.array(f, dtype=np.float64, order='c')
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

    cdef double analytic_gradient(self, double px, double py, double pz, int index_x, int index_y, int index_z, int order_x, int order_y, int order_z):
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
            self._x[index_x], self._x[index_x + 1], self._y[index_y], self._y[index_y + 1], self._z[index_z], self._z[index_z + 1],
            self._f[index_x:index_x + 2, index_y:index_y + 2, index_z:index_z + 2], px, py, pz
        )

    cdef double analytic_gradient(self, double px, double py, double pz, int index_x, int index_y, int index_z, int order_x, int order_y, int order_z):
        """
        Calculate the normalised derivative of specified order in a unit cube.

        The order of the derivative corresponds to order_x and order_y as the number of times differentiated. For 
        example order_x = 1 and order_y = 1 is d2f/dxdy. The normalised gradient is calculated of the trilinear 
        function f(x, y, z) = a0 + a1x + a2y + a3z + a4xy + a5xz +a6yz + a7xyz, which is the product of 3 linear 
        functions. The derivatives are therefore df/dx = a1 + a4y +a5z +a7yz ; df/dy = a2 + a4x +a6z +a7xz ; 
        df/dz = a3 + a5x +a6y +a7xy ; d2f/dxdy = a4 +a7z; d2f/dxdz = a5 +a7y; d2f/dydz = a6 +a7x; d3f/dxdydz = a7. 
        The derivatives are calculated on the normalised unit cube.

        :param double px: the point for which an interpolated value is required.
        :param double py: the point for which an interpolated value is required.
        :param double pz: the point for which an interpolated value is required.
        :param int index_x: the lower index of the bin containing point px. (Result of bisection search).   
        :param int index_y: the lower index of the bin containing point py. (Result of bisection search).   
        :param int index_z: the lower index of the bin containing point pz. (Result of bisection search).   
        :param int order_x: the derivative order in the x direction.
        :param int order_y: the derivative order in the y direction.
        :param int order_z: the derivative order in the z direction.
        """
        cdef double df_dn
        cdef double[8] a
        self.calculate_coefficients(index_x, index_y, index_z, a)
        if order_x == 1 and order_y == 1 and order_z == 1:
            df_dn = a[7]
        elif order_x == 1 and order_y == 1 and order_z == 0:
            df_dn = a[4] + a[7] * (pz - self._z[index_z]) / (self._z[index_z + 1] - self._z[index_z])
        elif order_x == 1 and order_y == 0 and order_z == 1:
            df_dn = a[5] + a[7] * (py - self._y[index_y]) / (self._y[index_y + 1] - self._y[index_y])
        elif order_x == 0 and order_y == 1 and order_z == 1:
            df_dn = a[6] + a[7] * (px - self._x[index_x]) / (self._x[index_x + 1] - self._x[index_x])
        elif order_x == 1 and order_y == 0 and order_z == 0:
            df_dn = a[1] + a[4] * (py - self._y[index_y]) / (self._y[index_y + 1] - self._y[index_y]) + \
                    a[5] * (pz - self._z[index_z]) / (self._z[index_z + 1] - self._z[index_z]) + \
                    a[7]* (py - self._y[index_y]) / (self._y[index_y + 1] - self._y[index_y]) * (pz - self._z[index_z]) / (self._z[index_z + 1] - self._z[index_z])
        elif order_x == 0 and order_y == 1 and order_z == 0:
            df_dn = a[2] + a[4] * (px - self._x[index_x]) / (self._x[index_x + 1] - self._x[index_x]) + \
                    a[6] * (pz - self._z[index_z]) / (self._z[index_z + 1] - self._z[index_z]) + \
                    a[7]* (px - self._x[index_x]) / (self._x[index_x + 1] - self._x[index_x]) * (pz - self._z[index_z]) / (self._z[index_z + 1] - self._z[index_z])
        elif order_x == 0 and order_y == 0 and order_z == 1:
            df_dn = a[3] + a[5] * (px - self._x[index_x]) / (self._x[index_x + 1] - self._x[index_x]) + \
                    a[6] * (py - self._y[index_y]) / (self._y[index_y + 1] - self._y[index_y]) + \
                    a[7] * (px - self._x[index_x]) / (self._x[index_x + 1] - self._x[index_x]) * (py - self._y[index_y]) / (self._y[index_y + 1] - self._y[index_y])
        else:
            raise ValueError('order_x and order_y must be 1 and 0 for the linear interpolator.')
        return df_dn


    cdef calculate_coefficients(self, int index_x, int index_y, int index_z, double[8] a):
        """
        Calculate the trilinear coefficients in a unit cube.

        The trilinear function (which is the product of 3 linear functions) 
        f(x, y, z) = a0 + a1x + a2y + a3z + a4xy + a5xz +a6yz + a7xyz. Coefficients a0 - a7 are calculated for one unit 
        cube. The coefficients are calculated from inverting the equation Xa = fv.
        Where:
        X = [[1, x1, y1, z1, x1y1, x1z1, y1z1, x1y1z1],         a = [a0,         fv = [f(0, 0, 0),
            [1, x2, y1, z1, x2y1, x2z1, y1z1, x2y1z1],               a1,               f(1, 0, 0),
            [1, x1, y2, z1, x1y2, x1z1, y2z1, x1y2z1],               a2,               f(0, 1, 0),
            [1, x2, y2, z1, x2y2, x2z1, y2z1, x2y2z1],               a3,               f(1, 1, 0),
            [1, x1, y1, z2, x1y1, x1z2, y1z2, x1y1z2],               a4,               f(0, 0, 1),
            [1, x2, y1, z2, x2y1, x2z2, y1z2, x2y1z2],               a5,               f(1, 0, 1),
            [1, x1, y2, z2, x1y2, x1z2, y2z2, x1y2z2],               a6,               f(0, 1, 1),
            [1, x2, y2, z2, x2y2, x2z2, y2z2, x2y2z2]]               a7]               f(1, 1, 1)]
        This simplifies where x1, y1, z1 = 0, x2, y2, z2 = 1 for the unit cube to find a = X^{-1} fv
        where:
        a[0] = f[0][0][0]
        a[1] = - f[0][0][0] + f[1][0][0]
        a[2] = - f[0][0][0] + f[0][1][0]
        a[3] = - f[0][0][0] + f[0][0][1]
        a[4] = f[0][0][0] - f[0][1][0] - f[1][0][0] + f[1][1][0]
        a[5] = f[0][0][0] - f[0][0][1] - f[1][0][0] + f[1][0][1]
        a[6] = f[0][0][0] - f[0][0][1] - f[0][1][0] + f[0][1][1]
        a[7] = - f[0][0][0] + f[0][0][1] + f[0][1][0] - f[0][1][1] + f[1][0][0] - f[1][0][1] - f[1][1][0] + f[1][1][1]

        :param int index_x: the lower index of the bin containing point px. (Result of bisection search).   
        :param int index_y: the lower index of the bin containing point py. (Result of bisection search). 
        :param int index_z: the lower index of the bin containing point pz. (Result of bisection search). 
        :param double[8] a: The coefficients of the trilinear equation a0, - a7.
        """
        cdef double[2][2][2] f

        # Calculate the coefficients (and gradients at each spline point) if they dont exist
        f[0][0][0] = self._f[index_x, index_y, index_z]
        f[0][1][0] = self._f[index_x, index_y + 1, index_z]
        f[0][0][1] = self._f[index_x, index_y, index_z + 1]
        f[0][1][1] = self._f[index_x, index_y + 1, index_z + 1]
        f[1][0][0] = self._f[index_x + 1, index_y, index_z]
        f[1][1][0] = self._f[index_x + 1, index_y + 1, index_z]
        f[1][0][1] = self._f[index_x + 1, index_y, index_z + 1]
        f[1][1][1] = self._f[index_x + 1, index_y + 1, index_z + 1]

        a[0] = f[0][0][0]
        a[1] = - f[0][0][0] + f[1][0][0]
        a[2] = - f[0][0][0] + f[0][1][0]
        a[3] = - f[0][0][0] + f[0][0][1]
        a[4] = f[0][0][0] - f[0][1][0] - f[1][0][0] + f[1][1][0]
        a[5] = f[0][0][0] - f[0][0][1] - f[1][0][0] + f[1][0][1]
        a[6] = f[0][0][0] - f[0][0][1] - f[0][1][0] + f[0][1][1]
        a[7] = - f[0][0][0] + f[0][0][1] + f[0][1][0] - f[0][1][1] + f[1][0][0] - f[1][0][1] - f[1][1][0] + f[1][1][1]



cdef class _Interpolator3DCubic(_Interpolator3D):
    """
    Cubic interpolation of a 3D function.

    When called, stores cubic polynomial coefficients from the value of the function, df/dx, df/dy, df/dz , d2f/dxdy,
    d2f/dxdz , d2f/dydz and d3f/dxdydz at the neighbouring spline knots using _ArrayDerivative3D object.
    The polynomial coefficients and gradients are calculated between each spline knots in a unit square.

    :param x: 1D memory view of the spline point x positions.
    :param y: 1D memory view of the spline point y positions.
    :param z: 1D memory view of the spline point z positions.
    :param f: 3D memory view of the function value at spline point x, y, z positions.
    """

    ID = 'cubic'

    def __init__(self, double[::1] x, double[::1] y, double[::1] z, double[:, :, ::1] f):
        super().__init__(x, y, z, f)

        # Where 'a' has been calculated already the mask value = 1
        self._mask_a = np.zeros((self._last_index_x, self._last_index_y, self._last_index_z), dtype=np.float64)
        self._a = np.zeros((self._last_index_x, self._last_index_y, self._last_index_z, 4, 4, 4), dtype=np.float64)
        self._a_mv = self._a

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double px, double py, double pz, int index_x, int index_y, int index_z) except? -1e999:
        # rescale x between 0 and 1
        cdef double x_scal
        cdef double y_scal
        cdef double z_scal
        cdef double x_bound, y_bound, z_bound
        cdef double[4][4][4] a

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
        z_bound = self._z[index_z + 1] - self._z[index_z]
        if z_bound != 0:
            z_scal = (pz - self._z[index_z]) / z_bound
        else:
            raise ZeroDivisionError('Two adjacent spline points have the same z value!')

        # Calculate the coefficients (and gradients at each spline point) if they dont exist.
        self.cache_coefficients(index_x, index_y, index_z, a)

        return evaluate_cubic_3d(a, x_scal, y_scal, z_scal)

    cdef cache_coefficients(self, int index_x, int index_y, int index_z, double[4][4][4] a):
        """
        Calculates and stores, or loads previously stored cubic coefficients.

        :param int index_x: the lower index of the bin containing point px. (Result of bisection search).   
        :param int index_y: the lower index of the bin containing point py. (Result of bisection search). 
        :param int index_z: the lower index of the bin containing point pz. (Result of bisection search). 
        :param double[4][4][4] a: The coefficients of the tricubic equation.
        """
        cdef double[2][2][2] f, dfdx, dfdy, dfdz, d2fdxdy, d2fdxdz, d2fdydz, d3fdxdydz
        cdef int i, j, k

        # Calculate the coefficients (and gradients at each spline point) if they dont exist
        if not self._mask_a[index_x, index_y, index_z]:
            f[0][0][0] = self._f[index_x, index_y, index_z]
            f[0][1][0] = self._f[index_x, index_y + 1, index_z]
            f[0][1][1] = self._f[index_x, index_y + 1, index_z + 1]
            f[0][0][1] = self._f[index_x, index_y, index_z + 1]
            f[1][0][0] = self._f[index_x + 1, index_y, index_z]
            f[1][1][0] = self._f[index_x + 1, index_y + 1, index_z]
            f[1][0][1] = self._f[index_x + 1, index_y, index_z + 1]
            f[1][1][1] = self._f[index_x + 1, index_y + 1, index_z + 1]

            array_derivative = _ArrayDerivative3D(self._x, self._y, self._z, self._f)
            dfdx[0][0][0] = array_derivative(index_x, index_y, index_z, 1, 0, 0, False, False, False)
            dfdx[0][1][0] = array_derivative(index_x, index_y + 1, index_z, 1, 0, 0, False, True, False)
            dfdx[0][1][1] = array_derivative(index_x, index_y + 1, index_z + 1, 1, 0, 0, False, True, True)
            dfdx[0][0][1] = array_derivative(index_x, index_y, index_z + 1, 1, 0, 0, False, False, True)
            dfdx[1][0][0] = array_derivative(index_x + 1, index_y, index_z, 1, 0, 0, True, False, False)
            dfdx[1][1][0] = array_derivative(index_x + 1, index_y + 1, index_z, 1, 0, 0, True, True, False)
            dfdx[1][0][1] = array_derivative(index_x + 1, index_y, index_z + 1, 1, 0, 0, True, False, True)
            dfdx[1][1][1] = array_derivative(index_x + 1, index_y + 1, index_z + 1, 1, 0, 0, True, True, True)

            dfdy[0][0][0] = array_derivative(index_x, index_y, index_z, 0, 1, 0, False, False, False)
            dfdy[0][1][0] = array_derivative(index_x, index_y + 1, index_z, 0, 1, 0, False, True, False)
            dfdy[0][1][1] = array_derivative(index_x, index_y + 1, index_z + 1, 0, 1, 0, False, True, True)
            dfdy[0][0][1] = array_derivative(index_x, index_y, index_z + 1, 0, 1, 0, False, False, True)
            dfdy[1][0][0] = array_derivative(index_x + 1, index_y, index_z, 0, 1, 0, True, False, False)
            dfdy[1][1][0] = array_derivative(index_x + 1, index_y + 1, index_z, 0, 1, 0, True, True, False)
            dfdy[1][0][1] = array_derivative(index_x + 1, index_y, index_z + 1, 0, 1, 0, True, False, True)
            dfdy[1][1][1] = array_derivative(index_x + 1, index_y + 1, index_z + 1, 0, 1, 0, True, True, True)

            dfdz[0][0][0] = array_derivative(index_x, index_y, index_z, 0, 0, 1, False, False, False)
            dfdz[0][1][0] = array_derivative(index_x, index_y + 1, index_z, 0, 0, 1, False, True, False)
            dfdz[0][1][1] = array_derivative(index_x, index_y + 1, index_z + 1, 0, 0, 1, False, True, True)
            dfdz[0][0][1] = array_derivative(index_x, index_y, index_z + 1, 0, 0, 1, False, False, True)
            dfdz[1][0][0] = array_derivative(index_x + 1, index_y, index_z, 0, 0, 1, True, False, False)
            dfdz[1][1][0] = array_derivative(index_x + 1, index_y + 1, index_z, 0, 0, 1, True, True, False)
            dfdz[1][0][1] = array_derivative(index_x + 1, index_y, index_z + 1, 0, 0, 1, True, False, True)
            dfdz[1][1][1] = array_derivative(index_x + 1, index_y + 1, index_z + 1, 0, 0, 1, True, True, True)

            d2fdxdy[0][0][0] = array_derivative(index_x, index_y, index_z, 1, 1, 0, False, False, False)
            d2fdxdy[0][1][0] = array_derivative(index_x, index_y + 1, index_z, 1, 1, 0, False, True, False)
            d2fdxdy[0][1][1] = array_derivative(index_x, index_y + 1, index_z + 1, 1, 1, 0, False, True, True)
            d2fdxdy[0][0][1] = array_derivative(index_x, index_y, index_z + 1, 1, 1, 0, False, False, True)
            d2fdxdy[1][0][0] = array_derivative(index_x + 1, index_y, index_z, 1, 1, 0, True, False, False)
            d2fdxdy[1][1][0] = array_derivative(index_x + 1, index_y + 1, index_z, 1, 1, 0, True, True, False)
            d2fdxdy[1][0][1] = array_derivative(index_x + 1, index_y, index_z + 1, 1, 1, 0, True, False, True)
            d2fdxdy[1][1][1] = array_derivative(index_x + 1, index_y + 1, index_z + 1, 1, 1, 0, True, True, True)

            d2fdxdz[0][0][0] = array_derivative(index_x, index_y, index_z, 1, 0, 1, False, False, False)
            d2fdxdz[0][1][0] = array_derivative(index_x, index_y + 1, index_z, 1, 0, 1, False, True, False)
            d2fdxdz[0][1][1] = array_derivative(index_x, index_y + 1, index_z + 1, 1, 0, 1, False, True, True)
            d2fdxdz[0][0][1] = array_derivative(index_x, index_y, index_z + 1, 1, 0, 1, False, False, True)
            d2fdxdz[1][0][0] = array_derivative(index_x + 1, index_y, index_z, 1, 0, 1, True, False, False)
            d2fdxdz[1][1][0] = array_derivative(index_x + 1, index_y + 1, index_z, 1, 0, 1, True, True, False)
            d2fdxdz[1][0][1] = array_derivative(index_x + 1, index_y, index_z + 1, 1, 0, 1, True, False, True)
            d2fdxdz[1][1][1] = array_derivative(index_x + 1, index_y + 1, index_z + 1, 1, 0, 1, True, True, True)

            d2fdydz[0][0][0] = array_derivative(index_x, index_y, index_z, 0, 1, 1, False, False, False)
            d2fdydz[0][1][0] = array_derivative(index_x, index_y + 1, index_z, 0, 1, 1, False, True, False)
            d2fdydz[0][1][1] = array_derivative(index_x, index_y + 1, index_z + 1, 0, 1, 1, False, True, True)
            d2fdydz[0][0][1] = array_derivative(index_x, index_y, index_z + 1, 0, 1, 1, False, False, True)
            d2fdydz[1][0][0] = array_derivative(index_x + 1, index_y, index_z, 0, 1, 1, True, False, False)
            d2fdydz[1][1][0] = array_derivative(index_x + 1, index_y + 1, index_z, 0, 1, 1, True, True, False)
            d2fdydz[1][0][1] = array_derivative(index_x + 1, index_y, index_z + 1, 0, 1, 1, True, False, True)
            d2fdydz[1][1][1] = array_derivative(index_x + 1, index_y + 1, index_z + 1, 0, 1, 1, True, True, True)


            d3fdxdydz[0][0][0] = array_derivative(index_x, index_y, index_z, 1, 1, 1, False, False, False)
            d3fdxdydz[0][1][0] = array_derivative(index_x, index_y + 1, index_z, 1, 1, 1, False, True, False)
            d3fdxdydz[0][1][1] = array_derivative(index_x, index_y + 1, index_z + 1, 1, 1, 1, False, True, True)
            d3fdxdydz[0][0][1] = array_derivative(index_x, index_y, index_z + 1, 1, 1, 1, False, False, True)
            d3fdxdydz[1][0][0] = array_derivative(index_x + 1, index_y, index_z, 1, 1, 1, True, False, False)
            d3fdxdydz[1][1][0] = array_derivative(index_x + 1, index_y + 1, index_z, 1, 1, 1, True, True, False)
            d3fdxdydz[1][0][1] = array_derivative(index_x + 1, index_y, index_z + 1, 1, 1, 1, True, False, True)
            d3fdxdydz[1][1][1] = array_derivative(index_x + 1, index_y + 1, index_z + 1, 1, 1, 1, True, True, True)

            calc_coefficients_3d(f, dfdx, dfdy, dfdz, d2fdxdy, d2fdxdz, d2fdydz, d3fdxdydz, a)
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        self._a[index_x, index_y, index_z, i, j, k] = a[i][j][k]
            self._mask_a[index_x, index_y, index_z] = 1
        else:
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        a[i][j][k] = self._a[index_x, index_y, index_z, i, j, k]

    cdef double analytic_gradient(self, double px, double py, double pz, int index_x, int index_y, int index_z, int order_x, int order_y, int order_z):
        #TODO edit docstring
        """
        Calculate the normalised gradient of specified order in a unit cube.

        The order of the derivative corresponds to order_x, order_y and order_z as the number of times differentiated. For 
        example order_x = 1 and order_y = 1 is d2f/dxdy. The normalised gradient is calculated for the tricubic by 
        generalising each orders derivative coefficient to n!/(n-order)! . e.g. for n = [1, 2, 3], for order 1
        these are [1, 2, 3] for the derivative dfdx = a1 + 2*a2*x + 3*a3*x^2, order 2 has [2, 6] for 
        d2fdx2 = 2*a2 + 6*a3*x. These combine in x and y by selecting elements of the matrix 'a' and the x^n and y^n 
        that remains after differentiation.

        :param double px: the point for which an interpolated value is required.
        :param double py: the point for which an interpolated value is required.
        :param double pz: the point for which an interpolated value is required.
        :param int index_x: the lower index of the bin containing point px. (Result of bisection search).   
        :param int index_y: the lower index of the bin containing point py. (Result of bisection search).   
        :param int index_z: the lower index of the bin containing point pz. (Result of bisection search).   
        :param int order_x: the derivative order in the x direction.
        :param int order_y: the derivative order in the y direction.
        :param int order_z: the derivative order in the z direction.
        """
        # rescale x between 0 and 1
        cdef double x_scal
        cdef double y_scal
        cdef double z_scal
        cdef double x_bound, y_bound, z_bound
        cdef double[4][4][4] a
        cdef double[4] x_powers, y_powers, z_powers
        cdef double df_dn

        if order_x > 3:
            raise ValueError('Can\'t get a gradient of order 4 or more in cubic.')
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
        z_bound = self._z[index_z + 1] - self._z[index_z]
        if z_bound != 0:
            z_scal = (pz - self._z[index_z]) / z_bound
        else:
            raise ZeroDivisionError('Two adjacent spline points have the same z value!')

        # Calculate the coefficients (and gradients at each spline point) if they dont exist
        self.cache_coefficients(index_x, index_y, index_z, a)
        x_powers[0] = 1
        x_powers[1] = x_scal
        x_powers[2] = x_scal * x_scal
        x_powers[3] = x_scal * x_scal * x_scal
        y_powers[0] = 1
        y_powers[1] = y_scal
        y_powers[2] = y_scal * y_scal
        y_powers[3] = y_scal * y_scal * y_scal
        z_powers[0] = 1
        z_powers[1] = z_scal
        z_powers[2] = z_scal * z_scal
        z_powers[3] = z_scal * z_scal * z_scal
        df_dn = 0
        for i in range(order_x, 4):
            for j in range(order_y, 4):
                for k in range(order_z, 4):
                    df_dn += (a[i][j][k] * (lookup_factorial(i)/lookup_factorial(i-order_x)) * (lookup_factorial(j)/lookup_factorial(j-order_y))* (lookup_factorial(k)/lookup_factorial(k-order_z)) *
                              x_powers[i-order_x] * y_powers[j-order_y] * z_powers[k-order_z])
        return  df_dn


cdef class _Extrapolator3D:
    """
    Base class for Function3D extrapolators.

    :param x: 1D memory view of the spline point x positions.
    :param y: 1D memory view of the spline point y positions.
    :param z: 1D memory view of the spline point z positions.
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



cdef class _Extrapolator3DNearest(_Extrapolator3D):
    """
    Extrapolator that returns nearest input value.

    :param x: 1D memory view of the spline point x positions.
    :param y: 1D memory view of the spline point y positions.
    :param z: 1D memory view of the spline point y positions.
    :param f: 3D memory view of the function value at spline point x, y, z positions.
    :param external_interpolator: stored _Interpolator2D object that is being used.
    """

    ID = 'nearest'

    def __init__(self, double[::1] x, double[::1] y, double[::1] z, double[:, :, ::1] f, _Interpolator3D external_interpolator, double extrapolation_range_x, double extrapolation_range_y, double extrapolation_range_z):
           super().__init__(x, y, z, f, external_interpolator, extrapolation_range_x, extrapolation_range_y, extrapolation_range_z)

    cdef double evaluate_edge_x(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        return self._external_interpolator.evaluate(self._x[edge_x_index], py, pz, index_x, index_y, index_z)

    cdef double evaluate_edge_y(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        return self._external_interpolator.evaluate(px, self._y[edge_y_index], pz, index_x, index_y, index_z)

    cdef double evaluate_edge_z(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        return self._external_interpolator.evaluate(px, py, self._z[edge_z_index], index_x, index_y, index_z)

    cdef double evaluate_edge_xy(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        return self._external_interpolator.evaluate(self._x[edge_x_index], self._y[edge_y_index], pz, index_x, index_y, index_z)

    cdef double evaluate_edge_xz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        return self._external_interpolator.evaluate(self._x[edge_x_index], py, self._z[edge_z_index], index_x, index_y, index_z)

    cdef double evaluate_edge_yz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        return self._external_interpolator.evaluate(px, self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z)

    cdef double evaluate_edge_xyz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        return self._external_interpolator.evaluate(self._x[edge_x_index], self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z)


cdef class _Extrapolator3DLinear(_Extrapolator3D):
    """
    Extrapolator that returns linearly extrapolated input value.

    :param x: 1D memory view of the spline point x positions.
    :param y: 1D memory view of the spline point y positions.
    :param z: 1D memory view of the spline point z positions.
    :param f: 3D memory view of the function value at spline point x, y, z positions.
    :param external_interpolator: stored _Interpolator3D object that is being used.
    """

    ID = 'linear'

    def __init__(self, double[::1] x, double[::1] y, double[::1] z, double[:, :, ::1] f, _Interpolator3D external_interpolator, double extrapolation_range_x, double extrapolation_range_y, double extrapolation_range_z):
           super().__init__(x, y, z, f, external_interpolator, extrapolation_range_x, extrapolation_range_y, extrapolation_range_z)

    cdef double evaluate_edge_x(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        cdef double f_value, fx_value
        f_value = self._external_interpolator.evaluate(self._x[edge_x_index], py, pz, index_x, index_y, index_z)
        fx_value = self._external_interpolator.analytic_gradient(self._x[edge_x_index], py, pz, index_x, index_y, index_z, 1, 0, 0) / \
                   (self._x[index_x + 1] - self._x[index_x])
        return f_value + fx_value * (px - self._x[edge_x_index])

    cdef double evaluate_edge_y(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        cdef double f_value, fy_value
        f_value = self._external_interpolator.evaluate(px, self._y[edge_y_index], pz, index_x, index_y, index_z)
        fy_value = self._external_interpolator.analytic_gradient(px, self._y[edge_y_index], pz, index_x, index_y, index_z, 0, 1, 0) / \
                   (self._y[index_y + 1] - self._y[index_y])
        return f_value + fy_value * (py - self._y[edge_y_index])

    cdef double evaluate_edge_z(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        cdef double f_value, fz_value
        f_value = self._external_interpolator.evaluate(px, py, self._z[edge_z_index], index_x, index_y, index_z)
        fz_value = self._external_interpolator.analytic_gradient(px, py, self._z[edge_z_index], index_x, index_y, index_z, 0, 0, 1) / \
                   (self._z[index_z + 1] - self._z[index_z])
        return f_value + fz_value * (pz - self._z[edge_z_index])

    cdef double evaluate_edge_xy(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        cdef double f_value, fx_value, fy_value, fxy_value
        f_value = self._external_interpolator.evaluate(self._x[edge_x_index], self._y[edge_y_index], pz, index_x, index_y, index_z)
        fx_value = self._external_interpolator.analytic_gradient(
            self._x[edge_x_index], self._y[edge_y_index], pz, index_x, index_y, index_z, 1, 0, 0
        )/(self._x[index_x + 1] - self._x[index_x])
        fy_value = self._external_interpolator.analytic_gradient(
            self._x[edge_x_index], self._y[edge_y_index], pz, index_x, index_y, index_z, 0, 1, 0
        )/(self._y[index_y + 1] - self._y[index_y])
        fxy_value = self._external_interpolator.analytic_gradient(
            self._x[edge_x_index], self._y[edge_y_index], pz, index_x, index_y, index_z, 1, 1, 0
        )/((self._x[index_x + 1] - self._x[index_x])*(self._y[index_y + 1] - self._y[index_y]))
        return f_value + fx_value * (px - self._x[edge_x_index]) + fy_value * (py - self._y[edge_y_index]) + fxy_value* (py - self._y[edge_y_index])* (px - self._x[edge_x_index])

    cdef double evaluate_edge_xz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        cdef double f_value, fx_value, fz_value, fxz_value
        f_value = self._external_interpolator.evaluate(self._x[edge_x_index], py, self._z[edge_z_index], index_x, index_y, index_z)
        fx_value = self._external_interpolator.analytic_gradient(
            self._x[edge_x_index], py, self._z[edge_z_index], index_x, index_y, index_z, 1, 0, 0
        )/(self._x[index_x + 1] - self._x[index_x])
        fz_value = self._external_interpolator.analytic_gradient(
            self._x[edge_x_index], py, self._z[edge_z_index], index_x, index_y, index_z, 0, 0, 1
        )/(self._z[index_z + 1] - self._z[index_z])
        fxz_value = self._external_interpolator.analytic_gradient(
            self._x[edge_x_index], py, self._z[edge_z_index], index_x, index_y, index_z, 1, 0, 1
        )/((self._x[index_x + 1] - self._x[index_x])*(self._z[index_z + 1] - self._z[index_z]))
        return f_value + fx_value * (px - self._x[edge_x_index]) + fz_value * (pz - self._z[edge_z_index]) + fxz_value* (pz - self._z[edge_z_index])* (px - self._x[edge_x_index])

    cdef double evaluate_edge_yz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        cdef double f_value, fy_value, fz_value, fyz_value
        f_value = self._external_interpolator.evaluate(px, self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z)
        fy_value = self._external_interpolator.analytic_gradient(
            px, self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z, 0, 1, 0
        )/(self._y[index_y + 1] - self._y[index_y])
        fz_value = self._external_interpolator.analytic_gradient(
            px, self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z, 0, 0, 1
        )/(self._z[index_z + 1] - self._z[index_z])
        fyz_value = self._external_interpolator.analytic_gradient(
            px, self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z, 0, 1, 1
        )/((self._y[index_y + 1] - self._y[index_y])*(self._z[index_z + 1] - self._z[index_z]))
        return f_value + fy_value * (py - self._y[edge_y_index]) + fz_value * (pz - self._z[edge_z_index]) + fyz_value* (pz - self._z[edge_z_index])* (py - self._y[edge_y_index])

    cdef double evaluate_edge_xyz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        """
        Extrapolate beyond the spline knot domain in the x, y and z directions.

        The extrapolated value uses the closest value of the function at the edge in x and y, df/dx, df/dy, df/dz, 
        d2f/dxdy, d2f/dxdz, d2f/dydz  
        of the interpolator at the edge of the spline knot grid to extrapolate as 
        f_extrap = f(edge) + Dx*df(edge)/dx + Dy*df(edge)/dy + Dx*Dy*d2f(edge)/dxdy + ... where 
        Dx = px - edge_x, Dy = py - edge_y... This is because the trilinear equation f(x, y, z) as a taylor expansion 
        only has terms with x, y and z to a maximum power of 1 in every term. All derivatives are un-normalised before 
        extrapolation.

        :param double px: the point for which an interpolated value is required.
        :param double py: the point for which an interpolated value is required.
        :param double pz: the point for which an interpolated value is required.
        :param int index_x: the lower index of the bin containing point px. (Result of bisection search).   
        :param int index_y: the lower index of the bin containing point py. (Result of bisection search).   
        :param int index_z: the lower index of the bin containing point pz. (Result of bisection search).   
        :param int edge_x_index: the index of the closest edge spline knot in the x direction.
        :param int edge_y_index: the index of the closest edge spline knot in the y direction.
        :param int edge_z_index: the index of the closest edge spline knot in the z direction.

        """
        cdef double f_value, fx_value, fy_value, fz_value, fxy_value, fxz_value, fyz_value, fxyz_value
        f_value = self._external_interpolator.evaluate(self._x[edge_x_index], self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z)
        fx_value = self._external_interpolator.analytic_gradient(
            self._x[edge_x_index], self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z, 1, 0, 0
        )/(self._x[index_x + 1] - self._x[index_x])
        fy_value = self._external_interpolator.analytic_gradient(
            self._x[edge_x_index], self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z, 0, 1, 0
        )/(self._y[index_y + 1] - self._y[index_y])
        fz_value = self._external_interpolator.analytic_gradient(
            self._x[edge_x_index], self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z, 0, 0, 1
        )/(self._z[index_z + 1] - self._z[index_z])
        fxy_value = self._external_interpolator.analytic_gradient(
            self._x[edge_x_index], self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z, 1, 1, 0
        )/((self._y[index_y + 1] - self._y[index_y])*(self._x[index_x + 1] - self._x[index_x]))
        fxz_value = self._external_interpolator.analytic_gradient(
            self._x[edge_x_index], self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z, 1, 0, 1
        )/((self._x[index_x + 1] - self._x[index_x])*(self._z[index_z + 1] - self._z[index_z]))
        fyz_value = self._external_interpolator.analytic_gradient(
            self._x[edge_x_index], self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z, 0, 1, 1
        )/((self._y[index_y + 1] - self._y[index_y])*(self._z[index_z + 1] - self._z[index_z]))
        fxyz_value = self._external_interpolator.analytic_gradient(
            self._x[edge_x_index], self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z, 1, 1, 1
        )/((self._x[index_x + 1] - self._x[index_x])*(self._y[index_y + 1] - self._y[index_y])*(self._z[index_z + 1] - self._z[index_z]))
        return f_value + fx_value * (px - self._x[edge_x_index]) + fy_value * (py - self._y[edge_y_index]) + \
               fz_value * (pz - self._z[edge_z_index]) \
               + fxy_value* (px - self._x[edge_x_index])* (py - self._y[edge_y_index]) \
               + fxz_value* (pz - self._z[edge_z_index])* (px - self._x[edge_x_index]) \
               + fyz_value* (pz - self._z[edge_z_index])* (py - self._y[edge_y_index]) \
               + fxyz_value* (px - self._x[edge_x_index])* (py - self._y[edge_y_index])* (pz - self._z[edge_z_index])


cdef class _ArrayDerivative3D:
    """
    Gradient method that returns the approximate derivative of a desired order at a specified grid point.

    These methods of finding derivatives are only valid on a 3D grid of points, at the values at the points. Other
    derivative methods would be dependent on the interpolator types.

    :param x: 1D memory view of the spline point x positions.
    :param y: 1D memory view of the spline point y positions.
    :param z: 1D memory view of the spline point z positions.
    :param f: 3D memory view of the function value at spline point x, y, z positions.
    """
    def __init__(self, double[::1] x, double[::1] y, double[::1] z, double[:, :, ::1] f):

        self._x = x
        self._y = y
        self._z = z
        self._f = f
        self._last_index_x = self._x.shape[0] - 1
        self._last_index_y = self._y.shape[0] - 1
        self._last_index_z = self._z.shape[0] - 1

    cdef double evaluate(self, int index_x, int index_y, int index_z, int derivative_order_x, int derivative_order_y, int derivative_order_z, bint rescale_norm_x, bint rescale_norm_y, bint rescale_norm_z) except? -1e999:
        """
        Evaluate the derivative of specific order at a grid point.

        The grid of spline knots is reduced to a 2X2X2 to 3X3X3 grid for gradient evaluation depending on if the requested
        derivative is near the edge or not. If near the edge in 1 dimension, grid size is 2X3X3 or a combination for
        example.

        :param index_x: The lower index of the x grid cell to evaluate.
        :param index_y: The lower index of the y grid cell to evaluate.
        :param index_z: The lower index of the z grid cell to evaluate.
        :param derivative_order_x: An integer of the derivative order x. Only zero if derivative_order_y/z is nonzero.
        :param derivative_order_y: An integer of the derivative order y. Only zero if derivative_order_x/z is nonzero.
        :param derivative_order_z: An integer of the derivative order z. Only zero if derivative_order_x/y is nonzero.
        :param rescale_norm_x: A boolean as whether to rescale to the delta before x[index_x] or after (default).
        :param rescale_norm_y: A boolean as whether to rescale to the delta before y[index_y] or after (default).
        :param rescale_norm_z: A boolean as whether to rescale to the delta before z[index_z] or after (default).
        """
        # Find if at the edge of the grid, and in what direction. Then evaluate the gradient.
        cdef double dfdn
        cdef int x_centre_add, y_centre_add, z_centre_add
        cdef int index_x_input, index_y_input, index_z_input

        # If at the upper edge of the array, the index of the point to find needs to be at the upper edge.
        if index_x == self._last_index_x:
            x_centre_add = 1
            index_x_input = index_x - 1
        else:
            x_centre_add = 0
            index_x_input = index_x
        if index_y == self._last_index_y:
            y_centre_add = 1
            index_y_input = index_y - 1
        else:
            y_centre_add = 0
            index_y_input = index_y
        if index_z == self._last_index_z:
            z_centre_add = 1
            index_z_input = index_z - 1
        else:
            z_centre_add = 0
            index_z_input = index_z

        if index_x == 0 or index_x == self._last_index_x:
            if index_y == 0 or index_y == self._last_index_y:
                if index_z == 0 or index_z == self._last_index_z:
                    dfdn = self.eval_edge_xyz(
                        index_x_input, index_y_input, index_z_input, derivative_order_x, derivative_order_y, derivative_order_z,
                        x_centre_add=x_centre_add, y_centre_add=y_centre_add, z_centre_add=z_centre_add
                    )
                else:
                    dfdn = self.eval_edge_xy(
                        index_x_input, index_y_input, index_z_input, derivative_order_x, derivative_order_y, derivative_order_z,
                        x_centre_add=x_centre_add, y_centre_add=y_centre_add, z_centre_add=z_centre_add
                    )
            else:
                if index_z == 0 or index_z == self._last_index_z:
                    dfdn = self.eval_edge_xz(
                        index_x_input, index_y_input, index_z_input, derivative_order_x, derivative_order_y, derivative_order_z,
                        x_centre_add=x_centre_add, y_centre_add=y_centre_add, z_centre_add=z_centre_add
                    )
                else:
                    dfdn = self.eval_edge_x(
                        index_x_input, index_y_input, index_z_input, derivative_order_x, derivative_order_y, derivative_order_z,
                        x_centre_add=x_centre_add, y_centre_add=y_centre_add, z_centre_add=z_centre_add
                    )
        else:
            if index_y == 0 or index_y == self._last_index_y:
                if index_z == 0 or index_z == self._last_index_z:
                    dfdn = self.eval_edge_yz(
                        index_x_input, index_y_input, index_z_input, derivative_order_x, derivative_order_y, derivative_order_z,
                        x_centre_add=x_centre_add, y_centre_add=y_centre_add, z_centre_add=z_centre_add
                    )
                else:
                    dfdn = self.eval_edge_y(
                        index_x_input, index_y_input, index_z_input, derivative_order_x, derivative_order_y, derivative_order_z,
                        x_centre_add=x_centre_add, y_centre_add=y_centre_add, z_centre_add=z_centre_add
                    )
            else:
                if index_z == 0 or index_z == self._last_index_z:
                    dfdn = self.eval_edge_z(
                        index_x_input, index_y_input, index_z_input, derivative_order_x, derivative_order_y, derivative_order_z,
                        x_centre_add=x_centre_add, y_centre_add=y_centre_add, z_centre_add=z_centre_add
                    )
                else:
                    dfdn = self.eval_xyz(
                        index_x_input, index_y_input, index_z_input, derivative_order_x, derivative_order_y, derivative_order_z,
                    )
        if rescale_norm_x:
            if not (index_x == 0 or index_x == self._last_index_x):
                for i in range(derivative_order_x):
                    dfdn = rescale_lower_normalisation(dfdn,  self._x[index_x - 1], self._x[index_x], self._x[index_x + 1])
        if rescale_norm_y:
            if not (index_y == 0 or index_y == self._last_index_y):
                for i in range(derivative_order_y):
                    dfdn = rescale_lower_normalisation(dfdn,  self._y[index_y - 1], self._y[index_y], self._y[index_y + 1])
        if rescale_norm_z:
            if not (index_z == 0 or index_z == self._last_index_z):
                for i in range(derivative_order_z):
                    dfdn = rescale_lower_normalisation(dfdn,  self._z[index_z - 1], self._z[index_z], self._z[index_z + 1])
        return dfdn

    cdef double eval_edge_x(self, int index_x, int index_y, int index_z, int derivative_order_x, int derivative_order_y, int derivative_order_z, int x_centre_add, int y_centre_add, int z_centre_add):
        cdef double dfdn
        cdef double[:] x_range, y_range, z_range
        cdef double[:, :, :] f_range
        cdef int x_centre = 0, y_centre = 1, z_centre = 1
        x_range = self._x[index_x:index_x + 2]
        y_range = self._y[index_y - 1:index_y + 2]
        z_range = self._z[index_z - 1:index_z + 2]
        f_range = self._f[index_x:index_x + 2, index_y - 1:index_y + 2, index_z - 1:index_z + 2]
        dfdn = 0
        if derivative_order_x == 1 and derivative_order_y == 0 and derivative_order_z == 0:
            dfdn = self.derivitive_dfdx_edge(f_range[:, y_centre + y_centre_add, z_centre + z_centre_add])
        elif derivative_order_x == 0 and derivative_order_y == 1 and derivative_order_z == 0:
            dfdn = self.derivitive_dfdx(y_range, f_range[x_centre + x_centre_add, :, z_centre + z_centre_add])
        elif derivative_order_x == 0 and derivative_order_y == 0 and derivative_order_z == 1:
            dfdn = self.derivitive_dfdx(z_range, f_range[x_centre + x_centre_add, y_centre + y_centre_add, :])
        elif derivative_order_x == 1 and derivative_order_y == 1 and derivative_order_z == 0:
            dfdn = self.derivitive_d2fdxdy_edge_x(y_range, f_range[:, :, z_centre + z_centre_add])
        elif derivative_order_x == 1 and derivative_order_y == 0 and derivative_order_z == 1:
            dfdn = self.derivitive_d2fdxdy_edge_x(z_range, f_range[:, y_centre + y_centre_add, :])
        elif derivative_order_x == 0 and derivative_order_y == 1 and derivative_order_z == 1:
            dfdn = self.derivitive_d2fdxdy(y_range, z_range, f_range[x_centre + x_centre_add, :, :])
        elif derivative_order_x == 1 and derivative_order_y == 1 and derivative_order_z == 1:
            dfdn = self.derivitive_d3fdxdydz_edge_x(y_range, z_range, f_range)
        else:
            raise ValueError('No higher order derivatives implemented')
        return dfdn

    cdef double eval_edge_y(self, int index_x, int index_y, int index_z, int derivative_order_x, int derivative_order_y, int derivative_order_z, int x_centre_add, int y_centre_add, int z_centre_add):
        cdef double dfdn
        cdef double[:] x_range, y_range, z_range
        cdef double[:, :, :] f_range
        cdef int x_centre = 1, y_centre = 0, z_centre = 1
        x_range = self._x[index_x - 1:index_x + 2]
        y_range = self._y[index_y:index_y + 2]
        z_range = self._z[index_z - 1:index_z + 2]
        f_range = self._f[index_x - 1:index_x + 2, index_y:index_y + 2, index_z - 1:index_z + 2]
        dfdn = 0
        if derivative_order_x == 1 and derivative_order_y == 0 and derivative_order_z == 0:
            dfdn = self.derivitive_dfdx(x_range, f_range[:, y_centre + y_centre_add, z_centre + z_centre_add])
        elif derivative_order_x == 0 and derivative_order_y == 1 and derivative_order_z == 0:
            dfdn = self.derivitive_dfdx_edge(f_range[x_centre + x_centre_add, :, z_centre + z_centre_add])
        elif derivative_order_x == 0 and derivative_order_y == 0 and derivative_order_z == 1:
            dfdn = self.derivitive_dfdx(z_range, f_range[x_centre + x_centre_add, y_centre + y_centre_add, :])
        elif derivative_order_x == 1 and derivative_order_y == 1 and derivative_order_z == 0:
            # fixme ALl these functions don't specify 'f' as c contiguous. Can't input this if f_range is cdef double[:, :, ::1] f_range
            dfdn = self.derivitive_d2fdxdy_edge_y(x_range, f_range[:, :, z_centre + z_centre_add])
        elif derivative_order_x == 1 and derivative_order_y == 0 and derivative_order_z == 1:
            dfdn = self.derivitive_d2fdxdy(x_range, z_range, f_range[:, y_centre + y_centre_add, :])
        elif derivative_order_x == 0 and derivative_order_y == 1 and derivative_order_z == 1:
            dfdn = self.derivitive_d2fdxdy_edge_x(z_range, f_range[x_centre + x_centre_add, :, :])# This should work still, edge is in second variable
        elif derivative_order_x == 1 and derivative_order_y == 1 and derivative_order_z == 1:
            dfdn = self.derivitive_d3fdxdydz_edge_y(x_range, z_range, f_range)
        else:
            raise ValueError('No higher order derivatives implemented')
        return dfdn

    cdef double eval_edge_z(self, int index_x, int index_y, int index_z, int derivative_order_x, int derivative_order_y, int derivative_order_z, int x_centre_add, int y_centre_add, int z_centre_add) except? -1e999:
        cdef double dfdn
        cdef double[:] x_range, y_range, z_range
        cdef double[:, :, :] f_range
        cdef int x_centre = 1, y_centre = 1, z_centre = 0
        x_range = self._x[index_x - 1:index_x + 2]
        y_range = self._y[index_y - 1:index_y + 2]
        z_range = self._z[index_z:index_z + 2]
        f_range = self._f[index_x - 1:index_x + 2, index_y - 1:index_y + 2, index_z:index_z + 2]
        dfdn = 0
        if derivative_order_x == 1 and derivative_order_y == 0 and derivative_order_z == 0:
            dfdn = self.derivitive_dfdx(x_range, f_range[:, y_centre + y_centre_add, z_centre + z_centre_add])
        elif derivative_order_x == 0 and derivative_order_y == 1 and derivative_order_z == 0:
            dfdn = self.derivitive_dfdx(y_range, f_range[x_centre + x_centre_add, :, z_centre + z_centre_add])
        elif derivative_order_x == 0 and derivative_order_y == 0 and derivative_order_z == 1:
            dfdn = self.derivitive_dfdx_edge(f_range[x_centre + x_centre_add, y_centre + y_centre_add, :])
        elif derivative_order_x == 1 and derivative_order_y == 1 and derivative_order_z == 0:
            dfdn = self.derivitive_d2fdxdy(x_range, y_range, f_range[:, :, z_centre + z_centre_add])
        elif derivative_order_x == 1 and derivative_order_y == 0 and derivative_order_z == 1:
            dfdn = self.derivitive_d2fdxdy_edge_y(x_range, f_range[:, y_centre + y_centre_add, :]) # This should work still, edge is in second variable
        elif derivative_order_x == 0 and derivative_order_y == 1 and derivative_order_z == 1:
            dfdn = self.derivitive_d2fdxdy_edge_y(y_range, f_range[x_centre + x_centre_add, :, :]) # This should work still, edge is in second variable
        elif derivative_order_x == 1 and derivative_order_y == 1 and derivative_order_z == 1:
            dfdn = self.derivitive_d3fdxdydz_edge_z(x_range, y_range, f_range)
        else:
            raise ValueError('No higher order derivatives implemented')
        return dfdn

    cdef double eval_edge_xy(self, int index_x, int index_y, int index_z, int derivative_order_x, int derivative_order_y, int derivative_order_z, int x_centre_add, int y_centre_add, int z_centre_add) except? -1e999:
        cdef double dfdn
        cdef double[:] x_range, y_range, z_range
        cdef double[:, :, :] f_range
        cdef int x_centre = 0, y_centre = 0, z_centre = 1
        x_range = self._x[index_x:index_x + 2]
        y_range = self._y[index_y:index_y + 2]
        z_range = self._z[index_z - 1:index_z + 2]
        f_range = self._f[index_x:index_x + 2, index_y:index_y + 2, index_z - 1:index_z + 2]
        dfdn = 0
        if derivative_order_x == 1 and derivative_order_y == 0 and derivative_order_z == 0:
            dfdn = self.derivitive_dfdx_edge(f_range[:, y_centre + y_centre_add, z_centre + z_centre_add])
        elif derivative_order_x == 0 and derivative_order_y == 1 and derivative_order_z == 0:
            dfdn = self.derivitive_dfdx_edge(f_range[x_centre + x_centre_add, :, z_centre + z_centre_add])
        elif derivative_order_x == 0 and derivative_order_y == 0 and derivative_order_z == 1:
            dfdn = self.derivitive_dfdx(z_range, f_range[x_centre + x_centre_add, y_centre + y_centre_add, :])
        elif derivative_order_x == 1 and derivative_order_y == 1 and derivative_order_z == 0:
            dfdn = self.derivitive_d2fdxdy_edge_xy(x_range, y_range, f_range[:, :, z_centre + z_centre_add])
        elif derivative_order_x == 1 and derivative_order_y == 0 and derivative_order_z == 1:
            dfdn = self.derivitive_d2fdxdy_edge_x(z_range, f_range[:, y_centre + y_centre_add, :])
        elif derivative_order_x == 0 and derivative_order_y == 1 and derivative_order_z == 1:
            dfdn = self.derivitive_d2fdxdy_edge_x(z_range, f_range[x_centre + x_centre_add, :, :])
        elif derivative_order_x == 1 and derivative_order_y == 1 and derivative_order_z == 1:
            dfdn = self.derivitive_d3fdxdydz_edge_xy(z_range, f_range)
        else:
            raise ValueError('No higher order derivatives implemented')
        return dfdn

    cdef double eval_edge_xz(self, int index_x, int index_y, int index_z, int derivative_order_x, int derivative_order_y, int derivative_order_z, int x_centre_add, int y_centre_add, int z_centre_add) except? -1e999:
        cdef double dfdn
        cdef double[:] x_range, y_range, z_range
        cdef double[:, :, :] f_range
        cdef int x_centre = 0, y_centre = 1, z_centre = 0
        x_range = self._x[index_x:index_x + 2]
        y_range = self._y[index_y - 1:index_y + 2]
        z_range = self._z[index_z:index_z + 2]
        f_range = self._f[index_x:index_x + 2, index_y - 1:index_y + 2, index_z:index_z + 2]
        dfdn = 0
        if derivative_order_x == 1 and derivative_order_y == 0 and derivative_order_z == 0:
            dfdn = self.derivitive_dfdx_edge(f_range[:, y_centre + y_centre_add, z_centre + z_centre_add])
        elif derivative_order_x == 0 and derivative_order_y == 1 and derivative_order_z == 0:
            dfdn = self.derivitive_dfdx(y_range, f_range[x_centre + x_centre_add, :, z_centre + z_centre_add])
        elif derivative_order_x == 0 and derivative_order_y == 0 and derivative_order_z == 1:
            dfdn = self.derivitive_dfdx_edge(f_range[x_centre + x_centre_add, y_centre + y_centre_add, :])
        elif derivative_order_x == 1 and derivative_order_y == 1 and derivative_order_z == 0:
            dfdn = self.derivitive_d2fdxdy_edge_x(y_range, f_range[:, :, z_centre + z_centre_add])
        elif derivative_order_x == 1 and derivative_order_y == 0 and derivative_order_z == 1:
            dfdn = self.derivitive_d2fdxdy_edge_xy(x_range, z_range, f_range[:, y_centre + y_centre_add, :])
        elif derivative_order_x == 0 and derivative_order_y == 1 and derivative_order_z == 1:
            dfdn = self.derivitive_d2fdxdy_edge_y(y_range, f_range[x_centre + x_centre_add, :, :])
        elif derivative_order_x == 1 and derivative_order_y == 1 and derivative_order_z == 1:
            dfdn = self.derivitive_d3fdxdydz_edge_xz(y_range, f_range)
        else:
            raise ValueError('No higher order derivatives implemented')
        return dfdn

    cdef double eval_edge_yz(self, int index_x, int index_y, int index_z, int derivative_order_x, int derivative_order_y, int derivative_order_z, int x_centre_add, int y_centre_add, int z_centre_add) except? -1e999:
        cdef double dfdn
        cdef double[:] x_range, y_range, z_range
        cdef double[:, :, :] f_range
        cdef int x_centre = 1, y_centre = 0, z_centre = 0
        x_range = self._x[index_x - 1:index_x + 2]
        y_range = self._y[index_y:index_y + 2]
        z_range = self._z[index_z:index_z + 2]
        f_range = self._f[index_x - 1:index_x + 2, index_y:index_y + 2, index_z:index_z + 2]
        dfdn = 0
        if derivative_order_x == 1 and derivative_order_y == 0 and derivative_order_z == 0:
            dfdn = self.derivitive_dfdx(x_range, f_range[:, y_centre + y_centre_add, z_centre + z_centre_add])
        elif derivative_order_x == 0 and derivative_order_y == 1 and derivative_order_z == 0:
            dfdn = self.derivitive_dfdx_edge(f_range[x_centre + x_centre_add, :, z_centre + z_centre_add])
        elif derivative_order_x == 0 and derivative_order_y == 0 and derivative_order_z == 1:
            dfdn = self.derivitive_dfdx_edge(f_range[x_centre + x_centre_add, y_centre + y_centre_add, :])
        elif derivative_order_x == 1 and derivative_order_y == 1 and derivative_order_z == 0:
            dfdn = self.derivitive_d2fdxdy_edge_y(x_range, f_range[:, :, z_centre + z_centre_add])
        elif derivative_order_x == 1 and derivative_order_y == 0 and derivative_order_z == 1:
            dfdn = self.derivitive_d2fdxdy_edge_y(x_range, f_range[:, y_centre + y_centre_add, :])
        elif derivative_order_x == 0 and derivative_order_y == 1 and derivative_order_z == 1:
            dfdn = self.derivitive_d2fdxdy_edge_xy(y_range, z_range, f_range[x_centre + x_centre_add, :, :])
        elif derivative_order_x == 1 and derivative_order_y == 1 and derivative_order_z == 1:
            dfdn = self.derivitive_d3fdxdydz_edge_yz(x_range, f_range)
        else:
            raise ValueError('No higher order derivatives implemented')
        return dfdn

    cdef double eval_edge_xyz(self, int index_x, int index_y, int index_z, int derivative_order_x, int derivative_order_y, int derivative_order_z, int x_centre_add, int y_centre_add, int z_centre_add) except? -1e999:
        cdef double dfdn
        cdef double[:] x_range, y_range, z_range
        cdef double[:, :, :] f_range
        cdef int x_centre = 0, y_centre = 0, z_centre = 0
        x_range = self._x[index_x:index_x + 2]
        y_range = self._y[index_y:index_y + 2]
        z_range = self._z[index_z:index_z + 2]
        f_range = self._f[index_x:index_x + 2, index_y:index_y + 2, index_z:index_z + 2]
        dfdn = 0
        if derivative_order_x == 1 and derivative_order_y == 0 and derivative_order_z == 0:
            dfdn = self.derivitive_dfdx_edge(f_range[:, y_centre + y_centre_add, z_centre + z_centre_add])
        elif derivative_order_x == 0 and derivative_order_y == 1 and derivative_order_z == 0:
            dfdn = self.derivitive_dfdx_edge(f_range[x_centre + x_centre_add, :, z_centre + z_centre_add])
        elif derivative_order_x == 0 and derivative_order_y == 0 and derivative_order_z == 1:
            dfdn = self.derivitive_dfdx_edge(f_range[x_centre + x_centre_add, y_centre + y_centre_add, :])
        elif derivative_order_x == 1 and derivative_order_y == 1 and derivative_order_z == 0:
            dfdn = self.derivitive_d2fdxdy_edge_xy(x_range, y_range, f_range[:, :, z_centre + z_centre_add])
        elif derivative_order_x == 1 and derivative_order_y == 0 and derivative_order_z == 1:
            dfdn = self.derivitive_d2fdxdy_edge_xy(x_range, z_range, f_range[:, y_centre + y_centre_add, :])
        elif derivative_order_x == 0 and derivative_order_y == 1 and derivative_order_z == 1:
            dfdn = self.derivitive_d2fdxdy_edge_xy(y_range, z_range, f_range[x_centre + x_centre_add, :, :])
        elif derivative_order_x == 1 and derivative_order_y == 1 and derivative_order_z == 1:
            dfdn = self.derivitive_d3fdxdydz_edge_xyz(f_range)
        else:
            raise ValueError('No higher order derivatives implemented')
        return dfdn

    cdef double eval_xyz(self, int index_x, int index_y, int index_z, int derivative_order_x, int derivative_order_y, int derivative_order_z):
        cdef double dfdn
        cdef double[:] x_range, y_range, z_range
        cdef double[:, :, :] f_range
        cdef int x_centre = 1, y_centre = 1, z_centre = 1
        x_range = self._x[index_x - 1:index_x + 2]
        y_range = self._y[index_y - 1:index_y + 2]
        z_range = self._z[index_z - 1:index_z + 2]
        f_range = self._f[index_x - 1:index_x + 2, index_y - 1:index_y + 2, index_z - 1:index_z + 2]
        dfdn = 0
        if derivative_order_x == 1 and derivative_order_y == 0 and derivative_order_z == 0:
            dfdn = self.derivitive_dfdx(x_range, f_range[:, y_centre, z_centre])
        elif derivative_order_x == 0 and derivative_order_y == 1 and derivative_order_z == 0:
            dfdn = self.derivitive_dfdx(y_range, f_range[x_centre, :, z_centre])
        elif derivative_order_x == 0 and derivative_order_y == 0 and derivative_order_z == 1:
            dfdn = self.derivitive_dfdx(z_range, f_range[x_centre, y_centre, :])
        elif derivative_order_x == 1 and derivative_order_y == 1 and derivative_order_z == 0:
            dfdn = self.derivitive_d2fdxdy(x_range, y_range, f_range[:, :, z_centre])
        elif derivative_order_x == 1 and derivative_order_y == 0 and derivative_order_z == 1:
            dfdn = self.derivitive_d2fdxdy(x_range, z_range, f_range[:, y_centre, :])
        elif derivative_order_x == 0 and derivative_order_y == 1 and derivative_order_z == 1:
            dfdn = self.derivitive_d2fdxdy(y_range, z_range, f_range[x_centre, :, :])
        elif derivative_order_x == 1 and derivative_order_y == 1 and derivative_order_z == 1:
            dfdn = self.derivitive_d3fdxdydz(x_range, y_range, z_range, f_range)
        else:
            raise ValueError('No higher order derivatives implemented')
        return dfdn


    cdef double derivitive_dfdx(self, double[:] x, double[:] f) except? -1e999:
        cdef double x1_n, x1_n2
        x1_n = (x[1] - x[0]) / (x[2] - x[1])
        x1_n2 = x1_n ** 2
        return (f[2] * x1_n2 - f[0] - f[1] * (x1_n2 - 1.)) / (x1_n + x1_n2)

    cdef double derivitive_dfdx_edge(self, double[:] f):
        return f[1] - f[0]

    cdef double derivitive_d2fdxdy(self,  double[:] x, double[:] y, double[:, :] f) except? -1e999:
        cdef double dx1, dy1
        dx1 = (x[1] - x[0]) / (x[2] - x[1])
        dy1 = (y[1] - y[0]) / (y[2] - y[1])
        return (f[2, 2] - f[0, 2] - f[2, 0] + f[0, 0]) / (1. + dx1 + dy1 + dx1 * dy1)

    cdef double derivitive_d2fdxdy_edge_xy(self, double[:] x, double[:] y, double[:, :] f) except? -1e999:
        return f[1, 1] - f[0, 1] - f[1, 0] + f[0, 0]

    cdef double derivitive_d2fdxdy_edge_x(self, double[:] y, double[:, :] f) except? -1e999:
        cdef double dy1
        dy1 = (y[1] - y[0]) / (y[2] - y[1])
        return (f[1, 2] - f[0, 2] - f[1, 0] + f[0, 0]) / (1. + dy1)

    cdef double derivitive_d2fdxdy_edge_y(self, double[:] x, double[:, :] f) except? -1e999:
        # fixme ALl these functions don't specify 'f' as c contiguous. Can't input this if f_range is cdef double[:, :, ::1] f_range slice as f_range[:, :, n]
        cdef double dx1
        dx1 = (x[1] - x[0]) / (x[2] - x[1])
        return (f[2, 1] - f[0, 1] - f[2, 0] + f[0, 0]) / (1. + dx1)

    cdef double derivitive_d3fdxdydz(self, double[:] x, double[:] y, double[:] z, double[:, :, :] f) except? -1e999:
        cdef double dx1, dy1, dz1
        dx1 = (x[1] - x[0]) / (x[2] - x[1])
        dy1 = (y[1] - y[0]) / (y[2] - y[1])
        dz1 = (z[1] - z[0]) / (z[2] - z[1])
        return (f[2, 2, 2] - f[0, 2, 2] - f[2, 0, 2] + f[0, 0, 2] - f[2, 2, 0] + f[0, 2, 0] + f[2, 0, 0] - f[0, 0, 0]) / (1. + dx1 + dy1 + dz1 + dx1 * dy1 + dx1 * dz1 + dy1 * dz1 + dx1 * dy1 * dz1)

    cdef double derivitive_d3fdxdydz_edge_x(self, double[:] y, double[:] z, double[:, :, :] f) except? -1e999:
        cdef double dy1, dz1
        dy1 = (y[1] - y[0]) / (y[2] - y[1])
        dz1 = (z[1] - z[0]) / (z[2] - z[1])
        return (f[1, 2, 2] - f[0, 2, 2] - f[1, 0, 2] + f[0, 0, 2] - f[1, 2, 0] + f[0, 2, 0] + f[1, 0, 0] - f[0, 0, 0]) / (1. + dy1 + dz1 + dy1 * dz1)

    cdef double derivitive_d3fdxdydz_edge_y(self, double[:] x, double[:] z, double[:, :, :] f) except? -1e999:
        cdef double dx1, dz1
        dx1 = (x[1] - x[0]) / (x[2] - x[1])
        dz1 = (z[1] - z[0]) / (z[2] - z[1])
        return (f[2, 1, 2] - f[0, 1, 2] - f[2, 0, 2] + f[0, 0, 2] - f[2, 1, 0] + f[0, 1, 0] + f[2, 0, 0] - f[0, 0, 0]) / (1. + dx1 + dz1 + dx1 * dz1)

    cdef double derivitive_d3fdxdydz_edge_z(self, double[:] x, double[:] y, double[:, :, :] f) except? -1e999:
        cdef double dx1, dy1
        dx1 = (x[1] - x[0]) / (x[2] - x[1])
        dy1 = (y[1] - y[0]) / (y[2] - y[1])
        return (f[2, 2, 1] - f[0, 2, 1] - f[2, 0, 1] + f[0, 0, 1] - f[2, 2, 0] + f[0, 2, 0] + f[2, 0, 0] - f[0, 0, 0]) / (1. + dx1 + dy1 + dx1 * dy1)

    cdef double derivitive_d3fdxdydz_edge_xy(self, double[:] z, double[:, :, :] f) except? -1e999:
        cdef double dz1
        dz1 = (z[1] - z[0]) / (z[2] - z[1])
        return (f[1, 1, 2] - f[0, 1, 2] - f[1, 0, 2] + f[0, 0, 2] - f[1, 1, 0] + f[0, 1, 0] + f[1, 0, 0] - f[0, 0, 0]) / (1. + dz1)

    cdef double derivitive_d3fdxdydz_edge_xz(self, double[:] y, double[:, :, :] f) except? -1e999:
        cdef double dy1
        dy1 = (y[1] - y[0]) / (y[2] - y[1])
        return (f[1, 2, 1] - f[0, 2, 1] - f[1, 0, 1] + f[0, 0, 1] - f[1, 2, 0] + f[0, 2, 0] + f[1, 0, 0] - f[0, 0, 0]) / (1. + dy1)

    cdef double derivitive_d3fdxdydz_edge_yz(self, double[:] x, double[:, :, :] f) except? -1e999:
        cdef double dx1
        dx1 = (x[1] - x[0]) / (x[2] - x[1])
        return (f[2, 1, 1] - f[0, 1, 1] - f[2, 0, 1] + f[0, 0, 1] - f[2, 1, 0] + f[0, 1, 0] + f[2, 0, 0] - f[0, 0, 0]) / (1. + dx1)

    cdef double derivitive_d3fdxdydz_edge_xyz(self, double[:, :, :] f) except? -1e999:
        return f[1, 1, 1] - f[0, 1, 1] - f[1, 0, 1] + f[0, 0, 1] - f[1, 1, 0] + f[0, 1, 0] + f[1, 0, 0] - f[0, 0, 0]

    def __call__(self, index_x, index_y, index_z, derivative_order_x, derivative_order_y, derivative_order_z, rescale_norm_x, rescale_norm_y ,rescale_norm_z):
        return self.evaluate(index_x, index_y, index_z, derivative_order_x, derivative_order_y, derivative_order_z, rescale_norm_x, rescale_norm_y ,rescale_norm_z)


id_to_interpolator = {
    _Interpolator3DLinear.ID: _Interpolator3DLinear,
    _Interpolator3DCubic.ID: _Interpolator3DCubic
}

id_to_extrapolator = {
    _Extrapolator3DNone.ID: _Extrapolator3DNone,
    _Extrapolator3DNearest.ID: _Extrapolator3DNearest,
    _Extrapolator3DLinear.ID: _Extrapolator3DLinear,
    # _Extrapolator3DQuadratic.ID: _Extrapolator3DQuadratic
}

permitted_interpolation_combinations = {
    _Interpolator3DLinear.ID: [_Extrapolator3DNone.ID, _Extrapolator3DNearest.ID, _Extrapolator3DLinear.ID],
    _Interpolator3DCubic.ID: [_Extrapolator3DNone.ID, _Extrapolator3DNearest.ID, _Extrapolator3DLinear.ID]
}