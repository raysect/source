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

import numpy as np
from libc.math cimport fabs
cimport cython
from raysect.core.math.cython.utility cimport find_index
from raysect.core.math.cython.interpolation.linear cimport linear3d
from raysect.core.math.cython.interpolation.cubic cimport calc_coefficients_3d, evaluate_cubic_3d


cdef double[4] FACTORIAL
FACTORIAL[0] = 1.
FACTORIAL[1] = 1.
FACTORIAL[2] = 2.
FACTORIAL[3] = 6.


@cython.cdivision(True)
cdef double rescale_lower_normalisation(double dfdn, double x_lower, double x, double x_upper):
    """
    Derivatives that are normalised to the unit square (x_upper - x) = 1 are un-normalised, then re-normalised to
    (x - x_lower)
    """
    
    return dfdn * (x - x_lower) / (x_upper - x)


cdef int to_cell_index(int index, int last_index):
    """
    Transforming the output of find_index to find the index lower index of a cell required for an extrapolator.

    Finding the left most index of a grid cell from the output of find_index. The output of find_index is -1 at the 
    lower index, which has a lower border at index 0, and index = last_index at the upper border which is changed to 
    index = last_index - 1.

    :param int index: the index of the lower side of a unit cell.
    :param int last_index: the index of the final point.
    :return: the index of the lower cell at the border of the interpolator spline knots.
    """

    if index == -1:
        return 0
    elif index == last_index:
        return last_index - 1
    return index


cdef int to_knot_index(int index, int last_index):
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
        return 0
    elif index == last_index:
        return last_index
    return index



cdef class Interpolator3DArray(Function3D):
    """
    A configurable interpolator for 3D arrays.

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

    """

    def __init__(self, object x, object y, object z, object f, str interpolation_type, str extrapolation_type,
                 double extrapolation_range_x, double extrapolation_range_y, double extrapolation_range_z):

        x = np.array(x, dtype=np.float64, order='c')
        y = np.array(y, dtype=np.float64, order='c')
        z = np.array(z, dtype=np.float64, order='c')
        f = np.array(f, dtype=np.float64, order='c')

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

        if x.shape[0] < 2:
            raise ValueError(f'There must be at least 2 spline knots in all dimensions to interpolate. The shape of the x spline knot array is ({x.shape}).')

        if y.shape[0] < 2:
            raise ValueError(f'There must be at least 2 spline knots in all dimensions to interpolate. The shape of the y spline knot array is ({y.shape}).')

        if z.shape[0] < 2:
            raise ValueError(f'There must be at least 2 spline knots in all dimensions to interpolate. The shape of the z spline knot array is ({z.shape}).')

        if x.shape[0] != f.shape[0]:
            raise ValueError(f'Shape mismatch between x array ({x.shape}) and f array ({f.shape}).')

        if y.shape[0] != f.shape[1]:
            raise ValueError(f'Shape mismatch between y array ({y.shape}) and f array ({f.shape}).')

        if z.shape[0] != f.shape[2]:
            raise ValueError(f'Shape mismatch between z array ({z.shape}) and f array ({f.shape}).')

        # test monotonicity
        if (np.diff(x) <= 0).any():
            raise ValueError('The x array must be monotonically increasing.')

        if (np.diff(y) <= 0).any():
            raise ValueError('The y array must be monotonically increasing.')

        if (np.diff(z) <= 0).any():
            raise ValueError('The z array must be monotonically increasing.')

        self.x = x
        self.y = y
        self.z = z

        # obtain memory views for fast data access
        self._x_mv = x
        self._y_mv = y
        self._z_mv = z
        self._f_mv = f

        # prevent users being able to change the data arrays
        x.flags.writeable = False
        y.flags.writeable = False
        z.flags.writeable = False
        f.flags.writeable = False

        self._last_index_x = self.x.shape[0] - 1
        self._last_index_y = self.y.shape[0] - 1
        self._last_index_z = self.z.shape[0] - 1

        self._extrapolation_range_x = extrapolation_range_x
        self._extrapolation_range_y = extrapolation_range_y
        self._extrapolation_range_z = extrapolation_range_z

        # Check the requested interpolation type exists.
        interpolation_type = interpolation_type.lower()
        if interpolation_type not in id_to_interpolator:
            raise ValueError(
                f'Interpolation type {interpolation_type} not found. Options are {id_to_interpolator.keys()}.')

        # Check the requested extrapolation type exists.
        extrapolation_type = extrapolation_type.lower()
        if extrapolation_type not in id_to_extrapolator:
            raise ValueError(
                f'Extrapolation type {extrapolation_type} not found. Options are {id_to_extrapolator.keys()}.')

        # Permit combinations of interpolator and extrapolator where the order of extrapolator is higher than interpolator
        if extrapolation_type not in permitted_interpolation_combinations[interpolation_type]:
            raise ValueError(
                f'Extrapolation type {extrapolation_type} not compatible with interpolation type {interpolation_type}.')

        # Create the interpolator and extrapolator objects.
        self._interpolator = id_to_interpolator[interpolation_type](self._x_mv, self._y_mv, self._z_mv, self._f_mv)
        self._extrapolator = id_to_extrapolator[extrapolation_type](
            self._x_mv, self._y_mv, self._z_mv, self._f_mv, self._interpolator, extrapolation_range_x,
            extrapolation_range_y, extrapolation_range_z
        )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
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
        cdef int index_lower_x = to_cell_index(index_x, self._last_index_x)
        cdef int index_lower_y = to_cell_index(index_y, self._last_index_y)
        cdef int index_lower_z = to_cell_index(index_z, self._last_index_z)
        cdef bint outside_domain_x = index_x == -1 or (index_x == self._last_index_x and px != self._x_mv[self._last_index_x])
        cdef bint outside_domain_y = index_y == -1 or (index_y == self._last_index_y and py != self._y_mv[self._last_index_y])
        cdef bint outside_domain_z = index_z == -1 or (index_z == self._last_index_z and pz != self._z_mv[self._last_index_z])

        if outside_domain_x or outside_domain_y or outside_domain_z:
            return self._extrapolator.evaluate(px, py, pz, index_x, index_y, index_z)
        return self._interpolator.evaluate(px, py, pz, index_lower_x, index_lower_y, index_lower_z)

    @property
    def domain(self):
        """
        Returns min/max interval of 'x', 'y' and 'z' arrays.
        Order: min(x), max(x), min(y), max(y)., min(z), max(z).
        """
        return self._x_mv[0], self._x_mv[self._last_index_x], self._y_mv[0], self._y_mv[self._last_index_y], self._z_mv[0], self._z_mv[self._last_index_z]


cdef class _Interpolator3D:
    """
    Base class for 3D interpolators.

    :param x: 1D memory view of the spline point x positions.
    :param y: 1D memory view of the spline point y positions.
    :param z: 1D memory view of the spline point y positions.
    :param f: 3D memory view of the function value at spline point x, y, z positions.
    """

    ID = None

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
        Calculates interpolated value at a requested point.

        :param double px: the x position of the point for which an interpolated value is required.
        :param double py: the y position of the point for which an interpolated value is required.
        :param double pz: the z position of the point for which an interpolated value is required.
        :param int index_x: the lower index of the bin containing point px. (Result of bisection search).
        :param int index_y: the lower index of the bin containing point py. (Result of bisection search).
        :param int index_z: the lower index of the bin containing point pz. (Result of bisection search).
        """
        raise NotImplementedError('_Interpolator is an abstract base class.')

    cdef double analytic_gradient(self, double px, double py, double pz, int index_x, int index_y, int index_z, int order_x, int order_y, int order_z):
        """
        Calculates the interpolator's derivative of a valid order at a requested point.

        :param double px: the point for which an interpolated value is required.
        :param double py: the point for which an interpolated value is required.
        :param double pz: the point for which an interpolated value is required.
        :param int index_x: the lower index of the bin containing point px. (Result of bisection search).   
        :param int index_y: the lower index of the bin containing point py. (Result of bisection search).   
        :param int index_z: the lower index of the bin containing point pz. (Result of bisection search).   
        """

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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double px, double py, double pz, int index_x, int index_y, int index_z) except? -1e999:

        return linear3d(
            self._x[index_x], self._x[index_x + 1],
            self._y[index_y], self._y[index_y + 1],
            self._z[index_z], self._z[index_z + 1],
            self._f[index_x:index_x + 2, index_y:index_y + 2, index_z:index_z + 2],
            px, py, pz
        )

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
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
        cdef double dx, dy, dz

        if order_x == 1 and order_y == 1 and order_z == 1:
            return self._calculate_a7(index_x, index_y, index_z)

        elif order_x == 1 and order_y == 1 and order_z == 0:
            dz = (pz - self._z[index_z]) / (self._z[index_z + 1] - self._z[index_z])
            return self._calculate_a4(index_x, index_y, index_z) + self._calculate_a7(index_x, index_y, index_z) * dz

        elif order_x == 1 and order_y == 0 and order_z == 1:
            dy = (py - self._y[index_y]) / (self._y[index_y + 1] - self._y[index_y])
            return self._calculate_a5(index_x, index_y, index_z) + self._calculate_a7(index_x, index_y, index_z) * dy

        elif order_x == 0 and order_y == 1 and order_z == 1:
            dx = (px - self._x[index_x]) / (self._x[index_x + 1] - self._x[index_x])
            return self._calculate_a6(index_x, index_y, index_z) + self._calculate_a7(index_x, index_y, index_z) * dx

        elif order_x == 1 and order_y == 0 and order_z == 0:
            dy = (py - self._y[index_y]) / (self._y[index_y + 1] - self._y[index_y])
            dz = (pz - self._z[index_z]) / (self._z[index_z + 1] - self._z[index_z])
            return self._calculate_a1(index_x, index_y, index_z) \
                   + self._calculate_a4(index_x, index_y, index_z) * dy \
                   + self._calculate_a5(index_x, index_y, index_z) * dz \
                   + self._calculate_a7(index_x, index_y, index_z) * dy * dz

        elif order_x == 0 and order_y == 1 and order_z == 0:
            dx = (px - self._x[index_x]) / (self._x[index_x + 1] - self._x[index_x])
            dz = (pz - self._z[index_z]) / (self._z[index_z + 1] - self._z[index_z])
            return self._calculate_a2(index_x, index_y, index_z) \
                   + self._calculate_a4(index_x, index_y, index_z) * dx \
                   + self._calculate_a6(index_x, index_y, index_z) * dz \
                   + self._calculate_a7(index_x, index_y, index_z) * dx * dz

        elif order_x == 0 and order_y == 0 and order_z == 1:
            dx = (px - self._x[index_x]) / (self._x[index_x + 1] - self._x[index_x])
            dy = (py - self._y[index_y]) / (self._y[index_y + 1] - self._y[index_y])
            dz = (pz - self._z[index_z]) / (self._z[index_z + 1] - self._z[index_z])
            return self._calculate_a3(index_x, index_y, index_z) \
                   + self._calculate_a5(index_x, index_y, index_z) * dx \
                   + self._calculate_a6(index_x, index_y, index_z) * dy \
                   + self._calculate_a7(index_x, index_y, index_z) * dx * dy

        raise ValueError('The derivative order for x, y and z (order_x, order_y and order_z) must be a combination of 1 and 0 for the linear interpolator (but 0, 0, 0 should be handled by evaluating the interpolator).')

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _calculate_a0(self, int ix, int iy, int iz):
        """
        Calculate the trilinear coefficients in a unit cube. This function returns coefficient 0 (of the range 0-7).

        The trilinear function (which is the product of 3 linear functions) 
        f(x, y, z) = a0 + a1x + a2y + a3z + a4xy + a5xz +a6yz + a7xyz. Coefficients a0 - a7 are calculated for one unit 
        cube. The coefficients are calculated from inverting the equation Xa = fv. Where:
        
            X = [[1, x1, y1, z1, x1y1, x1z1, y1z1, x1y1z1],         a = [a0,         fv = [f(0, 0, 0),
                 [1, x2, y1, z1, x2y1, x2z1, y1z1, x2y1z1],              a1,               f(1, 0, 0),
                 [1, x1, y2, z1, x1y2, x1z1, y2z1, x1y2z1],              a2,               f(0, 1, 0),
                 [1, x2, y2, z1, x2y2, x2z1, y2z1, x2y2z1],              a3,               f(1, 1, 0),
                 [1, x1, y1, z2, x1y1, x1z2, y1z2, x1y1z2],              a4,               f(0, 0, 1),
                 [1, x2, y1, z2, x2y1, x2z2, y1z2, x2y1z2],              a5,               f(1, 0, 1),
                 [1, x1, y2, z2, x1y2, x1z2, y2z2, x1y2z2],              a6,               f(0, 1, 1),
                 [1, x2, y2, z2, x2y2, x2z2, y2z2, x2y2z2]]              a7]               f(1, 1, 1)]
        
        This simplifies where x1, y1, z1 = 0, x2, y2, z2 = 1 for the unit cube to find a = X^{-1} fv where:
        
            a[0] =   f[0][0][0]
            a[1] = - f[0][0][0] + f[1][0][0]
            a[2] = - f[0][0][0] + f[0][1][0]
            a[3] = - f[0][0][0] + f[0][0][1]
            a[4] =   f[0][0][0] - f[0][1][0] - f[1][0][0] + f[1][1][0]
            a[5] =   f[0][0][0] - f[0][0][1] - f[1][0][0] + f[1][0][1]
            a[6] =   f[0][0][0] - f[0][0][1] - f[0][1][0] + f[0][1][1]
            a[7] = - f[0][0][0] + f[0][0][1] + f[0][1][0] - f[0][1][1]
                   + f[1][0][0] - f[1][0][1] - f[1][1][0] + f[1][1][1]

        :param int ix: the lower index of the bin containing point px (result of bisection search).   
        :param int iy: the lower index of the bin containing point py (result of bisection search). 
        :param int iz: the lower index of the bin containing point pz (result of bisection search).
        """

        return self._f[ix, iy, iz]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _calculate_a1(self, int ix, int iy, int iz):
        """
        Calculate the trilinear coefficients in a unit cube. This function returns coefficient 1 (of the range 0-7).
        
        See _calculate_a0 for a full description.
        """

        return -self._f[ix, iy, iz] + self._f[ix + 1, iy, iz]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _calculate_a2(self, int ix, int iy, int iz):
        """
        Calculate the trilinear coefficients in a unit cube. This function returns coefficient 2 (of the range 0-7).

        See _calculate_a0 for a full description.
        """

        return -self._f[ix, iy, iz] + self._f[ix, iy + 1, iz]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _calculate_a3(self, int ix, int iy, int iz):
        """
        Calculate the trilinear coefficients in a unit cube. This function returns coefficient 3 (of the range 0-7).

        See _calculate_a0 for a full description.
        """

        return -self._f[ix, iy, iz] + self._f[ix, iy, iz + 1]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _calculate_a4(self, int ix, int iy, int iz):
        """
        Calculate the trilinear coefficients in a unit cube. This function returns coefficient 4 (of the range 0-7).

        See _calculate_a0 for a full description.
        """

        return self._f[ix, iy, iz] - self._f[ix, iy + 1, iz] \
               - self._f[ix + 1, iy, iz] + self._f[ix + 1, iy + 1, iz]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _calculate_a5(self, int ix, int iy, int iz):
        """
        Calculate the trilinear coefficients in a unit cube. This function returns coefficient 5 (of the range 0-7).

        See _calculate_a0 for a full description.
        """
        return self._f[ix, iy, iz] - self._f[ix, iy, iz + 1] \
               - self._f[ix + 1, iy, iz] + self._f[ix + 1, iy, iz + 1]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _calculate_a6(self, int ix, int iy, int iz):
        """
        Calculate the trilinear coefficients in a unit cube. This function returns coefficient 6 (of the range 0-7).

        See _calculate_a0 for a full description.
        """
        return self._f[ix, iy, iz] - self._f[ix, iy, iz + 1] \
               - self._f[ix, iy + 1, iz] + self._f[ix, iy + 1, iz + 1]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _calculate_a7(self, int ix, int iy, int iz):
        """
        Calculate the trilinear coefficients in a unit cube. This function returns coefficient 7 (of the range 0-7).

        See _calculate_a0 for a full description.
        """
        return - self._f[ix, iy, iz] + self._f[ix, iy, iz + 1] \
               + self._f[ix, iy + 1, iz] - self._f[ix, iy + 1, iz + 1] \
               + self._f[ix + 1, iy, iz] - self._f[ix + 1, iy, iz + 1] \
               - self._f[ix + 1, iy + 1, iz] + self._f[ix + 1, iy + 1, iz + 1]


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

        # where the spline coefficients (a) have been calculated the value is set to 1, 0 otherwise
        self._calculated = np.zeros((self._last_index_x, self._last_index_y, self._last_index_z), dtype=np.uint8)

        # Store the cubic spline coefficients, where increasing index values are the coefficients for the coefficients of higher powers of x, y, z in the last 3 dimensions.
        self._a = np.zeros((self._last_index_x, self._last_index_y, self._last_index_z, 4, 4, 4), dtype=np.float64)
        self._array_derivative = _ArrayDerivative3D(self._x, self._y, self._z, self._f)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double px, double py, double pz, int index_x, int index_y, int index_z) except? -1e999:

        # rescale x between 0 and 1
        cdef double nx, ny, nz
        cdef double[4][4][4] a

        # normalise x and y to a unit cell
        nx = (px - self._x[index_x]) / (self._x[index_x + 1] - self._x[index_x])
        ny = (py - self._y[index_y]) / (self._y[index_y + 1] - self._y[index_y])
        nz = (pz - self._z[index_z]) / (self._z[index_z + 1] - self._z[index_z])

        # Calculate the coefficients (and gradients at each spline point) if they dont exist.
        self._cache_coefficients(index_x, index_y, index_z, a)

        return evaluate_cubic_3d(a, nx, ny, nz)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef _cache_coefficients(self, int index_x, int index_y, int index_z, double[4][4][4] a):
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
        if not self._calculated[index_x, index_y, index_z]:

            f[0][0][0] = self._f[index_x, index_y, index_z]
            f[0][1][0] = self._f[index_x, index_y + 1, index_z]
            f[0][1][1] = self._f[index_x, index_y + 1, index_z + 1]
            f[0][0][1] = self._f[index_x, index_y, index_z + 1]
            f[1][0][0] = self._f[index_x + 1, index_y, index_z]
            f[1][1][0] = self._f[index_x + 1, index_y + 1, index_z]
            f[1][0][1] = self._f[index_x + 1, index_y, index_z + 1]
            f[1][1][1] = self._f[index_x + 1, index_y + 1, index_z + 1]

            dfdx[0][0][0] = self._array_derivative.evaluate_df_dx(index_x, index_y, index_z, False)
            dfdx[0][1][0] = self._array_derivative.evaluate_df_dx(index_x, index_y + 1, index_z, False)
            dfdx[0][1][1] = self._array_derivative.evaluate_df_dx(index_x, index_y + 1, index_z + 1, False)
            dfdx[0][0][1] = self._array_derivative.evaluate_df_dx(index_x, index_y, index_z + 1, False)
            dfdx[1][0][0] = self._array_derivative.evaluate_df_dx(index_x + 1, index_y, index_z, True)
            dfdx[1][1][0] = self._array_derivative.evaluate_df_dx(index_x + 1, index_y + 1, index_z, True)
            dfdx[1][0][1] = self._array_derivative.evaluate_df_dx(index_x + 1, index_y, index_z + 1, True)
            dfdx[1][1][1] = self._array_derivative.evaluate_df_dx(index_x + 1, index_y + 1, index_z + 1, True)

            dfdy[0][0][0] = self._array_derivative.evaluate_df_dy(index_x, index_y, index_z, False)
            dfdy[0][1][0] = self._array_derivative.evaluate_df_dy(index_x, index_y + 1, index_z, True)
            dfdy[0][1][1] = self._array_derivative.evaluate_df_dy(index_x, index_y + 1, index_z + 1, True)
            dfdy[0][0][1] = self._array_derivative.evaluate_df_dy(index_x, index_y, index_z + 1, False)
            dfdy[1][0][0] = self._array_derivative.evaluate_df_dy(index_x + 1, index_y, index_z, False)
            dfdy[1][1][0] = self._array_derivative.evaluate_df_dy(index_x + 1, index_y + 1, index_z, True)
            dfdy[1][0][1] = self._array_derivative.evaluate_df_dy(index_x + 1, index_y, index_z + 1, False)
            dfdy[1][1][1] = self._array_derivative.evaluate_df_dy(index_x + 1, index_y + 1, index_z + 1, True)

            dfdz[0][0][0] = self._array_derivative.evaluate_df_dz(index_x, index_y, index_z, False)
            dfdz[0][1][0] = self._array_derivative.evaluate_df_dz(index_x, index_y + 1, index_z, False)
            dfdz[0][1][1] = self._array_derivative.evaluate_df_dz(index_x, index_y + 1, index_z + 1, True)
            dfdz[0][0][1] = self._array_derivative.evaluate_df_dz(index_x, index_y, index_z + 1, True)
            dfdz[1][0][0] = self._array_derivative.evaluate_df_dz(index_x + 1, index_y, index_z, False)
            dfdz[1][1][0] = self._array_derivative.evaluate_df_dz(index_x + 1, index_y + 1, index_z, False)
            dfdz[1][0][1] = self._array_derivative.evaluate_df_dz(index_x + 1, index_y, index_z + 1, True)
            dfdz[1][1][1] = self._array_derivative.evaluate_df_dz(index_x + 1, index_y + 1, index_z + 1, True)

            d2fdxdy[0][0][0] = self._array_derivative.evaluate_d2fdxdy(index_x, index_y, index_z, False, False)
            d2fdxdy[0][1][0] = self._array_derivative.evaluate_d2fdxdy(index_x, index_y + 1, index_z, False, True)
            d2fdxdy[0][1][1] = self._array_derivative.evaluate_d2fdxdy(index_x, index_y + 1, index_z + 1, False, True)
            d2fdxdy[0][0][1] = self._array_derivative.evaluate_d2fdxdy(index_x, index_y, index_z + 1, False, False)
            d2fdxdy[1][0][0] = self._array_derivative.evaluate_d2fdxdy(index_x + 1, index_y, index_z, True, False)
            d2fdxdy[1][1][0] = self._array_derivative.evaluate_d2fdxdy(index_x + 1, index_y + 1, index_z, True, True)
            d2fdxdy[1][0][1] = self._array_derivative.evaluate_d2fdxdy(index_x + 1, index_y, index_z + 1, True, False)
            d2fdxdy[1][1][1] = self._array_derivative.evaluate_d2fdxdy(index_x + 1, index_y + 1, index_z + 1, True, True)

            d2fdxdz[0][0][0] = self._array_derivative.evaluate_d2fdxdz(index_x, index_y, index_z, False, False)
            d2fdxdz[0][1][0] = self._array_derivative.evaluate_d2fdxdz(index_x, index_y + 1, index_z, False, False)
            d2fdxdz[0][1][1] = self._array_derivative.evaluate_d2fdxdz(index_x, index_y + 1, index_z + 1, False, True)
            d2fdxdz[0][0][1] = self._array_derivative.evaluate_d2fdxdz(index_x, index_y, index_z + 1, False, True)
            d2fdxdz[1][0][0] = self._array_derivative.evaluate_d2fdxdz(index_x + 1, index_y, index_z, True, False)
            d2fdxdz[1][1][0] = self._array_derivative.evaluate_d2fdxdz(index_x + 1, index_y + 1, index_z, True, False)
            d2fdxdz[1][0][1] = self._array_derivative.evaluate_d2fdxdz(index_x + 1, index_y, index_z + 1, True, True)
            d2fdxdz[1][1][1] = self._array_derivative.evaluate_d2fdxdz(index_x + 1, index_y + 1, index_z + 1, True, True)

            d2fdydz[0][0][0] = self._array_derivative.evaluate_d2fdydz(index_x, index_y, index_z, False, False)
            d2fdydz[0][1][0] = self._array_derivative.evaluate_d2fdydz(index_x, index_y + 1, index_z, True, False)
            d2fdydz[0][1][1] = self._array_derivative.evaluate_d2fdydz(index_x, index_y + 1, index_z + 1, True, True)
            d2fdydz[0][0][1] = self._array_derivative.evaluate_d2fdydz(index_x, index_y, index_z + 1, False, True)
            d2fdydz[1][0][0] = self._array_derivative.evaluate_d2fdydz(index_x + 1, index_y, index_z, False, False)
            d2fdydz[1][1][0] = self._array_derivative.evaluate_d2fdydz(index_x + 1, index_y + 1, index_z, True, False)
            d2fdydz[1][0][1] = self._array_derivative.evaluate_d2fdydz(index_x + 1, index_y, index_z + 1, False, True)
            d2fdydz[1][1][1] = self._array_derivative.evaluate_d2fdydz(index_x + 1, index_y + 1, index_z + 1, True, True)

            d3fdxdydz[0][0][0] = self._array_derivative.evaluate_d3fdxdydz(index_x, index_y, index_z, False, False, False)
            d3fdxdydz[0][1][0] = self._array_derivative.evaluate_d3fdxdydz(index_x, index_y + 1, index_z, False, True, False)
            d3fdxdydz[0][1][1] = self._array_derivative.evaluate_d3fdxdydz(index_x, index_y + 1, index_z + 1, False, True, True)
            d3fdxdydz[0][0][1] = self._array_derivative.evaluate_d3fdxdydz(index_x, index_y, index_z + 1, False, False, True)
            d3fdxdydz[1][0][0] = self._array_derivative.evaluate_d3fdxdydz(index_x + 1, index_y, index_z, True, False, False)
            d3fdxdydz[1][1][0] = self._array_derivative.evaluate_d3fdxdydz(index_x + 1, index_y + 1, index_z, True, True, False)
            d3fdxdydz[1][0][1] = self._array_derivative.evaluate_d3fdxdydz(index_x + 1, index_y, index_z + 1, True, False, True)
            d3fdxdydz[1][1][1] = self._array_derivative.evaluate_d3fdxdydz(index_x + 1, index_y + 1, index_z + 1, True, True, True)

            calc_coefficients_3d(f, dfdx, dfdy, dfdz, d2fdxdy, d2fdxdz, d2fdydz, d3fdxdydz, a)
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        self._a[index_x, index_y, index_z, i, j, k] = a[i][j][k]
            self._calculated[index_x, index_y, index_z] = 1

        else:
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        a[i][j][k] = self._a[index_x, index_y, index_z, i, j, k]

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double analytic_gradient(self, double px, double py, double pz, int index_x, int index_y, int index_z, int order_x, int order_y, int order_z):
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
        cdef double nx, ny, nz
        cdef double x_bound, y_bound, z_bound
        cdef double[4][4][4] a
        cdef double[4] x_powers, y_powers, z_powers
        cdef double df_dn
        cdef int i, j, k

        if order_x > 3:
            raise ValueError('Can\'t get a gradient of order 4 or more in cubic.')

        # normalise x and y to a unit cell
        nx = (px - self._x[index_x]) / (self._x[index_x + 1] - self._x[index_x])
        ny = (py - self._y[index_y]) / (self._y[index_y + 1] - self._y[index_y])
        nz = (pz - self._z[index_z]) / (self._z[index_z + 1] - self._z[index_z])

        # Calculate the coefficients (and gradients at each spline point) if they dont exist
        self._cache_coefficients(index_x, index_y, index_z, a)
        x_powers[0] = 1
        x_powers[1] = nx
        x_powers[2] = nx * nx
        x_powers[3] = nx * x_powers[2]
        
        y_powers[0] = 1
        y_powers[1] = ny
        y_powers[2] = ny * ny
        y_powers[3] = ny * y_powers[2]
        
        z_powers[0] = 1
        z_powers[1] = nz
        z_powers[2] = nz * nz
        z_powers[3] = nz * z_powers[2]

        df_dn = 0.0
        for i in range(order_x, 4):
            for j in range(order_y, 4):
                for k in range(order_z, 4):
                    df_dn += (a[i][j][k]
                              * (FACTORIAL[i] / FACTORIAL[i - order_x]) * x_powers[i - order_x]
                              * (FACTORIAL[j] / FACTORIAL[j - order_y]) * y_powers[j - order_y]
                              * (FACTORIAL[k] / FACTORIAL[k - order_z]) * z_powers[k - order_z])
        return df_dn


cdef class _Extrapolator3D:
    """
    Base class for Function3D extrapolators.

    :param x: 1D memory view of the spline point x positions.
    :param y: 1D memory view of the spline point y positions.
    :param z: 1D memory view of the spline point z positions.
    :param f: 3D memory view of the function value at spline point x, y, z positions.
    :param interpolator: stored _Interpolator2D object that is being used.
    """

    ID = None

    def __init__(self, double[::1] x, double[::1] y, double[::1] z, double[:, :, ::1] f, _Interpolator3D interpolator, double extrapolation_range_x, double extrapolation_range_y, double extrapolation_range_z):

        self._x = x
        self._y = y
        self._z = z
        self._f = f
        self._last_index_x = self._x.shape[0] - 1
        self._last_index_y = self._y.shape[0] - 1
        self._last_index_z = self._z.shape[0] - 1
        self._interpolator = interpolator
        self._extrapolation_range_x = extrapolation_range_x
        self._extrapolation_range_y = extrapolation_range_y
        self._extrapolation_range_z = extrapolation_range_z

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double px, double py, double pz, int index_x, int index_y, int index_z) except? -1e999:

        cdef int index_lower_x = to_cell_index(index_x, self._last_index_x)
        cdef int index_lower_y = to_cell_index(index_y, self._last_index_y)
        cdef int index_lower_z = to_cell_index(index_z, self._last_index_z)
        cdef int edge_x_index = to_knot_index(index_x, self._last_index_x)
        cdef int edge_y_index = to_knot_index(index_y, self._last_index_y)
        cdef int edge_z_index = to_knot_index(index_z, self._last_index_z)

        # Corner in x, y, z
        if (index_x == -1 or index_x == self._last_index_x) and (index_y == -1 or index_y == self._last_index_y) and (index_z == -1 or index_z == self._last_index_z):

            if fabs(px - self._x[edge_x_index]) > self._extrapolation_range_x:
                raise ValueError(f'The specified value (x={px}) is outside of extrapolation range.')

            if fabs(py - self._y[edge_y_index]) > self._extrapolation_range_y:
                raise ValueError(f'The specified value (y={py}) is outside of extrapolation range.')

            if fabs(pz - self._z[edge_z_index]) > self._extrapolation_range_z:
                raise ValueError(f'The specified value (z={pz}) is outside of extrapolation range.')

            return self._evaluate_edge_xyz(px, py, pz, index_lower_x, index_lower_y, index_lower_z, edge_x_index, edge_y_index, edge_z_index)

        elif (index_x == -1 or index_x == self._last_index_x) and (index_y == -1 or index_y == self._last_index_y):

            if fabs(px - self._x[edge_x_index]) > self._extrapolation_range_x:
                raise ValueError(f'The specified value (x={px}) is outside of extrapolation range.')

            if fabs(py - self._y[edge_y_index]) > self._extrapolation_range_y:
                raise ValueError(f'The specified value (y={py}) is outside of extrapolation range.')

            return self._evaluate_edge_xy(px, py, pz, index_lower_x, index_lower_y, index_lower_z, edge_x_index, edge_y_index, edge_z_index)

        elif (index_x == -1 or index_x == self._last_index_x) and (index_z == -1 or index_z == self._last_index_z):

            if fabs(px - self._x[edge_x_index]) > self._extrapolation_range_x:
                raise ValueError(f'The specified value (x={px}) is outside of extrapolation range.')

            if fabs(pz - self._z[edge_z_index]) > self._extrapolation_range_z:
                raise ValueError(f'The specified value (z={pz}) is outside of extrapolation range.')

            return self._evaluate_edge_xz(px, py, pz, index_lower_x, index_lower_y, index_lower_z, edge_x_index, edge_y_index, edge_z_index)

        elif (index_y == -1 or index_y == self._last_index_y) and (index_z == -1 or index_z == self._last_index_z):

            if fabs(py - self._y[edge_y_index]) > self._extrapolation_range_y:
                raise ValueError(f'The specified value (y={py}) is outside of extrapolation range.')

            if fabs(pz - self._z[edge_z_index]) > self._extrapolation_range_z:
                raise ValueError(f'The specified value (z={pz}) is outside of extrapolation range.')

            return self._evaluate_edge_yz(px, py, pz, index_lower_x, index_lower_y, index_lower_z, edge_x_index, edge_y_index, edge_z_index)

        elif index_x == -1 or index_x == self._last_index_x:

            if fabs(px - self._x[edge_x_index]) > self._extrapolation_range_x:
                raise ValueError(f'The specified value (x={px}) is outside of extrapolation range.')

            return self._evaluate_edge_x(px, py, pz, index_lower_x, index_lower_y, index_lower_z, edge_x_index, edge_y_index, edge_z_index)

        elif index_y == -1 or index_y == self._last_index_y:

            if fabs(py - self._y[edge_y_index]) > self._extrapolation_range_y:
                raise ValueError(f'The specified value (y={py}) is outside of extrapolation range.')

            return self._evaluate_edge_y(px, py, pz, index_lower_x, index_lower_y, index_lower_z, edge_x_index, edge_y_index, edge_z_index)

        elif index_z == -1 or index_z == self._last_index_z:

            if fabs(pz - self._z[edge_z_index]) > self._extrapolation_range_z:
                raise ValueError(f'The specified value (z={pz}) is outside of extrapolation range.')

            return self._evaluate_edge_z(px, py, pz, index_lower_x, index_lower_y, index_lower_z, edge_x_index, edge_y_index, edge_z_index)

        else:
            raise ValueError('Interpolated index parsed to extrapolator.')

    cdef double _evaluate_edge_x(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise NotImplementedError(f'{self.__class__} not implemented.')

    cdef double _evaluate_edge_y(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise NotImplementedError(f'{self.__class__} not implemented.')

    cdef double _evaluate_edge_z(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise NotImplementedError(f'{self.__class__} not implemented.')

    cdef double _evaluate_edge_xy(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise NotImplementedError(f'{self.__class__} not implemented.')

    cdef double _evaluate_edge_xz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise NotImplementedError(f'{self.__class__} not implemented.')

    cdef double _evaluate_edge_yz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise NotImplementedError(f'{self.__class__} not implemented.')

    cdef double _evaluate_edge_xyz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise NotImplementedError(f'{self.__class__} not implemented.')


cdef class _Extrapolator3DNone(_Extrapolator3D):
    """
    Extrapolator that does nothing.

    :param x: 1D memory view of the spline point x positions.
    :param y: 1D memory view of the spline point y positions.
    :param z: 1D memory view of the spline point y positions.
    :param f: 3D memory view of the function value at spline point x, y, z positions.
    :param interpolator: stored _Interpolator2D object that is being used.
    """

    ID = 'none'

    cdef double _evaluate_edge_x(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise ValueError(f'Extrapolation not available. Interpolate within function range x {min(self._x)}-{max(self._x)}.')

    cdef double _evaluate_edge_y(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise ValueError(f'Extrapolation not available. Interpolate within function range y {min(self._y)}-{max(self._y)}.')

    cdef double _evaluate_edge_z(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise ValueError(f'Extrapolation not available. Interpolate within function range z {min(self._z)}-{max(self._z)}.')

    cdef double _evaluate_edge_xy(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise ValueError(f'Extrapolation not available. Interpolate within function range x {min(self._x)}-{max(self._x)} and y  {min(self._y)}-{max(self._y)}.')

    cdef double _evaluate_edge_xz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise ValueError(f'Extrapolation not available. Interpolate within function range x {min(self._x)}-{max(self._x)} and z  {min(self._z)}-{max(self._z)}.')

    cdef double _evaluate_edge_yz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise ValueError(f'Extrapolation not available. Interpolate within function range y {min(self._y)}-{max(self._y)} and z {min(self._z)}-{max(self._z)}.')

    cdef double _evaluate_edge_xyz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        raise ValueError(f'Extrapolation not available. Interpolate within function range x {min(self._x)}-{max(self._x)}, y  {min(self._y)}-{max(self._y)} and z {min(self._z)}-{max(self._z)}.')


cdef class _Extrapolator3DNearest(_Extrapolator3D):
    """
    Extrapolator that returns nearest input value.

    :param x: 1D memory view of the spline point x positions.
    :param y: 1D memory view of the spline point y positions.
    :param z: 1D memory view of the spline point y positions.
    :param f: 3D memory view of the function value at spline point x, y, z positions.
    :param interpolator: stored _Interpolator2D object that is being used.
    """

    ID = 'nearest'

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate_edge_x(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        return self._interpolator.evaluate(self._x[edge_x_index], py, pz, index_x, index_y, index_z)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate_edge_y(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        return self._interpolator.evaluate(px, self._y[edge_y_index], pz, index_x, index_y, index_z)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate_edge_z(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        return self._interpolator.evaluate(px, py, self._z[edge_z_index], index_x, index_y, index_z)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate_edge_xy(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        return self._interpolator.evaluate(self._x[edge_x_index], self._y[edge_y_index], pz, index_x, index_y, index_z)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate_edge_xz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        return self._interpolator.evaluate(self._x[edge_x_index], py, self._z[edge_z_index], index_x, index_y, index_z)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate_edge_yz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        return self._interpolator.evaluate(px, self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate_edge_xyz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
        return self._interpolator.evaluate(self._x[edge_x_index], self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z)


cdef class _Extrapolator3DLinear(_Extrapolator3D):
    """
    Extrapolator that returns linearly extrapolated input value.

    :param x: 1D memory view of the spline point x positions.
    :param y: 1D memory view of the spline point y positions.
    :param z: 1D memory view of the spline point z positions.
    :param f: 3D memory view of the function value at spline point x, y, z positions.
    :param interpolator: stored _Interpolator3D object that is being used.
    """

    ID = 'linear'

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate_edge_x(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:

        cdef double rdx, f, df_dx

        rdx = 1.0 / (self._x[index_x + 1] - self._x[index_x])
        f = self._interpolator.evaluate(self._x[edge_x_index], py, pz, index_x, index_y, index_z)
        df_dx = self._interpolator.analytic_gradient(self._x[edge_x_index], py, pz, index_x, index_y, index_z, 1, 0, 0) * rdx
        return f + df_dx * (px - self._x[edge_x_index])

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate_edge_y(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:

        cdef double rdy, f, df_dy

        rdy = 1.0 / (self._y[index_y + 1] - self._y[index_y])
        f = self._interpolator.evaluate(px, self._y[edge_y_index], pz, index_x, index_y, index_z)
        df_dy = self._interpolator.analytic_gradient(px, self._y[edge_y_index], pz, index_x, index_y, index_z, 0, 1, 0) * rdy
        return f + df_dy * (py - self._y[edge_y_index])

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate_edge_z(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:

        cdef double rdz, f, df_dz

        rdz = 1.0 / (self._z[index_z + 1] - self._z[index_z])
        f = self._interpolator.evaluate(px, py, self._z[edge_z_index], index_x, index_y, index_z)
        df_dz = self._interpolator.analytic_gradient(px, py, self._z[edge_z_index], index_x, index_y, index_z, 0, 0, 1) * rdz
        return f + df_dz * (pz - self._z[edge_z_index])

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate_edge_xy(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:

        cdef double rdx, rdy, f, df_dx, df_dy, d2f_dxdy

        rdx = 1.0 / (self._x[index_x + 1] - self._x[index_x])
        rdy = 1.0 / (self._y[index_y + 1] - self._y[index_y])

        f = self._interpolator.evaluate(self._x[edge_x_index], self._y[edge_y_index], pz, index_x, index_y, index_z)
        df_dx = self._interpolator.analytic_gradient(self._x[edge_x_index], self._y[edge_y_index], pz, index_x, index_y, index_z, 1, 0, 0) * rdx
        df_dy = self._interpolator.analytic_gradient(self._x[edge_x_index], self._y[edge_y_index], pz, index_x, index_y, index_z, 0, 1, 0) * rdy
        d2f_dxdy = self._interpolator.analytic_gradient(self._x[edge_x_index], self._y[edge_y_index], pz, index_x, index_y, index_z, 1, 1, 0) * rdx * rdy

        return f + df_dx * (px - self._x[edge_x_index]) \
                 + df_dy * (py - self._y[edge_y_index]) \
                 + d2f_dxdy * (py - self._y[edge_y_index]) * (px - self._x[edge_x_index])

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate_edge_xz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:

        cdef double rdx, rdz, f, df_dx, df_dz, d2f_dxdz

        rdx = 1.0 / (self._x[index_x + 1] - self._x[index_x])
        rdz = 1.0 / (self._z[index_z + 1] - self._z[index_z])

        f = self._interpolator.evaluate(self._x[edge_x_index], py, self._z[edge_z_index], index_x, index_y, index_z)
        df_dx = self._interpolator.analytic_gradient(self._x[edge_x_index], py, self._z[edge_z_index], index_x, index_y, index_z, 1, 0, 0) * rdx
        df_dz = self._interpolator.analytic_gradient(self._x[edge_x_index], py, self._z[edge_z_index], index_x, index_y, index_z, 0, 0, 1) * rdz
        d2f_dxdz = self._interpolator.analytic_gradient(self._x[edge_x_index], py, self._z[edge_z_index], index_x, index_y, index_z, 1, 0, 1) * rdx * rdz

        return f + df_dx * (px - self._x[edge_x_index]) \
                 + df_dz * (pz - self._z[edge_z_index]) \
                 + d2f_dxdz * (pz - self._z[edge_z_index]) * (px - self._x[edge_x_index])

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate_edge_yz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:

        cdef double rdy, rdz, f, df_dy, df_dz, d2f_dydz

        rdy = 1.0 / (self._y[index_y + 1] - self._y[index_y])
        rdz = 1.0 / (self._z[index_z + 1] - self._z[index_z])

        f = self._interpolator.evaluate(px, self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z)
        df_dy = self._interpolator.analytic_gradient(px, self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z, 0, 1, 0) * rdy
        df_dz = self._interpolator.analytic_gradient(px, self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z, 0, 0, 1) * rdz
        d2f_dydz = self._interpolator.analytic_gradient(px, self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z, 0, 1, 1) * rdy * rdz

        return f + df_dy * (py - self._y[edge_y_index]) \
                 + df_dz * (pz - self._z[edge_z_index]) \
                 + d2f_dydz * (pz - self._z[edge_z_index]) * (py - self._y[edge_y_index])

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _evaluate_edge_xyz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999:
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

        cdef double rdx, rdy, rdz, f, df_dx, df_dy, df_dz, d2f_dxdy, d2f_dxdz, d2f_dydz, d3f_dxdydz

        rdx = 1.0 / (self._x[index_x + 1] - self._x[index_x])
        rdy = 1.0 / (self._y[index_y + 1] - self._y[index_y])
        rdz = 1.0 / (self._z[index_z + 1] - self._z[index_z])

        f = self._interpolator.evaluate(self._x[edge_x_index], self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z)

        df_dx = self._interpolator.analytic_gradient(self._x[edge_x_index], self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z, 1, 0, 0) * rdx
        df_dy = self._interpolator.analytic_gradient(self._x[edge_x_index], self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z, 0, 1, 0) * rdy
        df_dz = self._interpolator.analytic_gradient(self._x[edge_x_index], self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z, 0, 0, 1) * rdz

        d2f_dxdy = self._interpolator.analytic_gradient(self._x[edge_x_index], self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z, 1, 1, 0) * rdx * rdy
        d2f_dxdz = self._interpolator.analytic_gradient(self._x[edge_x_index], self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z, 1, 0, 1) * rdx * rdz
        d2f_dydz = self._interpolator.analytic_gradient(self._x[edge_x_index], self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z, 0, 1, 1) * rdy * rdz

        d3f_dxdydz = self._interpolator.analytic_gradient(self._x[edge_x_index], self._y[edge_y_index], self._z[edge_z_index], index_x, index_y, index_z, 1, 1, 1) * rdx * rdy * rdz

        return f + df_dx * (px - self._x[edge_x_index]) \
                 + df_dy * (py - self._y[edge_y_index]) \
                 + df_dz * (pz - self._z[edge_z_index]) \
                 + d2f_dxdy * (px - self._x[edge_x_index]) * (py - self._y[edge_y_index]) \
                 + d2f_dxdz * (pz - self._z[edge_z_index]) * (px - self._x[edge_x_index]) \
                 + d2f_dydz * (pz - self._z[edge_z_index]) * (py - self._y[edge_y_index]) \
                 + d3f_dxdydz * (px - self._x[edge_x_index]) * (py - self._y[edge_y_index]) * (pz - self._z[edge_z_index])


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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate_df_dx(self, int index_x, int index_y, int index_z, bint rescale_norm_x) except? -1e999:
        """
        Evaluate df/dx at index_x, index_y, index_z. If near the edge of the grid, the array size that the 
        derivative is calculated on is reduced.

        :param index_x: The lower index of the x grid cell to evaluate.
        :param index_y: The lower index of the y grid cell to evaluate.
        :param index_z: The lower index of the z grid cell to evaluate.

        :param rescale_norm_x: A boolean as whether to rescale to the delta before x[index_x] or after (default).

        """

        cdef double dfdn

        if index_x == self._last_index_x:
            return self._derivative_dfdx_edge(index_x - 1, index_y, index_z)

        elif index_x == 0:
            return self._derivative_dfdx_edge(index_x, index_y, index_z)

        dfdn = self._derivative_dfdx(index_x - 1, index_y, index_z)
        if rescale_norm_x:
            dfdn = rescale_lower_normalisation(dfdn,  self._x[index_x - 1], self._x[index_x], self._x[index_x + 1])
        return dfdn

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate_df_dy(self, int index_x, int index_y, int index_z, bint rescale_norm_y) except? -1e999:
        """
        Evaluate df/dy at index_x, index_y, index_z. If near the edge of the grid, the array size that the 
        derivative is calculated on is reduced.

        :param index_x: The lower index of the x grid cell to evaluate.
        :param index_y: The lower index of the y grid cell to evaluate.
        :param index_z: The lower index of the z grid cell to evaluate.

        :param rescale_norm_y: A boolean as whether to rescale to the delta before y[index_y] or after (default).

        """

        cdef double dfdn

        if index_y == self._last_index_y:
            return self._derivative_dfdy_edge(index_x, index_y - 1, index_z)
        elif index_y == 0:
            return self._derivative_dfdy_edge(index_x, index_y, index_z)

        dfdn = self._derivative_dfdy(index_x, index_y - 1, index_z)
        if rescale_norm_y:
            dfdn = rescale_lower_normalisation(dfdn,  self._y[index_y - 1], self._y[index_y], self._y[index_y + 1])
        return dfdn

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate_df_dz(self, int index_x, int index_y, int index_z, bint rescale_norm_z) except? -1e999:
        """
        Evaluate df/dz at index_x, index_y, index_z. If near the edge of the grid, the array size that the 
        derivative is calculated on is reduced.

        :param index_x: The lower index of the x grid cell to evaluate.
        :param index_y: The lower index of the y grid cell to evaluate.
        :param index_z: The lower index of the z grid cell to evaluate.

        :param rescale_norm_z: A boolean as whether to rescale to the delta before z[index_z] or after (default).

        """

        cdef double dfdn

        if index_z == self._last_index_z:
            return self._derivative_dfdz_edge(index_x, index_y, index_z - 1)

        elif index_z == 0:
            return self._derivative_dfdz_edge(index_x, index_y, index_z)

        dfdn = self._derivative_dfdz(index_x, index_y, index_z - 1)
        if rescale_norm_z:
            dfdn = rescale_lower_normalisation(dfdn,  self._z[index_z - 1], self._z[index_z], self._z[index_z + 1])
        return dfdn

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate_d2fdxdy(self, int index_x, int index_y, int index_z, bint rescale_norm_x, bint rescale_norm_y) except? -1e999:
        """
        Evaluate d2f/dxdy at index_x, index_y, index_z. If near the edge of the grid, the array size that the 
        derivative is calculated on is reduced.

        :param index_x: The lower index of the x grid cell to evaluate.
        :param index_y: The lower index of the y grid cell to evaluate.
        :param index_z: The lower index of the z grid cell to evaluate.

        :param rescale_norm_x: A boolean as whether to rescale to the delta before x[index_x] or after (default).
        :param rescale_norm_y: A boolean as whether to rescale to the delta before y[index_y] or after (default).
        
        """

        cdef double dfdn

        if index_x == self._last_index_x:

            if index_y == self._last_index_y:
                dfdn = self._derivative_d2fdxdy_edge_xy(index_x - 1, index_y - 1, index_z)

            elif index_y == 0:
                dfdn = self._derivative_d2fdxdy_edge_xy(index_x - 1, index_y, index_z)

            else:
                dfdn = self._derivative_d2fdxdy_edge_x(index_x - 1, index_y - 1, index_z)

        elif index_x == 0:

            if index_y == self._last_index_y:
                dfdn = self._derivative_d2fdxdy_edge_xy(index_x, index_y - 1, index_z)

            elif index_y == 0:
                dfdn = self._derivative_d2fdxdy_edge_xy(index_x, index_y, index_z)

            else:
                dfdn = self._derivative_d2fdxdy_edge_x(index_x, index_y - 1, index_z)

        else:

            if index_y == self._last_index_y:
                dfdn = self._derivative_d2fdxdy_edge_y(index_x - 1, index_y - 1, index_z)

            elif index_y == 0:
                dfdn = self._derivative_d2fdxdy_edge_y(index_x - 1, index_y, index_z)

            else:
                dfdn = self._derivative_d2fdxdy(index_x - 1, index_y - 1, index_z)

        if rescale_norm_x:
            if not (index_x == 0 or index_x == self._last_index_x):
                dfdn = rescale_lower_normalisation(dfdn,  self._x[index_x - 1], self._x[index_x], self._x[index_x + 1])

        if rescale_norm_y:
            if not (index_y == 0 or index_y == self._last_index_y):
                dfdn = rescale_lower_normalisation(dfdn,  self._y[index_y - 1], self._y[index_y], self._y[index_y + 1])

        return dfdn

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate_d2fdxdz(self, int index_x, int index_y, int index_z, bint rescale_norm_x, bint rescale_norm_z) except? -1e999:
        """
        Evaluate d2f/dxdz at index_x, index_y, index_z. If near the edge of the grid, the array size that the 
        derivative is calculated on is reduced.

        :param index_x: The lower index of the x grid cell to evaluate.
        :param index_y: The lower index of the y grid cell to evaluate.
        :param index_z: The lower index of the z grid cell to evaluate.

        :param rescale_norm_x: A boolean as whether to rescale to the delta before x[index_x] or after (default).
        :param rescale_norm_z: A boolean as whether to rescale to the delta before z[index_z] or after (default).
        
        """

        cdef double dfdn

        if index_x == self._last_index_x:

            if index_z == self._last_index_z:
                dfdn = self._derivative_d2fdxdz_edge_xz(index_x - 1, index_y, index_z - 1)

            elif index_z == 0:
                dfdn = self._derivative_d2fdxdz_edge_xz(index_x - 1, index_y, index_z)

            else:
                dfdn = self._derivative_d2fdxdz_edge_x(index_x - 1, index_y, index_z - 1)

        elif index_x == 0:

            if index_z == self._last_index_z:
                dfdn = self._derivative_d2fdxdz_edge_xz(index_x, index_y, index_z - 1)

            elif index_z == 0:
                dfdn = self._derivative_d2fdxdz_edge_xz(index_x, index_y, index_z)

            else:
                dfdn = self._derivative_d2fdxdz_edge_x(index_x, index_y, index_z - 1)

        else:

            if index_z == self._last_index_z:
                dfdn = self._derivative_d2fdxdz_edge_z(index_x - 1, index_y, index_z - 1)

            elif index_z == 0:
                dfdn = self._derivative_d2fdxdz_edge_z(index_x - 1, index_y, index_z)

            else:
                dfdn = self._derivative_d2fdxdz(index_x - 1, index_y, index_z - 1)

        if rescale_norm_x:
            if not (index_x == 0 or index_x == self._last_index_x):
                dfdn = rescale_lower_normalisation(dfdn,  self._x[index_x - 1], self._x[index_x], self._x[index_x + 1])

        if rescale_norm_z:
            if not (index_z == 0 or index_z == self._last_index_z):
                dfdn = rescale_lower_normalisation(dfdn,  self._z[index_z - 1], self._z[index_z], self._z[index_z + 1])

        return dfdn

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate_d2fdydz(self, int index_x, int index_y, int index_z, bint rescale_norm_y, bint rescale_norm_z) except? -1e999:
        """
        Evaluate d2f/dydz at index_x, index_y, index_z. If near the edge of the grid, the array size that the 
        derivative is calculated on is reduced.
        
        :param index_x: The lower index of the x grid cell to evaluate.
        :param index_y: The lower index of the y grid cell to evaluate.
        :param index_z: The lower index of the z grid cell to evaluate.

        :param rescale_norm_y: A boolean as whether to rescale to the delta before y[index_y] or after (default).
        :param rescale_norm_z: A boolean as whether to rescale to the delta before z[index_z] or after (default).
        """

        cdef double dfdn

        if index_y == self._last_index_y:

            if index_z == self._last_index_z:
                dfdn = self._derivative_d2fdydz_edge_yz(index_x, index_y - 1, index_z - 1)

            elif index_z == 0:
                dfdn = self._derivative_d2fdydz_edge_yz(index_x, index_y - 1, index_z)

            else:
                dfdn = self._derivative_d2fdydz_edge_y(index_x, index_y - 1, index_z - 1)

        elif index_y == 0:

            if index_z == self._last_index_z:
                dfdn = self._derivative_d2fdydz_edge_yz(index_x, index_y, index_z - 1)

            elif index_z == 0:
                dfdn = self._derivative_d2fdydz_edge_yz(index_x, index_y, index_z)

            else:
                dfdn = self._derivative_d2fdydz_edge_y(index_x, index_y, index_z - 1)

        else:

            if index_z == self._last_index_z:
                dfdn = self._derivative_d2fdydz_edge_z(index_x, index_y - 1, index_z - 1)

            elif index_z == 0:
                dfdn = self._derivative_d2fdydz_edge_z(index_x, index_y - 1, index_z)

            else:
                dfdn = self._derivative_d2fdydz(index_x, index_y - 1, index_z - 1)

        if rescale_norm_y:
            if not (index_y == 0 or index_y == self._last_index_y):
                dfdn = rescale_lower_normalisation(dfdn,  self._y[index_y - 1], self._y[index_y], self._y[index_y + 1])

        if rescale_norm_z:
            if not (index_z == 0 or index_z == self._last_index_z):
                dfdn = rescale_lower_normalisation(dfdn,  self._z[index_z - 1], self._z[index_z], self._z[index_z + 1])

        return dfdn

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate_d3fdxdydz(self, int index_x, int index_y, int index_z, bint rescale_norm_x, bint rescale_norm_y, bint rescale_norm_z) except? -1e999:
        """
        Evaluate d3f/dxdydz at index_x, index_y, index_z. If near the edge of the grid, the array size that the 
        derivative is calculated on is reduced.

        :param index_x: The lower index of the x grid cell to evaluate.
        :param index_y: The lower index of the y grid cell to evaluate.
        :param index_z: The lower index of the z grid cell to evaluate.

        :param rescale_norm_x: A boolean as whether to rescale to the delta before x[index_x] or after (default).
        :param rescale_norm_y: A boolean as whether to rescale to the delta before y[index_y] or after (default).
        :param rescale_norm_z: A boolean as whether to rescale to the delta before z[index_z] or after (default).
        """

        cdef double dfdn

        if index_x == self._last_index_x:

            if index_y == self._last_index_y:

                if index_z == self._last_index_z:
                    dfdn = self._derivative_d3fdxdydz_edge_xyz(index_x - 1, index_y - 1, index_z - 1)

                elif index_z == 0:
                    dfdn = self._derivative_d3fdxdydz_edge_xyz(index_x - 1, index_y - 1, index_z)

                else:
                    dfdn = self._derivative_d3fdxdydz_edge_xy(index_x - 1, index_y - 1, index_z - 1)

            elif index_y == 0:

                if index_z == self._last_index_z:
                    dfdn = self._derivative_d3fdxdydz_edge_xyz(index_x - 1, index_y, index_z - 1)

                elif index_z == 0:
                    dfdn = self._derivative_d3fdxdydz_edge_xyz(index_x - 1, index_y, index_z)

                else:
                    dfdn = self._derivative_d3fdxdydz_edge_xy(index_x - 1, index_y, index_z - 1)

            else:

                if index_z == self._last_index_z:
                    dfdn = self._derivative_d3fdxdydz_edge_xz(index_x - 1, index_y - 1, index_z - 1)

                elif index_z == 0:
                    dfdn = self._derivative_d3fdxdydz_edge_xz(index_x - 1, index_y - 1, index_z)

                else:
                    dfdn = self._derivative_d3fdxdydz_edge_x(index_x - 1, index_y - 1, index_z - 1)

        elif index_x == 0:

            if index_y == self._last_index_y:

                if index_z == self._last_index_z:
                    dfdn = self._derivative_d3fdxdydz_edge_xyz(index_x, index_y - 1, index_z - 1)

                elif index_z == 0:
                    dfdn = self._derivative_d3fdxdydz_edge_xyz(index_x, index_y - 1, index_z)

                else:
                    dfdn = self._derivative_d3fdxdydz_edge_xy(index_x, index_y - 1, index_z - 1)

            elif index_y == 0:

                if index_z == self._last_index_z:
                    dfdn = self._derivative_d3fdxdydz_edge_xyz(index_x, index_y, index_z - 1)

                elif index_z == 0:
                    dfdn = self._derivative_d3fdxdydz_edge_xyz(index_x, index_y, index_z)

                else:
                    dfdn = self._derivative_d3fdxdydz_edge_xy(index_x, index_y, index_z - 1)

            else:

                if index_z == self._last_index_z:
                    dfdn = self._derivative_d3fdxdydz_edge_xz(index_x, index_y - 1, index_z - 1)

                elif index_z == 0:
                    dfdn = self._derivative_d3fdxdydz_edge_xz(index_x, index_y - 1, index_z)

                else:
                    dfdn = self._derivative_d3fdxdydz_edge_x(index_x, index_y - 1, index_z - 1)

        else:

            if index_y == self._last_index_y:

                if index_z == self._last_index_z:
                    dfdn = self._derivative_d3fdxdydz_edge_yz(index_x - 1, index_y - 1, index_z - 1)

                elif index_z == 0:
                    dfdn = self._derivative_d3fdxdydz_edge_yz(index_x - 1, index_y - 1, index_z)

                else:
                    dfdn = self._derivative_d3fdxdydz_edge_y(index_x - 1, index_y - 1, index_z - 1)

            elif index_y == 0:

                if index_z == self._last_index_z:
                    dfdn = self._derivative_d3fdxdydz_edge_yz(index_x - 1, index_y, index_z - 1)

                elif index_z == 0:
                    dfdn = self._derivative_d3fdxdydz_edge_yz(index_x - 1, index_y, index_z)

                else:
                    dfdn = self._derivative_d3fdxdydz_edge_y(index_x - 1, index_y, index_z - 1)

            else:

                if index_z == self._last_index_z:
                    dfdn = self._derivative_d3fdxdydz_edge_z(index_x - 1, index_y - 1, index_z - 1)

                elif index_z == 0:
                    dfdn = self._derivative_d3fdxdydz_edge_z(index_x - 1, index_y - 1, index_z)

                else:
                    dfdn = self._derivative_d3fdxdydz(index_x - 1, index_y - 1, index_z - 1)

        if rescale_norm_x:
            if not (index_x == 0 or index_x == self._last_index_x):
                dfdn = rescale_lower_normalisation(dfdn,  self._x[index_x - 1], self._x[index_x], self._x[index_x + 1])

        if rescale_norm_y:
            if not (index_y == 0 or index_y == self._last_index_y):
                dfdn = rescale_lower_normalisation(dfdn,  self._y[index_y - 1], self._y[index_y], self._y[index_y + 1])

        if rescale_norm_z:
            if not (index_z == 0 or index_z == self._last_index_z):
                dfdn = rescale_lower_normalisation(dfdn,  self._z[index_z - 1], self._z[index_z], self._z[index_z + 1])

        return dfdn

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_dfdx(self, int lower_index_x, int slice_index_y, int slice_index_z) except? -1e999:
        """
        Calculate df/dx on an unevenly spaced grid as a 2nd order approximation.

        This calculation is the same as in 1D, and is made by writing out 2 second order taylor expansions at 
        f(x + dx0, ...),  f(x - dx1, ...) then multiplying the first by dx1**2, the second by dx0**2 and subtracting 
        the second from the first to leave df/dx terms. 

        For unit square normalisation, dx0 = 1, the denominator simplifies, and dx1 
        is rescaled to be normalised to distance dx0.
        """

        cdef double x1_n, x1_n2

        x1_n = (self._x[lower_index_x + 1] - self._x[lower_index_x]) / (self._x[lower_index_x + 2] - self._x[lower_index_x + 1])
        x1_n2 = x1_n * x1_n
        return (self._f[lower_index_x + 2, slice_index_y, slice_index_z] * x1_n2
                - self._f[lower_index_x, slice_index_y, slice_index_z]
                - self._f[lower_index_x + 1, slice_index_y, slice_index_z] * (x1_n2 - 1.)) / (x1_n + x1_n2)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_dfdx_edge(self, int lower_index_x, int slice_index_y, int slice_index_z) except? -1e999:
        """
        Calculate df/dx on an unevenly spaced grid as a 1st order approximation. Valid at the x edge of the array.

        This calculation is the same as in 1D, and is made by writing out 2 first order taylor expansions at 
        f(x + dx0, ...),  f(x, ...) and subtracting the second from the first.

        For unit square normalisation, dx0 = 1, the denominator simplifies.
        """

        return self._f[lower_index_x + 1, slice_index_y, slice_index_z] - self._f[lower_index_x, slice_index_y, slice_index_z]

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_dfdy(self, int slice_index_x, int lower_index_y, int slice_index_z) except? -1e999:
        """
        Calculate df/dy on an unevenly spaced grid as a 2nd order approximation.

        This calculation is the same as in 1D, and is made by writing out 2 second order taylor expansions at 
        f(y + dy0, ...),  f(y - dy1, ...) then multiplying the first by dy1**2, the second by dy0**2 and subtracting 
        the second from the first to leave df/dy terms. 

        For unit square normalisation, dy0 = 1, the denominator simplifies, and dy1 
        is rescaled to be normalised to distance dy0.
        """

        cdef double x1_n, x1_n2

        x1_n = (self._y[lower_index_y + 1] - self._y[lower_index_y]) / (self._y[lower_index_y + 2] - self._y[lower_index_y + 1])
        x1_n2 = x1_n * x1_n
        return (self._f[slice_index_x, lower_index_y + 2, slice_index_z] * x1_n2
                - self._f[slice_index_x, lower_index_y, slice_index_z]
                - self._f[slice_index_x, lower_index_y + 1, slice_index_z] * (x1_n2 - 1.)) / (x1_n + x1_n2)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_dfdy_edge(self, int slice_index_x, int lower_index_y, int slice_index_z) except? -1e999:
        """
        Calculate df/dy on an unevenly spaced grid as a 1st order approximation. Valid at the y edge of the array.

        This calculation is the same as in 1D, and is made by writing out 2 first order taylor expansions at 
        f(y + dy0, ...),  f(y, ...) and subtracting the second from the first.

        For unit square normalisation, dy0 = 1, the denominator simplifies.

        """

        return self._f[slice_index_x, lower_index_y + 1, slice_index_z] - self._f[slice_index_x, lower_index_y, slice_index_z]

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_dfdz(self, int slice_index_x, int slice_index_y, int lower_index_z) except? -1e999:
        """
        Calculate df/dz on an unevenly spaced grid as a 2nd order approximation.

        This calculation is the same as in 1D, and is made by writing out 2 second order taylor expansions at 
        f(z + dz0, ...),  f(z - dz1, ...) then multiplying the first by dz1**2, the second by dz0**2 and subtracting 
        the second from the first to leave df/dz terms. 

        For unit square normalisation, dz0 = 1, the denominator simplifies, and dz1 
        is rescaled to be normalised to distance dz0.
        """

        cdef double x1_n, x1_n2

        x1_n = (self._z[lower_index_z + 1] - self._z[lower_index_z]) / (self._z[lower_index_z + 2] - self._z[lower_index_z + 1])
        x1_n2 = x1_n * x1_n
        return (self._f[slice_index_x, slice_index_y, lower_index_z + 2] * x1_n2
                - self._f[slice_index_x, slice_index_y, lower_index_z]
                - self._f[slice_index_x, slice_index_y, lower_index_z + 1] * (x1_n2 - 1.)) / (x1_n + x1_n2)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_dfdz_edge(self, int slice_index_x, int slice_index_y, int lower_index_z) except? -1e999:
        """
        Calculate df/dz on an unevenly spaced grid as a 1st order approximation. Valid at the z edge of the array.

        This calculation is the same as in 1D, and is made by writing out 2 first order taylor expansions at 
        f(z + dz0, ...),  f(z, ...) and subtracting the second from the first.

        For unit square normalisation, dz0 = 1, the denominator simplifies.

        """

        return self._f[slice_index_x, slice_index_y, lower_index_z + 1] - self._f[slice_index_x, slice_index_y, lower_index_z]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_d2fdxdy_edge_xy(self, int lower_index_x, int lower_index_y, int slice_index_z) except? -1e999:
        """
        Calculate d2f/dxdy on an unevenly spaced grid as a 2nd order approximation. Valid at the xy edge of the array.

        This calculation is the same as in 2D, but just handles second order taylor expansions at 4 different 
        points around f(x, y, z) for every combination of +dx0, +dy0 or -dx1, -dy1 are written out (where both of -dx1 
        and -dy1 are 0 because it is near an edge of the spline knot array). The Taylor expansions 
        are subtracted from each other that have the same delta x positions, but higher/lower dy (dy0, 
        dy1 respectively). The remaining 2 equations have higher/lower dy (dy1, dy2 respectively), so finally 
        subtract the higher delta y from the lower delta y to cancel all terms leaving the d2f/dxdy.

        For unit square normalisation, dx0 = 1 and dy0 = 1, the denominator simplifies.
        """

        return self._f[lower_index_x + 1, lower_index_y + 1, slice_index_z] \
               - self._f[lower_index_x, lower_index_y + 1, slice_index_z] \
               - self._f[lower_index_x + 1, lower_index_y, slice_index_z] \
               + self._f[lower_index_x, lower_index_y, slice_index_z]

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_d2fdxdy_edge_x(self, int lower_index_x, int lower_index_y, int slice_index_z) except? -1e999:
        """
        Calculate d2f/dxdy on an unevenly spaced grid as a 2nd order approximation. Valid at the x edge of the array.

        This calculation is the same as in 2D, but just handles second order taylor expansions at 4 different 
        points around f(x, y, z) for every combination of +dx0, +dy0 or -dx1, -dy1 are written out (where -dx1 
        is 0 because it is near an edge of the spline knot array). The Taylor expansions 
        are subtracted from each other that have the same delta x positions, but higher/lower dy (dy0, 
        dy1 respectively). The remaining 2 equations have higher/lower dy (dy0, dy1 respectively), so finally 
        subtract the higher delta y from the lower delta y to cancel all terms leaving the d2f/dxdy.

        For unit square normalisation, dx0 = 1 and dy0 = 1, the denominator simplifies, and dy1 is rescaled to 
        be normalised to distance dy0.
        """

        cdef double dy

        dy = (self._y[lower_index_y + 1] - self._y[lower_index_y]) / (self._y[lower_index_y + 2] - self._y[lower_index_y + 1])
        return (self._f[lower_index_x + 1, lower_index_y + 2, slice_index_z]
                - self._f[lower_index_x, lower_index_y + 2, slice_index_z]
                - self._f[lower_index_x + 1, lower_index_y, slice_index_z]
                + self._f[lower_index_x, lower_index_y, slice_index_z]) / (1. + dy)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_d2fdxdy_edge_y(self, int lower_index_x, int lower_index_y, int slice_index_z) except? -1e999:
        """
        Calculate d2f/dxdy on an unevenly spaced grid as a 2nd order approximation. Valid at the y edge of the array.

        This calculation is the same as in 2D, but just handles second order taylor expansions at 4 different 
        points around f(x, y, z) for every combination of +dx0, +dy0 or -dx1, -dy1 are written out (where -dy1 
        is 0 because it is near an edge of the spline knot array). The Taylor expansions 
        are subtracted from each other that have the same delta x positions, but higher/lower dy (dy0, 
        dy1 respectively). The remaining 2 equations have higher/lower dy (dy0, dy1 respectively), so finally 
        subtract the higher delta y from the lower delta y to cancel all terms leaving the d2f/dxdy.

        For unit square normalisation, dx0 = 1 and dy0 = 1, the denominator simplifies, and dx1 is rescaled to 
        be normalised to distance dx0.
        """

        cdef double dx

        dx = (self._x[lower_index_x + 1] - self._x[lower_index_x]) / (self._x[lower_index_x + 2] - self._x[lower_index_x + 1])
        return (self._f[lower_index_x + 2, lower_index_y + 1, slice_index_z]
                - self._f[lower_index_x, lower_index_y + 1, slice_index_z]
                - self._f[lower_index_x + 2, lower_index_y, slice_index_z]
                + self._f[lower_index_x, lower_index_y, slice_index_z]) / (1. + dx)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_d2fdxdy(self, int lower_index_x, int lower_index_y, int slice_index_z) except? -1e999:
        """
        Calculate d2f/dxdy on an unevenly spaced grid as a 2nd order approximation.

        This calculation is the same as in 2D, but just handles second order taylor expansions at 4 different 
        points around f(x, y, z) for every combination of +dx0, +dy0 or -dx1, -dy1 are written out. The Taylor 
        expansions are subtracted from each other that have the same delta x positions, but higher/lower dy (dy0, 
        dy1 respectively). The remaining 2 equations have higher/lower dy (dy1, dy2 respectively), so finally 
        subtract the higher delta y from the lower delta n to cancel all terms leaving the d2f/dxdy.

        For unit square normalisation, dx0 = 1 and dy0 = 1, the denominator simplifies, and dx1 and dy1 
        are rescaled to be normalised to distance dx0 and dy0.
        """

        cdef double dx, dy

        dx = (self._x[lower_index_x + 1] - self._x[lower_index_x]) / (self._x[lower_index_x + 2] - self._x[lower_index_x + 1])
        dy = (self._y[lower_index_y + 1] - self._y[lower_index_y]) / (self._y[lower_index_y + 2] - self._y[lower_index_y + 1])
        return (self._f[lower_index_x + 2, lower_index_y + 2, slice_index_z]
                - self._f[lower_index_x, lower_index_y + 2, slice_index_z]
                - self._f[lower_index_x + 2, lower_index_y, slice_index_z]
                + self._f[lower_index_x, lower_index_y, slice_index_z]) / (1. + dx + dy + dx * dy)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_d2fdxdz_edge_xz(self, int lower_index_x, int slice_index_y, int lower_index_z) except? -1e999:
        """
        Calculate d2f/dxdz on an unevenly spaced grid as a 2nd order approximation. Valid at the xz edge of the array.

        This calculation is the same as in 2D, but just handles second order taylor expansions at 4 different 
        points around f(x, y, z) for every combination of +dx0, +dz0 or -dx1, -dz1 are written out (where both of -dx1 
        and -dz1 are 0 because it is near an edge of the spline knot array). The Taylor expansions 
        are subtracted from each other that have the same delta x positions, but higher/lower dz (dz0, 
        dz1 respectively). The remaining 2 equations have higher/lower dz (dz1, dz2 respectively), so finally 
        subtract the higher delta z from the lower delta z to cancel all terms leaving the d2f/dxdz.

        For unit square normalisation, dx0 = 1 and dz0 = 1, the denominator simplifies.
        """

        return self._f[lower_index_x + 1, slice_index_y, lower_index_z + 1] \
               - self._f[lower_index_x, slice_index_y, lower_index_z + 1] \
               - self._f[lower_index_x + 1, slice_index_y, lower_index_z] \
               + self._f[lower_index_x, slice_index_y, lower_index_z]

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_d2fdxdz_edge_x(self, int lower_index_x, int slice_index_y, int lower_index_z) except? -1e999:
        """
        Calculate d2f/dxdz on an unevenly spaced grid as a 2nd order approximation. Valid at the x edge of the array.

        This calculation is the same as in 2D, but just handles second order taylor expansions at 4 different 
        points around f(x, y, z) for every combination of +dx0, +dz0 or -dx1, -dz1 are written out (where -dx1 
        is 0 because it is near an edge of the spline knot array). The Taylor expansions 
        are subtracted from each other that have the same delta x positions, but higher/lower dz (dz0, 
        dz1 respectively). The remaining 2 equations have higher/lower dz (dz0, dz1 respectively), so finally 
        subtract the higher delta z from the lower delta z to cancel all terms leaving the d2f/dxdz.

        For unit square normalisation, dx0 = 1 and dz0 = 1, the denominator simplifies, and dz1 is rescaled to 
        be normalised to distance dz0.
        """
        cdef double dz

        dz = (self._z[lower_index_z + 1] - self._z[lower_index_z]) / (self._z[lower_index_z + 2] - self._z[lower_index_z + 1])
        return (self._f[lower_index_x + 1, slice_index_y, lower_index_z + 2]
                - self._f[lower_index_x, slice_index_y, lower_index_z + 2]
                - self._f[lower_index_x + 1, slice_index_y, lower_index_z]
                + self._f[lower_index_x, slice_index_y, lower_index_z]) / (1. + dz)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_d2fdxdz_edge_z(self, int lower_index_x, int slice_index_y, int lower_index_z) except? -1e999:
        """
        Calculate d2f/dxdz on an unevenly spaced grid as a 2nd order approximation. Valid at the z edge of the array.

        This calculation is the same as in 2D, but just handles second order taylor expansions at 4 different 
        points around f(x, y, z) for every combination of +dx0, +dz0 or -dx1, -dz1 are written out (where -dz1 
        is 0 because it is near an edge of the spline knot array). The Taylor expansions 
        are subtracted from each other that have the same delta x positions, but higher/lower dz (dz0, 
        dz1 respectively). The remaining 2 equations have higher/lower dz (dz0, dz1 respectively), so finally 
        subtract the higher delta z from the lower delta z to cancel all terms leaving the d2f/dxdz.

        For unit square normalisation, dx0 = 1 and dz0 = 1, the denominator simplifies, and dx1 is rescaled to 
        be normalised to distance dx0.
        """

        cdef double dx

        dx = (self._x[lower_index_x + 1] - self._x[lower_index_x]) / (self._x[lower_index_x + 2] - self._x[lower_index_x + 1])
        return (self._f[lower_index_x + 2, slice_index_y, lower_index_z + 1]
                - self._f[lower_index_x, slice_index_y, lower_index_z + 1]
                - self._f[lower_index_x + 2, slice_index_y, lower_index_z]
                + self._f[lower_index_x, slice_index_y, lower_index_z]) / (1. + dx)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_d2fdxdz(self, int lower_index_x, int slice_index_y, int lower_index_z) except? -1e999:
        """
        Calculate d2f/dxdz on an unevenly spaced grid as a 2nd order approximation.

        This calculation is the same as in 2D, but just handles second order taylor expansions at 4 different 
        points around f(x, y, z) for every combination of +dx0, +dz0 or -dx1, -dz1 are written out. The Taylor 
        expansions are subtracted from each other that have the same delta x positions, but higher/lower dz (dz0, 
        dz1 respectively). The remaining 2 equations have higher/lower dz (dz1, dz2 respectively), so finally 
        subtract the higher delta z from the lower delta n to cancel all terms leaving the d2f/dxdz.

        For unit square normalisation, dx0 = 1 and dz0 = 1, the denominator simplifies, and dx1 and dz1 
        are rescaled to be normalised to distance dx0 and dz0.
        """

        cdef double dx, dz

        dx = (self._x[lower_index_x + 1] - self._x[lower_index_x]) / (self._x[lower_index_x + 2] - self._x[lower_index_x + 1])
        dz = (self._z[lower_index_z + 1] - self._z[lower_index_z]) / (self._z[lower_index_z + 2] - self._z[lower_index_z + 1])
        return (self._f[lower_index_x + 2, slice_index_y, lower_index_z + 2]
                - self._f[lower_index_x, slice_index_y, lower_index_z + 2]
                - self._f[lower_index_x + 2, slice_index_y, lower_index_z]
                + self._f[lower_index_x, slice_index_y, lower_index_z]) / (1. + dx + dz + dx * dz)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_d2fdydz_edge_yz(self, int slice_index_x, int lower_index_y, int lower_index_z) except? -1e999:
        """
        Calculate d2f/dydz on an unevenly spaced grid as a 2nd order approximation. Valid at the yz edge of the array.

        This calculation is the same as in 2D, but just handles second order taylor expansions at 4 different 
        points around f(x, y, z) for every combination of +dy0, +dz0 or -dy1, -dz1 are written out (where both of -dy1 
        and -dz1 are 0 because it is near an edge of the spline knot array). The Taylor expansions 
        are subtracted from each other that have the same delta y positions, but higher/lower dz (dz0, 
        dz1 respectively). The remaining 2 equations have higher/lower dz (dz1, dz2 respectively), so finally 
        subtract the higher delta z from the lower delta z to cancel all terms leaving the d2f/dydz.

        For unit square normalisation, dy0 = 1 and dz0 = 1, the denominator simplifies.
        """

        return self._f[slice_index_x, lower_index_y + 1, lower_index_z + 1] \
               - self._f[slice_index_x, lower_index_y, lower_index_z + 1] \
               - self._f[slice_index_x, lower_index_y + 1, lower_index_z] \
               + self._f[slice_index_x, lower_index_y, lower_index_z]

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_d2fdydz_edge_y(self, int slice_index_x, int lower_index_y, int lower_index_z) except? -1e999:
        """
        Calculate d2f/dydz on an unevenly spaced grid as a 2nd order approximation. Valid at the y edge of the array.

        This calculation is the same as in 2D, but just handles second order taylor expansions at 4 different 
        points around f(x, y, z) for every combination of +dy0, +dz0 or -dy1, -dz1 are written out (where -dy1 
        is 0 because it is near an edge of the spline knot array). The Taylor expansions 
        are subtracted from each other that have the same delta y positions, but higher/lower dz (dz0, 
        dz1 respectively). The remaining 2 equations have higher/lower dz (dz0, dz1 respectively), so finally 
        subtract the higher delta z from the lower delta z to cancel all terms leaving the d2f/dydz.

        For unit square normalisation, dy0 = 1 and dz0 = 1, the denominator simplifies, and dz1 is rescaled to 
        be normalised to distance dz0.
        """

        cdef double dz

        dz = (self._z[lower_index_z + 1] - self._z[lower_index_z]) / (self._z[lower_index_z + 2] - self._z[lower_index_z + 1])
        return (self._f[slice_index_x, lower_index_y + 1, lower_index_z + 2]
                - self._f[slice_index_x, lower_index_y, lower_index_z + 2]
                - self._f[slice_index_x, lower_index_y + 1, lower_index_z]
                + self._f[slice_index_x, lower_index_y, lower_index_z]) / (1. + dz)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_d2fdydz_edge_z(self, int slice_index_x, int lower_index_y, int lower_index_z) except? -1e999:
        """
        Calculate d2f/dydz on an unevenly spaced grid as a 2nd order approximation. Valid at the z edge of the array.

        This calculation is the same as in 2D, but just handles second order taylor expansions at 4 different 
        points around f(x, y, z) for every combination of +dy0, +dz0 or -dy1, -dz1 are written out (where -dz1 
        is 0 because it is near an edge of the spline knot array). The Taylor expansions 
        are subtracted from each other that have the same delta y positions, but higher/lower dz (dz0, 
        dz1 respectively). The remaining 2 equations have higher/lower dz (dz0, dz1 respectively), so finally 
        subtract the higher delta z from the lower delta z to cancel all terms leaving the d2f/dydz.

        For unit square normalisation, dy0 = 1 and dz0 = 1, the denominator simplifies, and dy1 is rescaled to 
        be normalised to distance dy0.
        """

        cdef double dy

        dy = (self._y[lower_index_y + 1] - self._y[lower_index_y]) / (self._y[lower_index_y + 2] - self._y[lower_index_y + 1])
        return (self._f[slice_index_x, lower_index_y + 2, lower_index_z + 1]
                - self._f[slice_index_x, lower_index_y, lower_index_z + 1]
                - self._f[slice_index_x, lower_index_y + 2, lower_index_z]
                + self._f[slice_index_x, lower_index_y, lower_index_z]) / (1. + dy)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_d2fdydz(self, int slice_index_x, int lower_index_y, int lower_index_z) except? -1e999:
        """
        Calculate d2f/dydz on an unevenly spaced grid as a 2nd order approximation.

        This calculation is the same as in 2D, but just handles second order taylor expansions at 4 different 
        points around f(x, y, z) for every combination of +dy0, +dz0 or -dy1, -dz1 are written out. The Taylor 
        expansions are subtracted from each other that have the same delta y positions, but higher/lower dz (dz0, 
        dz1 respectively). The remaining 2 equations have higher/lower dz (dz1, dz2 respectively), so finally 
        subtract the higher delta z from the lower delta n to cancel all terms leaving the d2f/dydz.

        For unit square normalisation, dy0 = 1 and dz0 = 1, the denominator simplifies, and dy1 and dz1 
        are rescaled to be normalised to distance dy0 and dz0.
        """
        cdef double dy, dz

        dy = (self._y[lower_index_y + 1] - self._y[lower_index_y]) / (self._y[lower_index_y + 2] - self._y[lower_index_y + 1])
        dz = (self._z[lower_index_z + 1] - self._z[lower_index_z]) / (self._z[lower_index_z + 2] - self._z[lower_index_z + 1])
        return (self._f[slice_index_x, lower_index_y + 2, lower_index_z + 2]
                - self._f[slice_index_x, lower_index_y, lower_index_z + 2]
                - self._f[slice_index_x, lower_index_y + 2, lower_index_z]
                + self._f[slice_index_x, lower_index_y, lower_index_z]) / (1. + dy + dz + dy * dz)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_d3fdxdydz(self, int lower_index_x, int lower_index_y, int lower_index_z) except? -1e999:
        """
        Calculate d3f/dxdydz on an unevenly spaced grid as a 3rd order approximation.

        A third order taylor expansions at 8 different points around f(x, y, z) for every combination of +dx0, +dy0, +dz0 or
        -dx1, -dy1, -dz1 are written out. The Taylor expansions are subtracted from each other that have the same 
        delta y and delta z positions, but higher/lower dx (dx0, dx1 respectively). The resulting equations are 
        subtracted from each other where they have the same delta z positions, but higher/lower dy 
        (dy0, dy1 respectively). The remaining 2 equations have higher/lower dz (dz1, dz2 respectively), so similarly 
        subtract the higher delta z from the lower delta z to cancel all terms.

        For unit square normalisation, dy0 = 1, dz0 = 1 and dx0 = 1, the denominator simplifies, and dx1, dy1 and dz1 
        are rescaled to be normalised to distance dx0, dy0, dz0.
        """

        cdef double dx1, dy1, dz1

        dx1 = (self._x[lower_index_x + 1] - self._x[lower_index_x]) / (self._x[lower_index_x + 2] - self._x[lower_index_x + 1])
        dy1 = (self._y[lower_index_y + 1] - self._y[lower_index_y]) / (self._y[lower_index_y + 2] - self._y[lower_index_y + 1])
        dz1 = (self._z[lower_index_z + 1] - self._z[lower_index_z]) / (self._z[lower_index_z + 2] - self._z[lower_index_z + 1])
        return (self._f[lower_index_x + 2, lower_index_y + 2, lower_index_z + 2]
                - self._f[lower_index_x, lower_index_y + 2, lower_index_z + 2]
                - self._f[lower_index_x + 2, lower_index_y, lower_index_z + 2]
                + self._f[lower_index_x, lower_index_y, lower_index_z + 2]
                - self._f[lower_index_x + 2, lower_index_y + 2, lower_index_z]
                + self._f[lower_index_x, lower_index_y + 2, lower_index_z]
                + self._f[lower_index_x + 2, lower_index_y, lower_index_z]
                - self._f[lower_index_x, lower_index_y, lower_index_z]) / (1. + dx1 + dy1 + dz1 + dx1 * dy1 + dx1 * dz1 + dy1 * dz1 + dx1 * dy1 * dz1)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_d3fdxdydz_edge_x(self, int lower_index_x, int lower_index_y, int lower_index_z) except? -1e999:
        """
        Calculate d3f/dxdydz on an unevenly spaced grid as a 3rd order approximation. This is used when on the border of
        the spline knot array in x.

        A third order taylor expansions at 8 different points around f(x, y, z) for every combination of +dx0, +dy0, +dz0 or
        -dx1=0, -dy1, -dz1 (where dx1 = 0 because we are looking at the edge in x), at the same location as the spline knot we are looking 
        at), are written out. The Taylor expansions are subtracted from each other that have the same 
        delta y and delta z positions, but higher/lower dx (dx0, dx1 respectively). The resulting equations are 
        subtracted from each other where they have the same delta z positions, but higher/lower dy 
        (dy0, dy1 respectively). The remaining 2 equations have higher/lower dz (dz1, dz2 respectively), so similarly 
        subtract the higher delta z from the lower delta z to cancel all terms.

        For unit square normalisation, dy0 = 1, dz0 = 1 and dx0 = 1, the denominator simplifies, and dy1 and dz1 
        are rescaled to be normalised to distance dy0, dz0.
        """

        cdef double dy1, dz1

        dy1 = (self._y[lower_index_y + 1] - self._y[lower_index_y]) / (self._y[lower_index_y + 2] - self._y[lower_index_y + 1])
        dz1 = (self._z[lower_index_z + 1] - self._z[lower_index_z]) / (self._z[lower_index_z + 2] - self._z[lower_index_z + 1])
        return (self._f[lower_index_x + 1, lower_index_y + 2, lower_index_z + 2]
                - self._f[lower_index_x, lower_index_y + 2, lower_index_z + 2]
                - self._f[lower_index_x + 1, lower_index_y, lower_index_z + 2]
                + self._f[lower_index_x, lower_index_y, lower_index_z + 2]
                - self._f[lower_index_x + 1, lower_index_y + 2, lower_index_z]
                + self._f[lower_index_x, lower_index_y + 2, lower_index_z]
                + self._f[lower_index_x + 1, lower_index_y, lower_index_z]
                - self._f[lower_index_x, lower_index_y, lower_index_z]) / (1. + dy1 + dz1 + dy1 * dz1)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_d3fdxdydz_edge_y(self, int lower_index_x, int lower_index_y, int lower_index_z) except? -1e999:
        """
        Calculate d3f/dxdydz on an unevenly spaced grid as a 3rd order approximation. This is used when on the border of
        the spline knot array in y.

        A third order taylor expansions at 8 different points around f(x, y, z) for every combination of +dx0, +dy0, +dz0 or
        -dx1, -dy1=0, -dz1 (where dy1 = 0 because we are looking at the edge in y), at the same location as the spline knot we are looking 
        at), are written out. The Taylor expansions are subtracted from each other that have the same 
        delta y and delta z positions, but higher/lower dx (dx0, dx1 respectively). The resulting equations are 
        subtracted from each other where they have the same delta z positions, but higher/lower dy 
        (dy0, dy1 respectively). The remaining 2 equations have higher/lower dz (dz1, dz2 respectively), so similarly 
        subtract the higher delta z from the lower delta z to cancel all terms.

        For unit square normalisation, dy0 = 1, dz0 = 1 and dx0 = 1, the denominator simplifies, and dx1 and dz1 
        are rescaled to be normalised to distance dx0, dz0.
        """

        cdef double dx1, dz1

        dx1 = (self._x[lower_index_x + 1] - self._x[lower_index_x]) / (self._x[lower_index_x + 2] - self._x[lower_index_x + 1])
        dz1 = (self._z[lower_index_z + 1] - self._z[lower_index_z]) / (self._z[lower_index_z + 2] - self._z[lower_index_z + 1])
        return (self._f[lower_index_x + 2, lower_index_y + 1, lower_index_z + 2]
                - self._f[lower_index_x, lower_index_y + 1, lower_index_z + 2]
                - self._f[lower_index_x + 2, lower_index_y, lower_index_z + 2]
                + self._f[lower_index_x, lower_index_y, lower_index_z + 2]
                - self._f[lower_index_x + 2, lower_index_y + 1, lower_index_z]
                + self._f[lower_index_x, lower_index_y + 1, lower_index_z]
                + self._f[lower_index_x + 2, lower_index_y, lower_index_z]
                - self._f[lower_index_x, lower_index_y, lower_index_z]) / (1. + dx1 + dz1 + dx1 * dz1)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_d3fdxdydz_edge_z(self, int lower_index_x, int lower_index_y, int lower_index_z) except? -1e999:
        """
        Calculate d3f/dxdydz on an unevenly spaced grid as a 3rd order approximation. This is used when on the border of
        the spline knot array in z.

        A third order taylor expansions at 8 different points around f(x, y, z) for every combination of +dx0, +dy0, +dz0 or
        -dx1, -dy1, -dz1=0 (where dz1 = 0 because we are looking at the edge in z), at the same location as the spline knot we are looking 
        at), are written out. The Taylor expansions are subtracted from each other that have the same 
        delta y and delta z positions, but higher/lower dx (dx0, dx1 respectively). The resulting equations are 
        subtracted from each other where they have the same delta z positions, but higher/lower dy 
        (dy0, dy1 respectively). The remaining 2 equations have higher/lower dz (dz1, dz2 respectively), so similarly 
        subtract the higher delta z from the lower delta z to cancel all terms.

        For unit square normalisation, dy0 = 1, dz0 = 1 and dx0 = 1, the denominator simplifies, and dx1 and dy1 
        are rescaled to be normalised to distance dx0, dy0.
        """

        cdef double dx1, dy1

        dx1 = (self._x[lower_index_x + 1] - self._x[lower_index_x]) / (self._x[lower_index_x + 2] - self._x[lower_index_x + 1])
        dy1 = (self._y[lower_index_y + 1] - self._y[lower_index_y]) / (self._y[lower_index_y + 2] - self._y[lower_index_y + 1])
        return (self._f[lower_index_x + 2, lower_index_y + 2, lower_index_z + 1]
                - self._f[lower_index_x, lower_index_y + 2, lower_index_z + 1]
                - self._f[lower_index_x + 2, lower_index_y, lower_index_z + 1]
                + self._f[lower_index_x, lower_index_y, lower_index_z + 1]
                - self._f[lower_index_x + 2, lower_index_y + 2, lower_index_z]
                + self._f[lower_index_x, lower_index_y + 2, lower_index_z]
                + self._f[lower_index_x + 2, lower_index_y, lower_index_z]
                - self._f[lower_index_x, lower_index_y, lower_index_z]) / (1. + dx1 + dy1 + dx1 * dy1)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_d3fdxdydz_edge_xy(self, int lower_index_x, int lower_index_y, int lower_index_z) except? -1e999:
        """
        Calculate d3f/dxdydz on an unevenly spaced grid as a 3rd order approximation. This is used when on the border of
        the spline knot array in x and y.

        A third order taylor expansions at 8 different points around f(x, y, z) for every combination of +dx0, +dy0, +dz0 or
        -dx1=0, -dy1=0, -dz1 (where dx1 = 0 and dy1 = 0 because we are looking at the edge in x and y), at the same location as the spline knot we are looking 
        at), are written out. The Taylor expansions are subtracted from each other that have the same 
        delta y and delta z positions, but higher/lower dx (dx0, dx1 respectively). The resulting equations are 
        subtracted from each other where they have the same delta z positions, but higher/lower dy 
        (dy0, dy1 respectively). The remaining 2 equations have higher/lower dz (dz1, dz2 respectively), so similarly 
        subtract the higher delta z from the lower delta z to cancel all terms.

        For unit square normalisation, dy0 = 1, dz0 = 1 and dx0 = 1, the denominator simplifies, and dz1 
        is rescaled to be normalised to distance dz0.
        """

        cdef double dz1

        dz1 = (self._z[lower_index_z + 1] - self._z[lower_index_z]) / (self._z[lower_index_z + 2] - self._z[lower_index_z + 1])
        return (self._f[lower_index_x + 1, lower_index_y + 1, lower_index_z + 2]
                - self._f[lower_index_x, lower_index_y + 1, lower_index_z + 2]
                - self._f[lower_index_x + 1, lower_index_y, lower_index_z + 2]
                + self._f[lower_index_x, lower_index_y, lower_index_z + 2]
                - self._f[lower_index_x + 1, lower_index_y + 1, lower_index_z]
                + self._f[lower_index_x, lower_index_y + 1, lower_index_z]
                + self._f[lower_index_x + 1, lower_index_y, lower_index_z]
                - self._f[lower_index_x, lower_index_y, lower_index_z]) / (1. + dz1)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_d3fdxdydz_edge_xz(self, int lower_index_x, int lower_index_y, int lower_index_z) except? -1e999:
        """
        Calculate d3f/dxdydz on an unevenly spaced grid as a 3rd order approximation. This is used when on the border of
        the spline knot array in x and z.

        A third order taylor expansions at 8 different points around f(x, y, z) for every combination of +dx0, +dy0, +dz0 or
        -dx1=0, -dy1, -dz1=0 (where dx1 = 0 and dz1 = 0 because we are looking at the edge in x and z), at the same location as the spline knot we are looking 
        at), are written out. The Taylor expansions are subtracted from each other that have the same 
        delta y and delta z positions, but higher/lower dx (dx0, dx1 respectively). The resulting equations are 
        subtracted from each other where they have the same delta z positions, but higher/lower dy 
        (dy0, dy1 respectively). The remaining 2 equations have higher/lower dz (dz1, dz2 respectively), so similarly 
        subtract the higher delta z from the lower delta z to cancel all terms.

        For unit square normalisation, dy0 = 1, dz0 = 1 and dx0 = 1, the denominator simplifies, and dy1
        is rescaled to be normalised to distance dy0.
        """

        cdef double dy1

        dy1 = (self._y[lower_index_y + 1] - self._y[lower_index_y]) / (self._y[lower_index_y + 2] - self._y[lower_index_y + 1])
        return (self._f[lower_index_x + 1, lower_index_y + 2, lower_index_z + 1]
                - self._f[lower_index_x, lower_index_y + 2, lower_index_z + 1]
                - self._f[lower_index_x + 1, lower_index_y, lower_index_z + 1]
                + self._f[lower_index_x, lower_index_y, lower_index_z + 1]
                - self._f[lower_index_x + 1, lower_index_y + 2, lower_index_z]
                + self._f[lower_index_x, lower_index_y + 2, lower_index_z]
                + self._f[lower_index_x + 1, lower_index_y, lower_index_z]
                - self._f[lower_index_x, lower_index_y, lower_index_z]) / (1. + dy1)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_d3fdxdydz_edge_yz(self, int lower_index_x, int lower_index_y, int lower_index_z) except? -1e999:
        """
        Calculate d3f/dxdydz on an unevenly spaced grid as a 3rd order approximation. This is used when on the border of
        the spline knot array in y and z.

        A third order taylor expansions at 8 different points around f(x, y, z) for every combination of +dx0, +dy0, +dz0 or
        -dx1, -dy1=0, -dz1=0 (where dy1 = 0 and dz1 = 0 because we are looking at the edge in y and z), at the same location as the spline knot we are looking 
        at), are written out. The Taylor expansions are subtracted from each other that have the same 
        delta y and delta z positions, but higher/lower dx (dx0, dx1 respectively). The resulting equations are 
        subtracted from each other where they have the same delta z positions, but higher/lower dy 
        (dy0, dy1 respectively). The remaining 2 equations have higher/lower dz (dz1, dz2 respectively), so similarly 
        subtract the higher delta z from the lower delta z to cancel all terms.

        For unit square normalisation, dy0 = 1, dz0 = 1 and dx0 = 1, the denominator simplifies, and dx1
        is rescaled to be normalised to distance dx0.
        """

        cdef double dx1

        dx1 = (self._x[lower_index_x + 1] - self._x[lower_index_x]) / (self._x[lower_index_x + 2] - self._x[lower_index_x + 1])
        return (self._f[lower_index_x + 2, lower_index_y + 1, lower_index_z + 1]
                - self._f[lower_index_x, lower_index_y + 1, lower_index_z + 1]
                - self._f[lower_index_x + 2, lower_index_y, lower_index_z + 1]
                + self._f[lower_index_x, lower_index_y, lower_index_z + 1]
                - self._f[lower_index_x + 2, lower_index_y + 1, lower_index_z]
                + self._f[lower_index_x, lower_index_y + 1, lower_index_z]
                + self._f[lower_index_x + 2, lower_index_y, lower_index_z]
                - self._f[lower_index_x, lower_index_y, lower_index_z]) / (1. + dx1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _derivative_d3fdxdydz_edge_xyz(self, int lower_index_x, int lower_index_y, int lower_index_z) except? -1e999:
        """
        Calculate d3f/dxdydz on an unevenly spaced grid as a 3rd order approximation. This is used when on the border of
        the spline knot array in x, y and z.

        A third order taylor expansions at 8 different points around f(x, y, z) for every combination of +dx0, +dy0, 
        +dz0 or -dx1=0, -dy1=0, -dz1=0 (where dy1 = 0 and dz1 = 0 because we are looking at the edge in x, y and z), 
        at the same location as the spline knot we are looking at), are written out. The Taylor expansions are 
        subtracted from each other that have the same delta y and delta z positions, but higher/lower dx (dx0, dx1 
        respectively). The resulting equations are subtracted from each other where they have the same delta z 
        positions, but higher/lower dy (dy0, dy1 respectively). The remaining 2 equations have higher/lower dz (dz1, 
        dz2 respectively), so similarly subtract the higher delta z from the lower delta z to cancel all terms.

        For unit square normalisation, dy0 = 1, dz0 = 1 and dx0 = 1, the denominator simplifies.
        """

        return self._f[lower_index_x + 1, lower_index_y + 1, lower_index_z + 1] \
               - self._f[lower_index_x, lower_index_y + 1, lower_index_z + 1] \
               - self._f[lower_index_x + 1, lower_index_y, lower_index_z + 1] \
               + self._f[lower_index_x, lower_index_y, lower_index_z + 1] \
               - self._f[lower_index_x + 1, lower_index_y + 1, lower_index_z] \
               + self._f[lower_index_x, lower_index_y + 1, lower_index_z] \
               + self._f[lower_index_x + 1, lower_index_y, lower_index_z] \
               - self._f[lower_index_x, lower_index_y, lower_index_z]


id_to_interpolator = {
    _Interpolator3DLinear.ID: _Interpolator3DLinear,
    _Interpolator3DCubic.ID: _Interpolator3DCubic
}


id_to_extrapolator = {
    _Extrapolator3DNone.ID: _Extrapolator3DNone,
    _Extrapolator3DNearest.ID: _Extrapolator3DNearest,
    _Extrapolator3DLinear.ID: _Extrapolator3DLinear,
}


permitted_interpolation_combinations = {
    _Interpolator3DLinear.ID: [_Extrapolator3DNone.ID, _Extrapolator3DNearest.ID, _Extrapolator3DLinear.ID],
    _Interpolator3DCubic.ID: [_Extrapolator3DNone.ID, _Extrapolator3DNearest.ID, _Extrapolator3DLinear.ID]
}
