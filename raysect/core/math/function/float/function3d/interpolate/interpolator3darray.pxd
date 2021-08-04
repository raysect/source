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

from raysect.core.math.function.float.function3d cimport Function3D
from numpy cimport ndarray


cdef double rescale_lower_normalisation(double dfdn, double x_lower, double x, double x_upper)


cdef int find_index_change(int index, int last_index)


cdef int find_edge_index(int index, int last_index)


cdef class Interpolator3DArray(Function3D):

    cdef:
        ndarray x, y, z, f
        double[::1] _x_mv, _y_mv, _z_mv
        double [:, :, ::1] _f_mv
        _Interpolator3D _interpolator
        _Extrapolator3D _extrapolator
        int _last_index_x, _last_index_y, _last_index_z
        double _extrapolation_range_x, _extrapolation_range_y, _extrapolation_range_z

    cdef double evaluate(self, double px, double py, double pz) except? -1e999


cdef class _Interpolator3D:
    cdef:
        double [::1] _x, _y, _z
        double [:, :, ::1] _f
        int _last_index_x, _last_index_y, _last_index_z

    cdef double evaluate(self, double px, double py, double pz, int index_x, int index_y, int index_z) except? -1e999

    cdef double analytic_gradient(self, double px, double py, double pz, int index_x, int index_y, int index_z, int order_x, int order_y, int order_z)


cdef class _Interpolator3DLinear(_Interpolator3D):
    cdef double _calculate_coefficients(self, int index_x, int index_y, int index_z, int coefficient_index)


cdef class _Interpolator3DCubic(_Interpolator3D):
    cdef:
        ndarray _a, _mask_a
        double[:, :, :, :, :, ::1] _a_mv

    cdef _cache_coefficients(self, int index_x, int index_y, int index_z, double[4][4][4] a)


cdef class _Extrapolator3D:
    cdef:
        double [::1] _x, _y, _z
        double [:, :, ::1] _f
        _Interpolator3D _interpolator
        int _last_index_x, _last_index_y, _last_index_z
        double _extrapolation_range_x, _extrapolation_range_y, _extrapolation_range_z

    cdef double evaluate(self, double px, double py, double pz, int index_x, int index_y, int index_z) except? -1e999

    cdef double _evaluate_edge_x(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999

    cdef double _evaluate_edge_y(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999

    cdef double _evaluate_edge_z(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999

    cdef double _evaluate_edge_xy(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999

    cdef double _evaluate_edge_xz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999

    cdef double _evaluate_edge_yz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999

    cdef double _evaluate_edge_xyz(self, double px, double py, double pz, int index_x, int index_y, int index_z, int edge_x_index, int edge_y_index, int edge_z_index) except? -1e999


cdef class _Extrapolator3DNone(_Extrapolator3D):
    pass


cdef class _Extrapolator3DLinear(_Extrapolator3D):
    pass


cdef class _ArrayDerivative3D:
    cdef:
        double [::1] _x, _y, _z
        double [:, :, ::1] _f
        int _last_index_x, _last_index_y, _last_index_z

    cdef double evaluate(self, int index_x, int index_y, int index_z, int derivative_order_x, int derivative_order_y, int derivative_order_z, bint rescale_norm_x, bint rescale_norm_y, bint rescale_norm_z) except? -1e999

    cdef double _derivitive_dfdx(self, double[:] x, double[:] f) except? -1e999

    cdef double _derivitive_dfdx_edge(self, double[:] f)

    cdef double _derivitive_d2fdxdy(self,  double[:] x, double[:] y, double[:, :] f) except? -1e999

    cdef double _derivitive_d2fdxdy_edge_xy(self, double[:] x, double[:] y, double[:, :] f) except? -1e999

    cdef double _derivitive_d2fdxdy_edge_x(self, double[:] y, double[:, :] f) except? -1e999

    cdef double _derivitive_d2fdxdy_edge_y(self, double[:] x, double[:, :] f) except? -1e999

    cdef double _derivitive_d3fdxdydz(self, double[:] x, double[:] y, double[:] z, double[:, :, :] f) except? -1e999

    cdef double _derivitive_d3fdxdydz_edge_x(self, double[:] y, double[:] z, double[:, :, :] f) except? -1e999

    cdef double _derivitive_d3fdxdydz_edge_y(self, double[:] x, double[:] z, double[:, :, :] f) except? -1e999

    cdef double _derivitive_d3fdxdydz_edge_z(self, double[:] x, double[:] y, double[:, :, :] f) except? -1e999

    cdef double _derivitive_d3fdxdydz_edge_xy(self, double[:] z, double[:, :, :] f) except? -1e999

    cdef double _derivitive_d3fdxdydz_edge_xz(self, double[:] y, double[:, :, :] f) except? -1e999

    cdef double _derivitive_d3fdxdydz_edge_yz(self, double[:] x, double[:, :, :] f) except? -1e999

    cdef double _derivitive_d3fdxdydz_edge_xyz(self, double[:, :, :] f) except? -1e999

    cdef double _eval_edge_x(self, int index_x, int index_y, int index_z, int derivative_order_x, int derivative_order_y, int derivative_order_z, int x_centre_add, int y_centre_add, int z_centre_add)

    cdef double _eval_edge_y(self, int index_x, int index_y, int index_z, int derivative_order_x, int derivative_order_y, int derivative_order_z, int x_centre_add, int y_centre_add, int z_centre_add)

    cdef double _eval_edge_z(self, int index_x, int index_y, int index_z, int derivative_order_x, int derivative_order_y, int derivative_order_z, int x_centre_add, int y_centre_add, int z_centre_add) except? -1e999

    cdef double _eval_edge_xy(self, int index_x, int index_y, int index_z, int derivative_order_x, int derivative_order_y, int derivative_order_z, int x_centre_add, int y_centre_add, int z_centre_add) except? -1e999

    cdef double _eval_edge_xz(self, int index_x, int index_y, int index_z, int derivative_order_x, int derivative_order_y, int derivative_order_z, int x_centre_add, int y_centre_add, int z_centre_add) except? -1e999

    cdef double _eval_edge_yz(self, int index_x, int index_y, int index_z, int derivative_order_x, int derivative_order_y, int derivative_order_z, int x_centre_add, int y_centre_add, int z_centre_add) except? -1e999

    cdef double _eval_edge_xyz(self, int index_x, int index_y, int index_z, int derivative_order_x, int derivative_order_y, int derivative_order_z, int x_centre_add, int y_centre_add, int z_centre_add) except? -1e999

    cdef double _eval_xyz(self, int index_x, int index_y, int index_z, int derivative_order_x, int derivative_order_y, int derivative_order_z)
