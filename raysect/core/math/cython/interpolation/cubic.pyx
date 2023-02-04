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

cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void calc_coefficients_1d(double f[2], double dfdx[2], double a[4]) nogil:
    """
    Calculates the cubic coefficients for a unit interval.

    This function calculates the polynomial coefficients for a 1D cubic. It
    requires the values and differentials at each vertex. The domain over 
    which the polynomial is valid is [0, 1].

    :param f: 
    :param dfdx: 
    :param a: 
    :return: 
    """
    a[0] = 2 * f[0] - 2 * f[1] + dfdx[0] + dfdx[1]
    a[1] = -3 * f[0] + 3 * f[1] - 2. * dfdx[0] - dfdx[1]
    a[2] = dfdx[0]
    a[3] = f[0]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void calc_coefficients_2d(double f[2][2], double dfdx[2][2], double dfdy[2][2], double d2fdxdy[2][2], double a[4][4]) nogil:
    """
    Calculates the cubic coefficients for a unit square.
    
    This function calculates the polynomial coefficients for a 2D cubic. It
    requires the values and differentials at each vertex of the square. The
    domain over which the polynomial is valid is [0, 1] in each dimension.
    
    :param f: 
    :param dfdx: 
    :param dfdy: 
    :param d2fdxdy: 
    :param a: 
    :return: 
    """

    a[0][0] =   f[0][0]
    a[0][1] =   dfdy[0][0]
    a[0][2] = - 3*f[0][0] + 3*f[0][1] - 2*dfdy[0][0] - dfdy[0][1]
    a[0][3] =   2*f[0][0] - 2*f[0][1] + dfdy[0][0] + dfdy[0][1]
    a[1][0] =   dfdx[0][0]
    a[1][1] =   d2fdxdy[0][0]
    a[1][2] = - 3*dfdx[0][0] + 3*dfdx[0][1] - 2*d2fdxdy[0][0] - d2fdxdy[0][1]
    a[1][3] =   2*dfdx[0][0] - 2*dfdx[0][1] + d2fdxdy[0][0] + d2fdxdy[0][1]
    a[2][0] = - 3*f[0][0] + 3*f[1][0] - 2*dfdx[0][0] - dfdx[1][0]
    a[2][1] = - 3*dfdy[0][0] + 3*dfdy[1][0] - 2*d2fdxdy[0][0] - d2fdxdy[1][0]
    a[2][2] =   9*f[0][0] - 9*f[0][1] - 9*f[1][0] + 9*f[1][1] \
              + 6*dfdx[0][0] - 6*dfdx[0][1] + 3*dfdx[1][0] - 3*dfdx[1][1] \
              + 6*dfdy[0][0] + 3*dfdy[0][1] - 6*dfdy[1][0] - 3*dfdy[1][1] \
              + 4*d2fdxdy[0][0] + 2*d2fdxdy[0][1] + 2*d2fdxdy[1][0] + d2fdxdy[1][1]
    a[2][3] = - 6*f[0][0] + 6*f[0][1] + 6*f[1][0] - 6*f[1][1] \
              - 4*dfdx[0][0] + 4*dfdx[0][1] - 2*dfdx[1][0] + 2*dfdx[1][1] \
              - 3*dfdy[0][0] - 3*dfdy[0][1] + 3*dfdy[1][0] + 3*dfdy[1][1] \
              - 2*d2fdxdy[0][0] - 2*d2fdxdy[0][1] - d2fdxdy[1][0] - d2fdxdy[1][1]
    a[3][0] =   2*f[0][0] - 2*f[1][0] + dfdx[0][0] + dfdx[1][0]
    a[3][1] =   2*dfdy[0][0] - 2*dfdy[1][0] + d2fdxdy[0][0] + d2fdxdy[1][0]
    a[3][2] = - 6*f[0][0] + 6*f[0][1] + 6*f[1][0] - 6*f[1][1] \
              - 3*dfdx[0][0] + 3*dfdx[0][1] - 3*dfdx[1][0] + 3*dfdx[1][1] \
              - 4*dfdy[0][0] - 2*dfdy[0][1] + 4*dfdy[1][0] + 2*dfdy[1][1] \
              - 2*d2fdxdy[0][0] - d2fdxdy[0][1] - 2*d2fdxdy[1][0] - d2fdxdy[1][1]
    a[3][3] =   4*f[0][0] - 4*f[0][1] - 4*f[1][0] + 4*f[1][1] \
              + 2*dfdx[0][0] - 2*dfdx[0][1] + 2*dfdx[1][0] - 2*dfdx[1][1] \
              + 2*dfdy[0][0] + 2*dfdy[0][1] - 2*dfdy[1][0] - 2*dfdy[1][1] \
              + d2fdxdy[0][0] + d2fdxdy[0][1] + d2fdxdy[1][0] + d2fdxdy[1][1]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void calc_coefficients_3d(double f[2][2][2], double dfdx[2][2][2], double dfdy[2][2][2], double dfdz[2][2][2],
                              double d2fdxdy[2][2][2], double d2fdxdz[2][2][2], double d2fdydz[2][2][2],
                              double d3fdxdydz[2][2][2], double a[4][4][4]) nogil:
    """
    Calculates the cubic coefficients for a unit cubic.
    
    This function calculates the polynomial coefficients for a 3D cubic. It
    requires the values and differentials at each vertex of the cubic. The
    domain over which the polynomial is valid is [0, 1] in each dimension.
    
    :param f: 
    :param dfdx: 
    :param dfdy: 
    :param dfdz: 
    :param d2fdxdy: 
    :param d2fdxdz: 
    :param d2fdydz: 
    :param d3fdxdydz: 
    :param a: 
    :return:
    """

    a[0][0][0] =   f[0][0][0]
    a[0][0][1] =   dfdz[0][0][0]
    a[0][0][2] = - 3*f[0][0][0] + 3*f[0][0][1] - 2*dfdz[0][0][0] - dfdz[0][0][1]
    a[0][0][3] =   2*f[0][0][0] - 2*f[0][0][1] + dfdz[0][0][0] + dfdz[0][0][1]
    a[0][1][0] =   dfdy[0][0][0]
    a[0][1][1] =   d2fdydz[0][0][0]
    a[0][1][2] = - 3*dfdy[0][0][0] + 3*dfdy[0][0][1] - 2*d2fdydz[0][0][0] - d2fdydz[0][0][1]
    a[0][1][3] =   2*dfdy[0][0][0] - 2*dfdy[0][0][1] + d2fdydz[0][0][0] + d2fdydz[0][0][1]
    a[0][2][0] = - 3*f[0][0][0] + 3*f[0][1][0] - 2*dfdy[0][0][0] - dfdy[0][1][0]
    a[0][2][1] = - 3*dfdz[0][0][0] + 3*dfdz[0][1][0] - 2*d2fdydz[0][0][0] - d2fdydz[0][1][0]
    a[0][2][2] =   9*f[0][0][0] - 9*f[0][0][1] - 9*f[0][1][0] + 9*f[0][1][1] \
                 + 6*dfdy[0][0][0] - 6*dfdy[0][0][1] + 3*dfdy[0][1][0] - 3*dfdy[0][1][1] \
                 + 6*dfdz[0][0][0] + 3*dfdz[0][0][1] - 6*dfdz[0][1][0] - 3*dfdz[0][1][1] \
                 + 4*d2fdydz[0][0][0] + 2*d2fdydz[0][0][1] + 2*d2fdydz[0][1][0] + d2fdydz[0][1][1]
    a[0][2][3] = - 6*f[0][0][0] + 6*f[0][0][1] + 6*f[0][1][0] - 6*f[0][1][1] \
                 - 4*dfdy[0][0][0] + 4*dfdy[0][0][1] - 2*dfdy[0][1][0] + 2*dfdy[0][1][1] \
                 - 3*dfdz[0][0][0] - 3*dfdz[0][0][1] + 3*dfdz[0][1][0] + 3*dfdz[0][1][1] \
                 - 2*d2fdydz[0][0][0] - 2*d2fdydz[0][0][1] - d2fdydz[0][1][0] - d2fdydz[0][1][1]
    a[0][3][0] =   2*f[0][0][0] - 2*f[0][1][0] + dfdy[0][0][0] + dfdy[0][1][0]
    a[0][3][1] =   2*dfdz[0][0][0] - 2*dfdz[0][1][0] + d2fdydz[0][0][0] + d2fdydz[0][1][0]
    a[0][3][2] = - 6*f[0][0][0] + 6*f[0][0][1] + 6*f[0][1][0] - 6*f[0][1][1] \
                 - 3*dfdy[0][0][0] + 3*dfdy[0][0][1] - 3*dfdy[0][1][0] + 3*dfdy[0][1][1] \
                 - 4*dfdz[0][0][0] - 2*dfdz[0][0][1] + 4*dfdz[0][1][0] + 2*dfdz[0][1][1] \
                 - 2*d2fdydz[0][0][0] - d2fdydz[0][0][1] - 2*d2fdydz[0][1][0] - d2fdydz[0][1][1]
    a[0][3][3] =   4*f[0][0][0] - 4*f[0][0][1] - 4*f[0][1][0] + 4*f[0][1][1] \
                 + 2*dfdy[0][0][0] - 2*dfdy[0][0][1] + 2*dfdy[0][1][0] - 2*dfdy[0][1][1] \
                 + 2*dfdz[0][0][0] + 2*dfdz[0][0][1] - 2*dfdz[0][1][0] - 2*dfdz[0][1][1] \
                 + d2fdydz[0][0][0] + d2fdydz[0][0][1] + d2fdydz[0][1][0] + d2fdydz[0][1][1]
    a[1][0][0] =   dfdx[0][0][0]
    a[1][0][1] =   d2fdxdz[0][0][0]
    a[1][0][2] = - 3*dfdx[0][0][0] + 3*dfdx[0][0][1] - 2*d2fdxdz[0][0][0] - d2fdxdz[0][0][1]
    a[1][0][3] =   2*dfdx[0][0][0] - 2*dfdx[0][0][1] + d2fdxdz[0][0][0] + d2fdxdz[0][0][1]
    a[1][1][0] =   d2fdxdy[0][0][0]
    a[1][1][1] =   d3fdxdydz[0][0][0]
    a[1][1][2] = - 3*d2fdxdy[0][0][0] + 3*d2fdxdy[0][0][1] - 2*d3fdxdydz[0][0][0] - d3fdxdydz[0][0][1]
    a[1][1][3] =   2*d2fdxdy[0][0][0] - 2*d2fdxdy[0][0][1] + d3fdxdydz[0][0][0] + d3fdxdydz[0][0][1]
    a[1][2][0] = - 3*dfdx[0][0][0] + 3*dfdx[0][1][0] - 2*d2fdxdy[0][0][0] - d2fdxdy[0][1][0]
    a[1][2][1] = - 3*d2fdxdz[0][0][0] + 3*d2fdxdz[0][1][0] - 2*d3fdxdydz[0][0][0] - d3fdxdydz[0][1][0]
    a[1][2][2] =   9*dfdx[0][0][0] - 9*dfdx[0][0][1] - 9*dfdx[0][1][0] + 9*dfdx[0][1][1] \
                 + 6*d2fdxdy[0][0][0] - 6*d2fdxdy[0][0][1] + 3*d2fdxdy[0][1][0] - 3*d2fdxdy[0][1][1] \
                 + 6*d2fdxdz[0][0][0] + 3*d2fdxdz[0][0][1] - 6*d2fdxdz[0][1][0] - 3*d2fdxdz[0][1][1] \
                 + 4*d3fdxdydz[0][0][0] + 2*d3fdxdydz[0][0][1] + 2*d3fdxdydz[0][1][0] + d3fdxdydz[0][1][1]
    a[1][2][3] = - 6*dfdx[0][0][0] + 6*dfdx[0][0][1] + 6*dfdx[0][1][0] - 6*dfdx[0][1][1] \
                 - 4*d2fdxdy[0][0][0] + 4*d2fdxdy[0][0][1] - 2*d2fdxdy[0][1][0] + 2*d2fdxdy[0][1][1] \
                 - 3*d2fdxdz[0][0][0] - 3*d2fdxdz[0][0][1] + 3*d2fdxdz[0][1][0] + 3*d2fdxdz[0][1][1] \
                 - 2*d3fdxdydz[0][0][0] - 2*d3fdxdydz[0][0][1] - d3fdxdydz[0][1][0] - d3fdxdydz[0][1][1]
    a[1][3][0] =   2*dfdx[0][0][0] - 2*dfdx[0][1][0] + d2fdxdy[0][0][0] + d2fdxdy[0][1][0]
    a[1][3][1] =   2*d2fdxdz[0][0][0] - 2*d2fdxdz[0][1][0] + d3fdxdydz[0][0][0] + d3fdxdydz[0][1][0]
    a[1][3][2] = - 6*dfdx[0][0][0] + 6*dfdx[0][0][1] + 6*dfdx[0][1][0] - 6*dfdx[0][1][1] \
                 - 3*d2fdxdy[0][0][0] + 3*d2fdxdy[0][0][1] - 3*d2fdxdy[0][1][0] + 3*d2fdxdy[0][1][1] \
                 - 4*d2fdxdz[0][0][0] - 2*d2fdxdz[0][0][1] + 4*d2fdxdz[0][1][0] + 2*d2fdxdz[0][1][1] \
                 - 2*d3fdxdydz[0][0][0] - d3fdxdydz[0][0][1] - 2*d3fdxdydz[0][1][0] - d3fdxdydz[0][1][1]
    a[1][3][3] =   4*dfdx[0][0][0] - 4*dfdx[0][0][1] - 4*dfdx[0][1][0] + 4*dfdx[0][1][1] \
                 + 2*d2fdxdy[0][0][0] - 2*d2fdxdy[0][0][1] + 2*d2fdxdy[0][1][0] - 2*d2fdxdy[0][1][1] \
                 + 2*d2fdxdz[0][0][0] + 2*d2fdxdz[0][0][1] - 2*d2fdxdz[0][1][0] - 2*d2fdxdz[0][1][1] \
                 + d3fdxdydz[0][0][0] + d3fdxdydz[0][0][1] + d3fdxdydz[0][1][0] + d3fdxdydz[0][1][1]
    a[2][0][0] = - 3*f[0][0][0] + 3*f[1][0][0] - 2*dfdx[0][0][0] - dfdx[1][0][0]
    a[2][0][1] = - 3*dfdz[0][0][0] + 3*dfdz[1][0][0] - 2*d2fdxdz[0][0][0] - d2fdxdz[1][0][0]
    a[2][0][2] =   9*f[0][0][0] - 9*f[0][0][1] - 9*f[1][0][0] + 9*f[1][0][1] \
                 + 6*dfdx[0][0][0] - 6*dfdx[0][0][1] + 3*dfdx[1][0][0] - 3*dfdx[1][0][1] \
                 + 6*dfdz[0][0][0] + 3*dfdz[0][0][1] - 6*dfdz[1][0][0] - 3*dfdz[1][0][1] \
                 + 4*d2fdxdz[0][0][0] + 2*d2fdxdz[0][0][1] + 2*d2fdxdz[1][0][0] + d2fdxdz[1][0][1]
    a[2][0][3] = - 6*f[0][0][0] + 6*f[0][0][1] + 6*f[1][0][0] - 6*f[1][0][1] \
                 - 4*dfdx[0][0][0] + 4*dfdx[0][0][1] - 2*dfdx[1][0][0] + 2*dfdx[1][0][1] \
                 - 3*dfdz[0][0][0] - 3*dfdz[0][0][1] + 3*dfdz[1][0][0] + 3*dfdz[1][0][1] \
                 - 2*d2fdxdz[0][0][0] - 2*d2fdxdz[0][0][1] - d2fdxdz[1][0][0] - d2fdxdz[1][0][1]
    a[2][1][0] = - 3*dfdy[0][0][0] + 3*dfdy[1][0][0] - 2*d2fdxdy[0][0][0] - d2fdxdy[1][0][0]
    a[2][1][1] = - 3*d2fdydz[0][0][0] + 3*d2fdydz[1][0][0] - 2*d3fdxdydz[0][0][0] - d3fdxdydz[1][0][0]
    a[2][1][2] =   9*dfdy[0][0][0] - 9*dfdy[0][0][1] - 9*dfdy[1][0][0] + 9*dfdy[1][0][1] \
                 + 6*d2fdxdy[0][0][0] - 6*d2fdxdy[0][0][1] + 3*d2fdxdy[1][0][0] - 3*d2fdxdy[1][0][1] \
                 + 6*d2fdydz[0][0][0] + 3*d2fdydz[0][0][1] - 6*d2fdydz[1][0][0] - 3*d2fdydz[1][0][1] \
                 + 4*d3fdxdydz[0][0][0] + 2*d3fdxdydz[0][0][1] + 2*d3fdxdydz[1][0][0] + d3fdxdydz[1][0][1]
    a[2][1][3] = - 6*dfdy[0][0][0] + 6*dfdy[0][0][1] + 6*dfdy[1][0][0] - 6*dfdy[1][0][1] \
                 - 4*d2fdxdy[0][0][0] + 4*d2fdxdy[0][0][1] - 2*d2fdxdy[1][0][0] + 2*d2fdxdy[1][0][1] \
                 - 3*d2fdydz[0][0][0] - 3*d2fdydz[0][0][1] + 3*d2fdydz[1][0][0] + 3*d2fdydz[1][0][1] \
                 - 2*d3fdxdydz[0][0][0] - 2*d3fdxdydz[0][0][1] - d3fdxdydz[1][0][0] - d3fdxdydz[1][0][1]
    a[2][2][0] =   9*f[0][0][0] - 9*f[0][1][0] - 9*f[1][0][0] + 9*f[1][1][0] \
                 + 6*dfdx[0][0][0] - 6*dfdx[0][1][0] + 3*dfdx[1][0][0] - 3*dfdx[1][1][0] \
                 + 6*dfdy[0][0][0] + 3*dfdy[0][1][0] - 6*dfdy[1][0][0] - 3*dfdy[1][1][0] \
                 + 4*d2fdxdy[0][0][0] + 2*d2fdxdy[0][1][0] + 2*d2fdxdy[1][0][0] + d2fdxdy[1][1][0]
    a[2][2][1] =   9*dfdz[0][0][0] - 9*dfdz[0][1][0] - 9*dfdz[1][0][0] + 9*dfdz[1][1][0] \
                 + 6*d2fdxdz[0][0][0] - 6*d2fdxdz[0][1][0] + 3*d2fdxdz[1][0][0] - 3*d2fdxdz[1][1][0] \
                 + 6*d2fdydz[0][0][0] + 3*d2fdydz[0][1][0] - 6*d2fdydz[1][0][0] - 3*d2fdydz[1][1][0] \
                 + 4*d3fdxdydz[0][0][0] + 2*d3fdxdydz[0][1][0] + 2*d3fdxdydz[1][0][0] + d3fdxdydz[1][1][0]
    a[2][2][2] = - 27*f[0][0][0] + 27*f[0][0][1] + 27*f[0][1][0] - 27*f[0][1][1] \
                 + 27*f[1][0][0] - 27*f[1][0][1] - 27*f[1][1][0] + 27*f[1][1][1] \
                 - 18*dfdx[0][0][0] + 18*dfdx[0][0][1] + 18*dfdx[0][1][0] - 18*dfdx[0][1][1] \
                 - 9*dfdx[1][0][0] + 9*dfdx[1][0][1] + 9*dfdx[1][1][0] - 9*dfdx[1][1][1] \
                 - 18*dfdy[0][0][0] + 18*dfdy[0][0][1] - 9*dfdy[0][1][0] + 9*dfdy[0][1][1] \
                 + 18*dfdy[1][0][0] - 18*dfdy[1][0][1] + 9*dfdy[1][1][0] - 9*dfdy[1][1][1] \
                 - 18*dfdz[0][0][0] - 9*dfdz[0][0][1] + 18*dfdz[0][1][0] + 9*dfdz[0][1][1] \
                 + 18*dfdz[1][0][0] + 9*dfdz[1][0][1] - 18*dfdz[1][1][0] - 9*dfdz[1][1][1] \
                 - 12*d2fdxdy[0][0][0] + 12*d2fdxdy[0][0][1] - 6*d2fdxdy[0][1][0] + 6*d2fdxdy[0][1][1] \
                 - 6*d2fdxdy[1][0][0] + 6*d2fdxdy[1][0][1] - 3*d2fdxdy[1][1][0] + 3*d2fdxdy[1][1][1] \
                 - 12*d2fdxdz[0][0][0] - 6*d2fdxdz[0][0][1] + 12*d2fdxdz[0][1][0] + 6*d2fdxdz[0][1][1] \
                 - 6*d2fdxdz[1][0][0] - 3*d2fdxdz[1][0][1] + 6*d2fdxdz[1][1][0] + 3*d2fdxdz[1][1][1] \
                 - 12*d2fdydz[0][0][0] - 6*d2fdydz[0][0][1] - 6*d2fdydz[0][1][0] - 3*d2fdydz[0][1][1] \
                 + 12*d2fdydz[1][0][0] + 6*d2fdydz[1][0][1] + 6*d2fdydz[1][1][0] + 3*d2fdydz[1][1][1] \
                 - 8*d3fdxdydz[0][0][0] - 4*d3fdxdydz[0][0][1] - 4*d3fdxdydz[0][1][0] - 2*d3fdxdydz[0][1][1] \
                 - 4*d3fdxdydz[1][0][0] - 2*d3fdxdydz[1][0][1] - 2*d3fdxdydz[1][1][0] - d3fdxdydz[1][1][1]
    a[2][2][3] =   18*f[0][0][0] - 18*f[0][0][1] - 18*f[0][1][0] + 18*f[0][1][1] \
                 - 18*f[1][0][0] + 18*f[1][0][1] + 18*f[1][1][0] - 18*f[1][1][1] \
                 + 12*dfdx[0][0][0] - 12*dfdx[0][0][1] - 12*dfdx[0][1][0] + 12*dfdx[0][1][1] \
                 + 6*dfdx[1][0][0] - 6*dfdx[1][0][1] - 6*dfdx[1][1][0] + 6*dfdx[1][1][1] \
                 + 12*dfdy[0][0][0] - 12*dfdy[0][0][1] + 6*dfdy[0][1][0] - 6*dfdy[0][1][1] \
                 - 12*dfdy[1][0][0] + 12*dfdy[1][0][1] - 6*dfdy[1][1][0] + 6*dfdy[1][1][1] \
                 + 9*dfdz[0][0][0] + 9*dfdz[0][0][1] - 9*dfdz[0][1][0] - 9*dfdz[0][1][1] \
                 - 9*dfdz[1][0][0] - 9*dfdz[1][0][1] + 9*dfdz[1][1][0] + 9*dfdz[1][1][1] \
                 + 8*d2fdxdy[0][0][0] - 8*d2fdxdy[0][0][1] + 4*d2fdxdy[0][1][0] - 4*d2fdxdy[0][1][1] \
                 + 4*d2fdxdy[1][0][0] - 4*d2fdxdy[1][0][1] + 2*d2fdxdy[1][1][0] - 2*d2fdxdy[1][1][1] \
                 + 6*d2fdxdz[0][0][0] + 6*d2fdxdz[0][0][1] - 6*d2fdxdz[0][1][0] - 6*d2fdxdz[0][1][1] \
                 + 3*d2fdxdz[1][0][0] + 3*d2fdxdz[1][0][1] - 3*d2fdxdz[1][1][0] - 3*d2fdxdz[1][1][1] \
                 + 6*d2fdydz[0][0][0] + 6*d2fdydz[0][0][1] + 3*d2fdydz[0][1][0] + 3*d2fdydz[0][1][1] \
                 - 6*d2fdydz[1][0][0] - 6*d2fdydz[1][0][1] - 3*d2fdydz[1][1][0] - 3*d2fdydz[1][1][1] \
                 + 4*d3fdxdydz[0][0][0] + 4*d3fdxdydz[0][0][1] + 2*d3fdxdydz[0][1][0] + 2*d3fdxdydz[0][1][1] \
                 + 2*d3fdxdydz[1][0][0] + 2*d3fdxdydz[1][0][1] + d3fdxdydz[1][1][0] + d3fdxdydz[1][1][1]
    a[2][3][0] = - 6*f[0][0][0] + 6*f[0][1][0] + 6*f[1][0][0] - 6*f[1][1][0] \
                 - 4*dfdx[0][0][0] + 4*dfdx[0][1][0] - 2*dfdx[1][0][0] + 2*dfdx[1][1][0] \
                 - 3*dfdy[0][0][0] - 3*dfdy[0][1][0] + 3*dfdy[1][0][0] + 3*dfdy[1][1][0] \
                 - 2*d2fdxdy[0][0][0] - 2*d2fdxdy[0][1][0] - d2fdxdy[1][0][0] - d2fdxdy[1][1][0]
    a[2][3][1] = - 6*dfdz[0][0][0] + 6*dfdz[0][1][0] + 6*dfdz[1][0][0] - 6*dfdz[1][1][0] \
                 - 4*d2fdxdz[0][0][0] + 4*d2fdxdz[0][1][0] - 2*d2fdxdz[1][0][0] + 2*d2fdxdz[1][1][0] \
                 - 3*d2fdydz[0][0][0] - 3*d2fdydz[0][1][0] + 3*d2fdydz[1][0][0] + 3*d2fdydz[1][1][0] \
                 - 2*d3fdxdydz[0][0][0] - 2*d3fdxdydz[0][1][0] - d3fdxdydz[1][0][0] - d3fdxdydz[1][1][0]
    a[2][3][2] =   18*f[0][0][0] - 18*f[0][0][1] - 18*f[0][1][0] + 18*f[0][1][1] \
                 - 18*f[1][0][0] + 18*f[1][0][1] + 18*f[1][1][0] - 18*f[1][1][1] \
                 + 12*dfdx[0][0][0] - 12*dfdx[0][0][1] - 12*dfdx[0][1][0] + 12*dfdx[0][1][1] \
                 + 6*dfdx[1][0][0] - 6*dfdx[1][0][1] - 6*dfdx[1][1][0] + 6*dfdx[1][1][1] \
                 + 9*dfdy[0][0][0] - 9*dfdy[0][0][1] + 9*dfdy[0][1][0] - 9*dfdy[0][1][1] \
                 - 9*dfdy[1][0][0] + 9*dfdy[1][0][1] - 9*dfdy[1][1][0] + 9*dfdy[1][1][1] \
                 + 12*dfdz[0][0][0] + 6*dfdz[0][0][1] - 12*dfdz[0][1][0] - 6*dfdz[0][1][1] \
                 - 12*dfdz[1][0][0] - 6*dfdz[1][0][1] + 12*dfdz[1][1][0] + 6*dfdz[1][1][1] \
                 + 6*d2fdxdy[0][0][0] - 6*d2fdxdy[0][0][1] + 6*d2fdxdy[0][1][0] - 6*d2fdxdy[0][1][1] \
                 + 3*d2fdxdy[1][0][0] - 3*d2fdxdy[1][0][1] + 3*d2fdxdy[1][1][0] - 3*d2fdxdy[1][1][1] \
                 + 8*d2fdxdz[0][0][0] + 4*d2fdxdz[0][0][1] - 8*d2fdxdz[0][1][0] - 4*d2fdxdz[0][1][1] \
                 + 4*d2fdxdz[1][0][0] + 2*d2fdxdz[1][0][1] - 4*d2fdxdz[1][1][0] - 2*d2fdxdz[1][1][1] \
                 + 6*d2fdydz[0][0][0] + 3*d2fdydz[0][0][1] + 6*d2fdydz[0][1][0] + 3*d2fdydz[0][1][1] \
                 - 6*d2fdydz[1][0][0] - 3*d2fdydz[1][0][1] - 6*d2fdydz[1][1][0] - 3*d2fdydz[1][1][1] \
                 + 4*d3fdxdydz[0][0][0] + 2*d3fdxdydz[0][0][1] + 4*d3fdxdydz[0][1][0] + 2*d3fdxdydz[0][1][1] \
                 + 2*d3fdxdydz[1][0][0] + d3fdxdydz[1][0][1] + 2*d3fdxdydz[1][1][0] + d3fdxdydz[1][1][1]
    a[2][3][3] = - 12*f[0][0][0] + 12*f[0][0][1] + 12*f[0][1][0] - 12*f[0][1][1] \
                 + 12*f[1][0][0] - 12*f[1][0][1] - 12*f[1][1][0] + 12*f[1][1][1] \
                 - 8*dfdx[0][0][0] + 8*dfdx[0][0][1] + 8*dfdx[0][1][0] - 8*dfdx[0][1][1] \
                 - 4*dfdx[1][0][0] + 4*dfdx[1][0][1] + 4*dfdx[1][1][0] - 4*dfdx[1][1][1] \
                 - 6*dfdy[0][0][0] + 6*dfdy[0][0][1] - 6*dfdy[0][1][0] + 6*dfdy[0][1][1] \
                 + 6*dfdy[1][0][0] - 6*dfdy[1][0][1] + 6*dfdy[1][1][0] - 6*dfdy[1][1][1] \
                 - 6*dfdz[0][0][0] - 6*dfdz[0][0][1] + 6*dfdz[0][1][0] + 6*dfdz[0][1][1] \
                 + 6*dfdz[1][0][0] + 6*dfdz[1][0][1] - 6*dfdz[1][1][0] - 6*dfdz[1][1][1] \
                 - 4*d2fdxdy[0][0][0] + 4*d2fdxdy[0][0][1] - 4*d2fdxdy[0][1][0] + 4*d2fdxdy[0][1][1] \
                 - 2*d2fdxdy[1][0][0] + 2*d2fdxdy[1][0][1] - 2*d2fdxdy[1][1][0] + 2*d2fdxdy[1][1][1] \
                 - 4*d2fdxdz[0][0][0] - 4*d2fdxdz[0][0][1] + 4*d2fdxdz[0][1][0] + 4*d2fdxdz[0][1][1] \
                 - 2*d2fdxdz[1][0][0] - 2*d2fdxdz[1][0][1] + 2*d2fdxdz[1][1][0] + 2*d2fdxdz[1][1][1] \
                 - 3*d2fdydz[0][0][0] - 3*d2fdydz[0][0][1] - 3*d2fdydz[0][1][0] - 3*d2fdydz[0][1][1] \
                 + 3*d2fdydz[1][0][0] + 3*d2fdydz[1][0][1] + 3*d2fdydz[1][1][0] + 3*d2fdydz[1][1][1] \
                 - 2*d3fdxdydz[0][0][0] - 2*d3fdxdydz[0][0][1] - 2*d3fdxdydz[0][1][0] - 2*d3fdxdydz[0][1][1] \
                 - d3fdxdydz[1][0][0] - d3fdxdydz[1][0][1] - d3fdxdydz[1][1][0] - d3fdxdydz[1][1][1]
    a[3][0][0] =   2*f[0][0][0] - 2*f[1][0][0] + dfdx[0][0][0] + dfdx[1][0][0]
    a[3][0][1] =   2*dfdz[0][0][0] - 2*dfdz[1][0][0] + d2fdxdz[0][0][0] + d2fdxdz[1][0][0]
    a[3][0][2] = - 6*f[0][0][0] + 6*f[0][0][1] + 6*f[1][0][0] - 6*f[1][0][1] \
                 - 3*dfdx[0][0][0] + 3*dfdx[0][0][1] - 3*dfdx[1][0][0] + 3*dfdx[1][0][1] \
                 - 4*dfdz[0][0][0] - 2*dfdz[0][0][1] + 4*dfdz[1][0][0] + 2*dfdz[1][0][1] \
                 - 2*d2fdxdz[0][0][0] - d2fdxdz[0][0][1] - 2*d2fdxdz[1][0][0] - d2fdxdz[1][0][1]
    a[3][0][3] =   4*f[0][0][0] - 4*f[0][0][1] - 4*f[1][0][0] + 4*f[1][0][1] \
                 + 2*dfdx[0][0][0] - 2*dfdx[0][0][1] + 2*dfdx[1][0][0] - 2*dfdx[1][0][1] \
                 + 2*dfdz[0][0][0] + 2*dfdz[0][0][1] - 2*dfdz[1][0][0] - 2*dfdz[1][0][1] \
                 + d2fdxdz[0][0][0] + d2fdxdz[0][0][1] + d2fdxdz[1][0][0] + d2fdxdz[1][0][1]
    a[3][1][0] =   2*dfdy[0][0][0] - 2*dfdy[1][0][0] + d2fdxdy[0][0][0] + d2fdxdy[1][0][0]
    a[3][1][1] =   2*d2fdydz[0][0][0] - 2*d2fdydz[1][0][0] + d3fdxdydz[0][0][0] + d3fdxdydz[1][0][0]
    a[3][1][2] = - 6*dfdy[0][0][0] + 6*dfdy[0][0][1] + 6*dfdy[1][0][0] - 6*dfdy[1][0][1] \
                 - 3*d2fdxdy[0][0][0] + 3*d2fdxdy[0][0][1] - 3*d2fdxdy[1][0][0] + 3*d2fdxdy[1][0][1] \
                 - 4*d2fdydz[0][0][0] - 2*d2fdydz[0][0][1] + 4*d2fdydz[1][0][0] + 2*d2fdydz[1][0][1] \
                 - 2*d3fdxdydz[0][0][0] - d3fdxdydz[0][0][1] - 2*d3fdxdydz[1][0][0] - d3fdxdydz[1][0][1]
    a[3][1][3] =   4*dfdy[0][0][0] - 4*dfdy[0][0][1] - 4*dfdy[1][0][0] + 4*dfdy[1][0][1] \
                 + 2*d2fdxdy[0][0][0] - 2*d2fdxdy[0][0][1] + 2*d2fdxdy[1][0][0] - 2*d2fdxdy[1][0][1] \
                 + 2*d2fdydz[0][0][0] + 2*d2fdydz[0][0][1] - 2*d2fdydz[1][0][0] - 2*d2fdydz[1][0][1] \
                 + d3fdxdydz[0][0][0] + d3fdxdydz[0][0][1] + d3fdxdydz[1][0][0] + d3fdxdydz[1][0][1]
    a[3][2][0] = - 6*f[0][0][0] + 6*f[0][1][0] + 6*f[1][0][0] - 6*f[1][1][0] \
                 - 3*dfdx[0][0][0] + 3*dfdx[0][1][0] - 3*dfdx[1][0][0] + 3*dfdx[1][1][0] \
                 - 4*dfdy[0][0][0] - 2*dfdy[0][1][0] + 4*dfdy[1][0][0] + 2*dfdy[1][1][0] \
                 - 2*d2fdxdy[0][0][0] - d2fdxdy[0][1][0] - 2*d2fdxdy[1][0][0] - d2fdxdy[1][1][0]
    a[3][2][1] = - 6*dfdz[0][0][0] + 6*dfdz[0][1][0] + 6*dfdz[1][0][0] - 6*dfdz[1][1][0] \
                 - 3*d2fdxdz[0][0][0] + 3*d2fdxdz[0][1][0] - 3*d2fdxdz[1][0][0] + 3*d2fdxdz[1][1][0] \
                 - 4*d2fdydz[0][0][0] - 2*d2fdydz[0][1][0] + 4*d2fdydz[1][0][0] + 2*d2fdydz[1][1][0] \
                 - 2*d3fdxdydz[0][0][0] - d3fdxdydz[0][1][0] - 2*d3fdxdydz[1][0][0] - d3fdxdydz[1][1][0]
    a[3][2][2] =   18*f[0][0][0] - 18*f[0][0][1] - 18*f[0][1][0] + 18*f[0][1][1] \
                 - 18*f[1][0][0] + 18*f[1][0][1] + 18*f[1][1][0] - 18*f[1][1][1] \
                 + 9*dfdx[0][0][0] - 9*dfdx[0][0][1] - 9*dfdx[0][1][0] + 9*dfdx[0][1][1] \
                 + 9*dfdx[1][0][0] - 9*dfdx[1][0][1] - 9*dfdx[1][1][0] + 9*dfdx[1][1][1] \
                 + 12*dfdy[0][0][0] - 12*dfdy[0][0][1] + 6*dfdy[0][1][0] - 6*dfdy[0][1][1] \
                 - 12*dfdy[1][0][0] + 12*dfdy[1][0][1] - 6*dfdy[1][1][0] + 6*dfdy[1][1][1] \
                 + 12*dfdz[0][0][0] + 6*dfdz[0][0][1] - 12*dfdz[0][1][0] - 6*dfdz[0][1][1] \
                 - 12*dfdz[1][0][0] - 6*dfdz[1][0][1] + 12*dfdz[1][1][0] + 6*dfdz[1][1][1] \
                 + 6*d2fdxdy[0][0][0] - 6*d2fdxdy[0][0][1] + 3*d2fdxdy[0][1][0] - 3*d2fdxdy[0][1][1] \
                 + 6*d2fdxdy[1][0][0] - 6*d2fdxdy[1][0][1] + 3*d2fdxdy[1][1][0] - 3*d2fdxdy[1][1][1] \
                 + 6*d2fdxdz[0][0][0] + 3*d2fdxdz[0][0][1] - 6*d2fdxdz[0][1][0] - 3*d2fdxdz[0][1][1] \
                 + 6*d2fdxdz[1][0][0] + 3*d2fdxdz[1][0][1] - 6*d2fdxdz[1][1][0] - 3*d2fdxdz[1][1][1] \
                 + 8*d2fdydz[0][0][0] + 4*d2fdydz[0][0][1] + 4*d2fdydz[0][1][0] + 2*d2fdydz[0][1][1] \
                 - 8*d2fdydz[1][0][0] - 4*d2fdydz[1][0][1] - 4*d2fdydz[1][1][0] - 2*d2fdydz[1][1][1] \
                 + 4*d3fdxdydz[0][0][0] + 2*d3fdxdydz[0][0][1] + 2*d3fdxdydz[0][1][0] + d3fdxdydz[0][1][1] \
                 + 4*d3fdxdydz[1][0][0] + 2*d3fdxdydz[1][0][1] + 2*d3fdxdydz[1][1][0] + d3fdxdydz[1][1][1]
    a[3][2][3] = - 12*f[0][0][0] + 12*f[0][0][1] + 12*f[0][1][0] - 12*f[0][1][1] \
                 + 12*f[1][0][0] - 12*f[1][0][1] - 12*f[1][1][0] + 12*f[1][1][1] \
                 - 6*dfdx[0][0][0] + 6*dfdx[0][0][1] + 6*dfdx[0][1][0] - 6*dfdx[0][1][1] \
                 - 6*dfdx[1][0][0] + 6*dfdx[1][0][1] + 6*dfdx[1][1][0] - 6*dfdx[1][1][1] \
                 - 8*dfdy[0][0][0] + 8*dfdy[0][0][1] - 4*dfdy[0][1][0] + 4*dfdy[0][1][1] \
                 + 8*dfdy[1][0][0] - 8*dfdy[1][0][1] + 4*dfdy[1][1][0] - 4*dfdy[1][1][1] \
                 - 6*dfdz[0][0][0] - 6*dfdz[0][0][1] + 6*dfdz[0][1][0] + 6*dfdz[0][1][1] \
                 + 6*dfdz[1][0][0] + 6*dfdz[1][0][1] - 6*dfdz[1][1][0] - 6*dfdz[1][1][1] \
                 - 4*d2fdxdy[0][0][0] + 4*d2fdxdy[0][0][1] - 2*d2fdxdy[0][1][0] + 2*d2fdxdy[0][1][1] \
                 - 4*d2fdxdy[1][0][0] + 4*d2fdxdy[1][0][1] - 2*d2fdxdy[1][1][0] + 2*d2fdxdy[1][1][1] \
                 - 3*d2fdxdz[0][0][0] - 3*d2fdxdz[0][0][1] + 3*d2fdxdz[0][1][0] + 3*d2fdxdz[0][1][1] \
                 - 3*d2fdxdz[1][0][0] - 3*d2fdxdz[1][0][1] + 3*d2fdxdz[1][1][0] + 3*d2fdxdz[1][1][1] \
                 - 4*d2fdydz[0][0][0] - 4*d2fdydz[0][0][1] - 2*d2fdydz[0][1][0] - 2*d2fdydz[0][1][1] \
                 + 4*d2fdydz[1][0][0] + 4*d2fdydz[1][0][1] + 2*d2fdydz[1][1][0] + 2*d2fdydz[1][1][1] \
                 - 2*d3fdxdydz[0][0][0] - 2*d3fdxdydz[0][0][1] - d3fdxdydz[0][1][0] - d3fdxdydz[0][1][1] \
                 - 2*d3fdxdydz[1][0][0] - 2*d3fdxdydz[1][0][1] - d3fdxdydz[1][1][0] - d3fdxdydz[1][1][1]
    a[3][3][0] =   4*f[0][0][0] - 4*f[0][1][0] - 4*f[1][0][0] + 4*f[1][1][0] \
                 + 2*dfdx[0][0][0] - 2*dfdx[0][1][0] + 2*dfdx[1][0][0] - 2*dfdx[1][1][0] \
                 + 2*dfdy[0][0][0] + 2*dfdy[0][1][0] - 2*dfdy[1][0][0] - 2*dfdy[1][1][0] \
                 + d2fdxdy[0][0][0] + d2fdxdy[0][1][0] + d2fdxdy[1][0][0] + d2fdxdy[1][1][0]
    a[3][3][1] =   4*dfdz[0][0][0] - 4*dfdz[0][1][0] - 4*dfdz[1][0][0] + 4*dfdz[1][1][0] \
                 + 2*d2fdxdz[0][0][0] - 2*d2fdxdz[0][1][0] + 2*d2fdxdz[1][0][0] - 2*d2fdxdz[1][1][0] \
                 + 2*d2fdydz[0][0][0] + 2*d2fdydz[0][1][0] - 2*d2fdydz[1][0][0] - 2*d2fdydz[1][1][0] \
                 + d3fdxdydz[0][0][0] + d3fdxdydz[0][1][0] + d3fdxdydz[1][0][0] + d3fdxdydz[1][1][0]
    a[3][3][2] = - 12*f[0][0][0] + 12*f[0][0][1] + 12*f[0][1][0] - 12*f[0][1][1] \
                 + 12*f[1][0][0] - 12*f[1][0][1] - 12*f[1][1][0] + 12*f[1][1][1] \
                 - 6*dfdx[0][0][0] + 6*dfdx[0][0][1] + 6*dfdx[0][1][0] - 6*dfdx[0][1][1] \
                 - 6*dfdx[1][0][0] + 6*dfdx[1][0][1] + 6*dfdx[1][1][0] - 6*dfdx[1][1][1] \
                 - 6*dfdy[0][0][0] + 6*dfdy[0][0][1] - 6*dfdy[0][1][0] + 6*dfdy[0][1][1] \
                 + 6*dfdy[1][0][0] - 6*dfdy[1][0][1] + 6*dfdy[1][1][0] - 6*dfdy[1][1][1] \
                 - 8*dfdz[0][0][0] - 4*dfdz[0][0][1] + 8*dfdz[0][1][0] + 4*dfdz[0][1][1] \
                 + 8*dfdz[1][0][0] + 4*dfdz[1][0][1] - 8*dfdz[1][1][0] - 4*dfdz[1][1][1] \
                 - 3*d2fdxdy[0][0][0] + 3*d2fdxdy[0][0][1] - 3*d2fdxdy[0][1][0] + 3*d2fdxdy[0][1][1] \
                 - 3*d2fdxdy[1][0][0] + 3*d2fdxdy[1][0][1] - 3*d2fdxdy[1][1][0] + 3*d2fdxdy[1][1][1] \
                 - 4*d2fdxdz[0][0][0] - 2*d2fdxdz[0][0][1] + 4*d2fdxdz[0][1][0] + 2*d2fdxdz[0][1][1] \
                 - 4*d2fdxdz[1][0][0] - 2*d2fdxdz[1][0][1] + 4*d2fdxdz[1][1][0] + 2*d2fdxdz[1][1][1] \
                 - 4*d2fdydz[0][0][0] - 2*d2fdydz[0][0][1] - 4*d2fdydz[0][1][0] - 2*d2fdydz[0][1][1] \
                 + 4*d2fdydz[1][0][0] + 2*d2fdydz[1][0][1] + 4*d2fdydz[1][1][0] + 2*d2fdydz[1][1][1] \
                 - 2*d3fdxdydz[0][0][0] - d3fdxdydz[0][0][1] - 2*d3fdxdydz[0][1][0] - d3fdxdydz[0][1][1] \
                 - 2*d3fdxdydz[1][0][0] - d3fdxdydz[1][0][1] - 2*d3fdxdydz[1][1][0] - d3fdxdydz[1][1][1]
    a[3][3][3] =   8*f[0][0][0] - 8*f[0][0][1] - 8*f[0][1][0] + 8*f[0][1][1] \
                 - 8*f[1][0][0] + 8*f[1][0][1] + 8*f[1][1][0] - 8*f[1][1][1] \
                 + 4*dfdx[0][0][0] - 4*dfdx[0][0][1] - 4*dfdx[0][1][0] + 4*dfdx[0][1][1] \
                 + 4*dfdx[1][0][0] - 4*dfdx[1][0][1] - 4*dfdx[1][1][0] + 4*dfdx[1][1][1] \
                 + 4*dfdy[0][0][0] - 4*dfdy[0][0][1] + 4*dfdy[0][1][0] - 4*dfdy[0][1][1] \
                 - 4*dfdy[1][0][0] + 4*dfdy[1][0][1] - 4*dfdy[1][1][0] + 4*dfdy[1][1][1] \
                 + 4*dfdz[0][0][0] + 4*dfdz[0][0][1] - 4*dfdz[0][1][0] - 4*dfdz[0][1][1] \
                 - 4*dfdz[1][0][0] - 4*dfdz[1][0][1] + 4*dfdz[1][1][0] + 4*dfdz[1][1][1] \
                 + 2*d2fdxdy[0][0][0] - 2*d2fdxdy[0][0][1] + 2*d2fdxdy[0][1][0] - 2*d2fdxdy[0][1][1] \
                 + 2*d2fdxdy[1][0][0] - 2*d2fdxdy[1][0][1] + 2*d2fdxdy[1][1][0] - 2*d2fdxdy[1][1][1] \
                 + 2*d2fdxdz[0][0][0] + 2*d2fdxdz[0][0][1] - 2*d2fdxdz[0][1][0] - 2*d2fdxdz[0][1][1] \
                 + 2*d2fdxdz[1][0][0] + 2*d2fdxdz[1][0][1] - 2*d2fdxdz[1][1][0] - 2*d2fdxdz[1][1][1] \
                 + 2*d2fdydz[0][0][0] + 2*d2fdydz[0][0][1] + 2*d2fdydz[0][1][0] + 2*d2fdydz[0][1][1] \
                 - 2*d2fdydz[1][0][0] - 2*d2fdydz[1][0][1] - 2*d2fdydz[1][1][0] - 2*d2fdydz[1][1][1] \
                 + d3fdxdydz[0][0][0] + d3fdxdydz[0][0][1] + d3fdxdydz[0][1][0] + d3fdxdydz[0][1][1] \
                 + d3fdxdydz[1][0][0] + d3fdxdydz[1][0][1] + d3fdxdydz[1][1][0] + d3fdxdydz[1][1][1]

cdef double evaluate_cubic_1d(double a[4], double x) nogil:

    cdef double x2 = x*x
    cdef double x3 = x2*x

    # calculate cubic polynomial
    return a[0]*x3 + a[1]*x2 + a[2]*x + a[3]


cdef double evaluate_cubic_2d(double a[4][4], double x, double y) nogil:

    cdef double x2 = x*x
    cdef double x3 = x2*x

    cdef double y2 = y*y
    cdef double y3 = y2*y

    # calculate cubic polynomial
    return   a[0][0]    + a[0][1]*y    + a[0][2]*y2    + a[0][3]*y3 \
           + a[1][0]*x  + a[1][1]*x*y  + a[1][2]*x*y2  + a[1][3]*x*y3 \
           + a[2][0]*x2 + a[2][1]*x2*y + a[2][2]*x2*y2 + a[2][3]*x2*y3 \
           + a[3][0]*x3 + a[3][1]*x3*y + a[3][2]*x3*y2 + a[3][3]*x3*y3


cdef double evaluate_cubic_3d(double a[4][4][4], double x, double y, double z) nogil:

    cdef double x2 = x*x
    cdef double x3 = x2*x

    cdef double y2 = y*y
    cdef double y3 = y2*y

    cdef double z2 = z*z
    cdef double z3 = z2*z

    return   a[0][0][0]       + a[0][0][1]*z       + a[0][0][2]*z2       + a[0][0][3]*z3 \
           + a[0][1][0]*y     + a[0][1][1]*y*z     + a[0][1][2]*y*z2     + a[0][1][3]*y*z3 \
           + a[0][2][0]*y2    + a[0][2][1]*y2*z    + a[0][2][2]*y2*z2    + a[0][2][3]*y2*z3 \
           + a[0][3][0]*y3    + a[0][3][1]*y3*z    + a[0][3][2]*y3*z2    + a[0][3][3]*y3*z3 \
           + a[1][0][0]*x     + a[1][0][1]*x*z     + a[1][0][2]*x*z2     + a[1][0][3]*x*z3 \
           + a[1][1][0]*x*y   + a[1][1][1]*x*y*z   + a[1][1][2]*x*y*z2   + a[1][1][3]*x*y*z3 \
           + a[1][2][0]*x*y2  + a[1][2][1]*x*y2*z  + a[1][2][2]*x*y2*z2  + a[1][2][3]*x*y2*z3 \
           + a[1][3][0]*x*y3  + a[1][3][1]*x*y3*z  + a[1][3][2]*x*y3*z2  + a[1][3][3]*x*y3*z3 \
           + a[2][0][0]*x2    + a[2][0][1]*x2*z    + a[2][0][2]*x2*z2    + a[2][0][3]*x2*z3 \
           + a[2][1][0]*x2*y  + a[2][1][1]*x2*y*z  + a[2][1][2]*x2*y*z2  + a[2][1][3]*x2*y*z3 \
           + a[2][2][0]*x2*y2 + a[2][2][1]*x2*y2*z + a[2][2][2]*x2*y2*z2 + a[2][2][3]*x2*y2*z3 \
           + a[2][3][0]*x2*y3 + a[2][3][1]*x2*y3*z + a[2][3][2]*x2*y3*z2 + a[2][3][3]*x2*y3*z3 \
           + a[3][0][0]*x3    + a[3][0][1]*x3*z    + a[3][0][2]*x3*z2    + a[3][0][3]*x3*z3 \
           + a[3][1][0]*x3*y  + a[3][1][1]*x3*y*z  + a[3][1][2]*x3*y*z2  + a[3][1][3]*x3*y*z3 \
           + a[3][2][0]*x3*y2 + a[3][2][1]*x3*y2*z + a[3][2][2]*x3*y2*z2 + a[3][2][3]*x3*y2*z3 \
           + a[3][3][0]*x3*y3 + a[3][3][1]*x3*y3*z + a[3][3][2]*x3*y3*z2 + a[3][3][3]*x3*y3*z3