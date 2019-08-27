# cython: language_level=3

# Copyright (c) 2014-2018, Dr Alex Meakins, Raysect Project
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
from libc.stdint cimport uint8_t
import numpy as np
cimport numpy as np

def test(x, y):

    return cubic2d(
        np.array([0.0, 5.0]),
        np.array([2.0, 4.0]),
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([[0.0, 0.0], [-1.0, -1.0]]),
        np.array([[0.0, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 0.0]]),
        x,
        y
    )

# todo: SHOULD THIS DO NORMALISATION? OR SHOULD IT BE A UNIT SQUARE?
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef double cubic2d(double[::1] vx, double[::1] vy, double[:,::1] f, double[:,::1] dfdx, double[:,::1] dfdy,
                    double[:,::1] d2fdxdy, double tx, double ty) nogil:

    cdef:
        int i, j
        double dx, dy, nx, ny
        double nf[2][2]
        double ndfdx[2][2]
        double ndfdy[2][2]
        double nd2fdxdy[2][2]
        double a[4][4]

    # normalise onto unit square
    dx = vx[1] - vx[0]
    dy = vy[1] - vy[0]

    nx = (tx - vx[0]) / dx
    ny = (ty - vy[0]) / dy

    for i in range(2):
        for j in range(2):
            nf[i][j] = f[i][j]
            ndfdx[i][j] = dfdx[i][j] * dx
            ndfdy[i][j] = dfdy[i][j] * dy
            nd2fdxdy[i][j] = d2fdxdy[i][j] * dx * dy

    calc_coefficients_2d(nf, ndfdx, ndfdy, nd2fdxdy, a)
    return evaluate_cubic_2d(a, nx, ny)


# todo: SHOULD THIS DO NORMALISATION? OR SHOULD IT BE A UNIT SQUARE?
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef double cubic3d(double[::1] vx, double[::1] vy, double[::1] vz, double[:,:,::1] f,
                    double[:,:,::1] dfdx, double[:,:,::1] dfdy, double[:,:,::1] dfdz,
                    double[:,:,::1] d2fdxdy, double[:,:,::1] d2fdxdz, double[:,:,::1] d2fdydz,
                    double[:,:,::1] d3fdxdydz, double tx, double ty, double tz) nogil:

    cdef:
        int i, j, k
        double dx, dy, dz, nx, ny, nz
        double nf[2][2][2]
        double ndfdx[2][2][2]
        double ndfdy[2][2][2]
        double ndfdz[2][2][2]
        double nd2fdxdy[2][2][2]
        double nd2fdxdz[2][2][2]
        double nd2fdydz[2][2][2]
        double nd3fdxdydz[2][2][2]
        double a[4][4][4]

    # normalise onto unit square
    dx = vx[1] - vx[0]
    dy = vy[1] - vy[0]
    dz = vz[1] - vz[0]

    nx = (tx - vx[0]) / dx
    ny = (ty - vy[0]) / dy
    nz = (tz - vz[0]) / dz

    for i in range(2):
        for j in range(2):
            for k in range(2):
                nf[i][j][k] = f[i][j][k]
                ndfdx[i][j][k] = dfdx[i][j][k] * dx
                ndfdy[i][j][k] = dfdy[i][j][k] * dy
                ndfdz[i][j][k] = dfdy[i][j][k] * dz
                nd2fdxdy[i][j][k] = d2fdxdy[i][j][k] * dx * dy
                nd2fdxdz[i][j][k] = d2fdxdy[i][j][k] * dx * dz
                nd2fdydz[i][j][k] = d2fdxdy[i][j][k] * dy * dz
                nd3fdxdydz[i][j][k] = d2fdxdy[i][j][k] * dx * dy * dz

    calc_coefficients_3d(nf, ndfdx, ndfdy, ndfdz, nd2fdxdy, nd2fdxdz, nd2fdydz, nd3fdxdydz, a)
    return evaluate_cubic_3d(a, nx, ny, nz)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void calc_coefficients_2d(double f[2][2], double dfdx[2][2], double dfdy[2][2], double d2fdxdy[2][2], double a[4][4]) nogil:

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