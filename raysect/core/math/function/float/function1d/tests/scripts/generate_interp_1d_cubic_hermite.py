
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
This script has been used to calculate the reference data for the 1D cubic interpolator tests.
"""

from raysect.core.math.function.float.function1d.tests.test_interpolator import X_LOWER, X_UPPER, NB_XSAMPLES, NB_X, \
    X_EXTRAP_DELTA_MAX, X_EXTRAP_DELTA_MIN, PRECISION

from raysect.core.math.function.float.function1d.interpolate import Interpolate1D
import numpy as np
from scipy.interpolate import CubicHermiteSpline
import scipy


def function_to_spline(x_func):
    return np.sin(x_func)


def linear_extrapolation(m, x2, x1, f1):
    return f1 + m*(x2-x1)

print('Using scipy version', scipy.__version__)

# Create array to generate spline knots on, and find their functional value
x = np.linspace(X_LOWER, X_UPPER, NB_X)
data_f = function_to_spline(x)
print('Save this to self.data in test_interpolator:\n', repr(data_f))

# Make the sampled points between spline knots and find the precalc_interpolation used in test_interpolator.setup_cubic
xsamples = np.linspace(X_LOWER, X_UPPER, NB_XSAMPLES)
precalc_interpolation_function_vals = function_to_spline(xsamples)
print('Save this to self.precalc_function in test_interpolator:\n', repr(precalc_interpolation_function_vals))

# Find the unnormalised gradient at each spline knot
df_dx = np.zeros(len(x))
df_dx[-1] = calc_gradient(x, data_f, len(x) - 1) / (x[-1] - x[-2])
for i in range(len(x) - 1):
    df_dx[i] = calc_gradient(x, data_f, i) / (x[i + 1] - x[i])

# Input spline knots and the gradient estimates into a known cubic interpolator similar to what we want to achieve
cubic_hermite = CubicHermiteSpline(x, data_f, df_dx)

# Calculate the cubic spline at the sampled points
f_out = cubic_hermite(xsamples)

print('Output of 3rd party cubic spline at xsamples. ',
      'Save this to self.precalc_interpolation in test_interpolator:\n', repr(f_out))

# Extrapolation x values
xsamples_extrap = np.array([
    X_LOWER - X_EXTRAP_DELTA_MAX, X_LOWER - X_EXTRAP_DELTA_MIN, X_UPPER + X_EXTRAP_DELTA_MIN,
    X_UPPER + X_EXTRAP_DELTA_MAX], dtype=np.float64,
)

f_extrap_nearest = np.array([data_f[0], data_f[0], data_f[-1], data_f[-1]])
expected_start_grad = ((data_f[1] - data_f[0]) / (x[1] - x[0]))
expected_end_grad = ((data_f[-2] - data_f[-1]) / (x[-2] - x[-1]))
print('Output of nearest neighbour extrapolation from the start and end spline knots ',
      'Save this to self.precalc_extrapolation_nearest in test_interpolator:\n', repr(f_extrap_nearest))

f_extrap_linear = linear_extrapolation(
    np.array([expected_start_grad, expected_start_grad, expected_end_grad, expected_end_grad]), xsamples_extrap,
    np.array([x[0], x[0], x[-1], x[-1]]), np.array([data_f[0], data_f[0], data_f[-1], data_f[-1]])
)
print('Output of linearly extrapolating from the start and end spline knots ',
      'Save this to self.precalc_extrapolation_linear in test_interpolator:\n', repr(f_extrap_linear))

check_plot = False
if check_plot:
    import matplotlib.pyplot as plt

    interp_cubic_extrap_nearest = Interpolate1D(
        x, data_f, 'cubic', 'linear', extrapolation_range=2.0
    )
    fig, ax = plt.subplots()
    f_check = np.zeros(len(xsamples))
    for i in range(len(xsamples)):
        f_check[i] = interp_cubic_extrap_nearest(xsamples[i])
        print(xsamples[i], f_check[i], f_out[i])
    ax.plot(xsamples, f_out, '-r')
    ax.plot(xsamples, f_check, 'bx')
    ax.plot(xsamples_extrap, cubic_hermite(xsamples_extrap), 'bx')
    ax.plot(x, data_f, 'bo')
    plt.show()

