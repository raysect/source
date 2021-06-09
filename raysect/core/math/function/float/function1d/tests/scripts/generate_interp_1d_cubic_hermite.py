
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
    X_EXTRAP_DELTA_MAX, X_EXTRAP_DELTA_MIN, PRECISION, BIG_VALUE_FACTOR, SMALL_VALUE_FACTOR

from raysect.core.math.function.float.function1d.interpolate import Interpolate1D
import numpy as np
from scipy.interpolate import CubicHermiteSpline, interp1d
import scipy

# Force scientific format to get the right number of significant figures
np.set_printoptions(30000, linewidth=100, formatter={'float': lambda x_str: format(x_str, '.'+str(PRECISION)+'E')})


def function_to_spline(x_func, factor):
    return factor*np.sin(x_func)


def linear_extrapolation(m, x2, x1, f1):
    return f1 + m*(x2-x1)


def linear_interpolation(x_interp, x1, f1, x2, f2):
    return f1 + (f2 - f1) * (x_interp - x1)/(x2 - x1)


def calc_gradient(x_spline, y_spline, index_left):
    if index_left == 0:
        dfdx = (y_spline[index_left + 1] - y_spline[index_left])
    elif index_left == len(x_spline) - 1:
        dfdx = y_spline[index_left] - y_spline[index_left - 1]
    else:
        # Finding the normalised distance x_eff
        x_eff = (x_spline[index_left + 1] - x_spline[index_left - 1]) \
                / (x_spline[index_left + 1] - x_spline[index_left])
        if x_eff != 0:
            dfdx = (y_spline[index_left + 1] - y_spline[index_left - 1]) / x_eff
        else:
            raise ZeroDivisionError('Two adjacent spline points have the same x value!')
    return dfdx


def different_quadratic_extrpolation_lower(x_interp, x_spline, y_spline):
    """
    This quadratic is determined by fixing the quadratic gradient at the 2 points near the lower border of the data
    set, then fixing the quadratic to pass through the lower data point.
    """
    index_lower_1 = 0
    index_lower_2 = 1
    x1_lower = x_spline[index_lower_1]
    x2_lower = x_spline[index_lower_2]
    f1_lower = y_spline[index_lower_1]

    df1_dx_lower = calc_gradient(x_spline, y_spline, index_lower_1)/(x2_lower - x1_lower)
    df2_dx_lower = calc_gradient(x_spline, y_spline, index_lower_2)/(x2_lower - x1_lower)

    # Solve 2ax-b = df_dx for the gradient at point 1 and 2
    # Rearrange both equations to find 'a' and 'b' quadratic coefficients
    a_lower = (df2_dx_lower - df1_dx_lower)/(2.*(x2_lower - x1_lower))
    b_lower = df1_dx_lower - 2.*a_lower*x1_lower

    # Find c by solving at the fixed points (f = a x**2 + bx + c) at point 1 for the lower, and point 2 for the upper
    c_lower = f1_lower - a_lower*x1_lower**2 - b_lower*x1_lower
    print(a_lower, b_lower, c_lower)
    return a_lower*x_interp**2 + b_lower*x_interp + c_lower


def different_quadratic_extrpolation_upper(x_interp, x_spline, y_spline):
    """
    This quadratic is determined by fixing the quadratic gradient at the 2 points near the upper border of the data
    set, then fixing the quadratic to pass through the upper data point.
    """

    index_upper_1 = len(x_spline) - 2
    index_upper_2 = len(x_spline) - 1
    x1_upper = x_spline[index_upper_1]
    x2_upper = x_spline[index_upper_2]
    f2_upper = y_spline[index_upper_2]

    df1_dx_upper = calc_gradient(x_spline, y_spline, index_upper_1)/(x2_upper - x1_upper)
    df2_dx_upper = calc_gradient(x_spline, y_spline, index_upper_2)/(x2_upper - x1_upper)

    # Solve 2ax-b = df_dx for the gradient at point 1 and 2
    # Rearrange both equations to find 'a' and 'b' quadratic coefficients
    a_upper = (df2_dx_upper - df1_dx_upper)/(2.*(x2_upper - x1_upper))
    b_upper = df1_dx_upper - 2.*a_upper*x1_upper

    # Find c by solving at the fixed points (f = a x**2 + bx + c) at point 1 for the lower, and point 2 for the upper
    c_upper = f2_upper - a_upper*x2_upper**2 - b_upper*x2_upper
    print(a_upper, b_upper, c_upper)
    return a_upper*x_interp**2 + b_upper*x_interp + c_upper


# Calculate for big values, small values, or normal values
big_values = False
small_values = False

print('Using scipy version', scipy.__version__)

# Create array to generate spline knots on, and find their functional value
x = np.linspace(X_LOWER, X_UPPER, NB_X)

# Make the sampled points between spline knots and find the precalc_interpolation used in test_interpolator.setup_cubic
xsamples = np.linspace(X_LOWER, X_UPPER, NB_XSAMPLES)

# Find the function values to be used
if big_values:
    factor = np.power(10., BIG_VALUE_FACTOR)
elif small_values:
    factor = np.power(10., SMALL_VALUE_FACTOR)
else:
    factor = 1.

data_f = function_to_spline(x, factor)
precalc_interpolation_function_vals = function_to_spline(xsamples, factor)

print('Save this to self.data in test_interpolator:\n', repr(data_f))
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
      'Save this to self.precalc_interpolation in test_interpolator in setup_cubic:\n', repr(f_out))

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

f_extrap_quadratic = np.array([
    different_quadratic_extrpolation_lower(xsamples_extrap[0], x, data_f),
    different_quadratic_extrpolation_lower(xsamples_extrap[1], x, data_f),
    different_quadratic_extrpolation_upper(xsamples_extrap[-2], x, data_f),
    different_quadratic_extrpolation_upper(xsamples_extrap[-1], x, data_f),
])

print('Output of quadratically extrapolating from the start and end spline knots ',
      'Save this to self.precalc_extrapolation_quadratic in test_interpolator:\n', repr(f_extrap_quadratic))

# Alternative linear test (use scipy instead)
x_lower_array = x[:-1]
x_upper_array = x[1:]
f_linear_out = np.zeros(len(xsamples))
for i in range(len(xsamples)):
    index = np.where(np.logical_and(x_lower_array <= xsamples[i], x_upper_array >= xsamples[i]))
    index_found = index[0][0]
    f_linear_out[i] = linear_interpolation(
        xsamples[i], x[index_found], data_f[index_found], x[index_found + 1], data_f[index_found + 1]
    )

linear_1d_interp = interp1d(x, data_f, kind='linear')
f_linear_out = linear_1d_interp(xsamples)
print('Linear spline at xsamples created using. interp1d(kind=linear)',
      'Save this to self.precalc_interpolation in test_interpolator in setup_linear:\n', repr(f_linear_out))

check_plot = True
if check_plot:
    import matplotlib.pyplot as plt

    interp_cubic_extrap_nearest = Interpolate1D(
        x, data_f, 'linear', 'quadratic', extrapolation_range=2.0
    )
    print(interp_cubic_extrap_nearest.test_edge_gradients())
    fig, ax = plt.subplots()
    f_check = np.zeros(len(xsamples))
    for i in range(len(xsamples)):
        f_check[i] = interp_cubic_extrap_nearest(xsamples[i])
    ax.plot(xsamples, f_out, '-r')
    ax.plot(xsamples, f_check, 'bx')
    f_check_extrap = np.zeros(len(xsamples_extrap))
    for i in range(len(xsamples_extrap)):
        f_check_extrap[i] = interp_cubic_extrap_nearest(xsamples_extrap[i])
    ax.plot(xsamples_extrap, f_check_extrap, 'mo')
    ax.plot(xsamples, f_linear_out, 'go')
    ax.plot(xsamples_extrap[0], different_quadratic_extrpolation_lower(xsamples_extrap[0], x, data_f), 'ko')
    ax.plot(xsamples_extrap[-1], different_quadratic_extrpolation_upper(xsamples_extrap[-1], x, data_f), 'ko')
    ax.plot(x, data_f, 'bo')
    plt.show()

