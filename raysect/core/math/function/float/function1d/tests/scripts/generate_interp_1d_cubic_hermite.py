
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

from raysect.core.math.function.float.function1d.tests.test_interpolator import X_LOWER, X_UPPER, NB_XSAMPLES, NB_X
from raysect.core.math.function.float.function1d.interpolate import _Interpolator1DCubic, Interpolate1D
import numpy as np
from scipy.interpolate import CubicHermiteSpline
import scipy


def function_to_spline(x_func):
    return np.sin(x_func)


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
cubic_interpolator_object = _Interpolator1DCubic(x, data_f)
df_dx = np.zeros(len(x))
df_dx[-1] = cubic_interpolator_object._test_calc_gradient(len(x) - 1) / (x[-1] - x[-2])
for i in range(len(x) - 1):
    df_dx[i] = cubic_interpolator_object._test_calc_gradient(i) / (x[i + 1] - x[i])

# Input spline knots and the gradient estimates into a known cubic interpolator similar to what we want to achieve
cubic_hermite = CubicHermiteSpline(x, data_f, df_dx)

# Calculate the cubic spline at the sampled points
f_out = cubic_hermite(xsamples)

print('Output of 3rd party cubic spline at xsamples. ',
      'Save this to self.precalc_extrapolation_cubic in test_interpolator:\n', repr(f_out))

check_plot = False
if check_plot:
    import matplotlib.pyplot as plt

    interp_cubic_extrap_nearest = Interpolate1D(
        x, data_f, 'cubic', 'nearest', extrapolation_range=2.0
    )
    fig, ax = plt.subplots()
    f_check = np.zeros(len(xsamples))
    for i in range(len(xsamples)):
        f_check[i] = interp_cubic_extrap_nearest(xsamples[i])
    ax.plot(xsamples, f_out, '-r')
    ax.plot(xsamples, f_check, 'bx')
    ax.plot(x, data_f, 'bo')
    plt.show()

