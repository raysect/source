import numpy as np
from raysect.core.math.function.float.function1d.interpolate import Interpolate1D, InterpType, ExtrapType

import matplotlib.pyplot as plt

x_in = np.arange(-1.73, -1.4, 0.1)  # np.array([0., 1., 2., 3., 4.])
y_in = np.sin(x_in)  # np.array([0., 1., 2., 3., 4.]
interp1d = Interpolate1D(x_in, y_in, InterpType.CubicInt, ExtrapType.NearestExt, extrapolation_range=2.0)
fig, ax = plt.subplots()
ax.plot(x_in, y_in, 'go')

range_interp = np.arange(-1.7, -1.5, 0.01)
for i in range(len(range_interp)):
    ax.plot([range_interp[i]], [interp1d(range_interp[i])], 'rx')

plt.show()