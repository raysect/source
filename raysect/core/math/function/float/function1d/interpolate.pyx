import numpy as np
from numpy cimport ndarray
cimport cython
from raysect.core.math.cython.interpolation.linear cimport linear1d
from raysect.core.math.cython.utility cimport find_index, lerp


cdef class Interpolate1D(Function1D):
    """
    Base class for Function1D interpolators.

    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    :param InterpType interpolation_type: Type of interpolation to use (linear, cubical)
    :param ExtrapType extrapolation_type: Type of extrapolation to use (no, nearest, linear, quadratic)
    """
    def __init__(self, object x, object f, InterpType interpolation_type,
                 ExtrapType extrapolation_type, double extrapolation_range):

        self.x = np.array(x, dtype=np.float64)
        self.f = np.array(f, dtype=np.float64)
        self._x = x
        self._f = f
        self._last_index = self._x.shape[0] -1
        self._extrapolation_range = extrapolation_range

        # dimensions checks
        if x.ndim != 1:
            raise ValueError(f"The x array must be 1D. Got {x.shape}")

        if f.ndim != 1:
            raise ValueError(f"The x array must be 1D. Got {f.shape}")

        if x.shape != f.shape:
            raise ValueError(f"Shape mismatch between x array ({x.shape}) and f array ({f.shape})")

        # test monotonicity
        if (np.diff(x) <= 0).any():
            raise ValueError("The x array must be monotonically increasing.")

        # create appropriate extrapolator to be passed to the actual interpolator
        if extrapolation_type == ExtrapType.NoExt:
            self._extrapolator = ExtrapolatorNone(x, f, extrapolation_range)
        elif extrapolation_type == ExtrapType.NearestExt:
            self._extrapolator = Extrapolator1DNearest(x, f, extrapolation_range)
        elif extrapolation_type == ExtrapType.LinearExt:
            self._extrapolator = Extrapolator1DLinear(x, f, extrapolation_range)
        else:
            raise ValueError(f"Unsupported extrapolator type {extrapolation_type}")

        # create interpolator
        if interpolation_type == InterpType.LinearInt:
            self._interpolator = Interpolator1DLinear(self._x, self._f)
        elif interpolation_type == InterpType.CubicInt:
            self._interpolator = Interpolator1DCubic(self._x, self._f)
        else:
            raise ValueError(f"Interpolation type {interpolation_type} not supported")


    cdef double evaluate(self, double x) except? -1e999:
        """
        Evaluates the interpolating function.

        :param double x: x coordinate.
        :return: the interpolated value.
        """
        cdef int index = find_index(self._x, x)

        if index == -1:
            if x < self._x[0] - self._extrapolation_range:
                raise ValueError(
                    f"The specified value (x={x}) is outside of extrapolation range")
            return self._extrapolator.evaluate(x, index)
        elif index == self._last_index:
            if x > self._x[self._last_index] + self._extrapolation_range:
                raise ValueError(
                    f"The specified value (x={x}) is outside of extrapolation range")
            return self._extrapolator.evaluate(x, index)
        else:
            return self._interpolator.evaluate(x, index)


    @property
    def domain(self):
        """
        Returns bounding box of the provided inputs.
        Order: min(x), max(x), min(f), max(f)
        :warning: doesn't take extrapolator into account at the moment
        """
        @property
        def domain(self):
            return np.min(self._x), np.max(self._x), np.min(self._f), np.max(self._f)

cdef class _Interpolator1D:
    cdef double evaluate(self, double px, int idx) except? -1e999:
        raise NotImplementedError("_Interpolator is an abstract base class")


cdef class Interpolator1DLinear(_Interpolator1D):
    """
    Linear interpolation of 1D function
    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    :param Extrapolator1D extrapolator: extrapolator object
    """
    def __init__(self, double[::1] x, double[::1] f):
        self._x = x
        self._f = f

    cdef double evaluate(self, double px, int idx) except? -1e999:
        return linear1d(self._x[idx], self._x[idx + 1], self._f[idx], self._f[idx + 1], px)


cdef class Interpolator1DCubic(_Interpolator1D):
    """
    Cubic interpolation of 1D function
    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    :param Extrapolator1D extrapolator: extrapolator object
    """
    def __init__(self, object x, object f):
        self._x = np.array(x, dtype=np.float64)
        self._f = np.array(f, dtype=np.float64)

        self._x_mv = self._x
        self._f_mv = self._f

        cdef int n
        n = len(x)
        self._n = n
        self._mask_a = np.zeros((n - 1,), dtype=np.float64)  # Where 'a' has been calculated already
        self._mask_dfdx = np.zeros((n,), dtype=np.float64)  # Where 'dfdx' has been calculated already
        self._dfdx = np.zeros((n,), dtype=np.float64)
        self._a = np.zeros((n - 1, 4), dtype=np.float64)
        self._a_mv = self._a

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef double get_gradient(self, double[::1] x_spline, double[::1] y_spline, int index):
        """
        Calculate the normalised gradient at x_spline[index] based on the central difference approximation unless at
        the edges of the array x_spline.

        At x[i], the gradient is normally estimated using the central difference approximation [y[i-1], y[i+1]]/2
        For a normalised range x[i], x[i+1] between 0 and 1, this is the same except for unevenly spaced data.
        Unevenly spaced data has a normalisation x[i-1] - x[i+1] != 2, it is defined as x_eff in this function by
        re-scaling the distance x[i-1] - x[i+1] using normalisation (x[i+1] - x[i]) = 1.

        At the start and end of the array, the forward or backward difference approximation is calculated over
        a  (x[i+1] - x[i]) = 1 or  (x[i] - x[i-1]) = 1 respectively. The end spline gradient is not used for
        extrapolation

        .. WARNING:: For speed, this function does not perform any zero division, type or bounds
          checking. Supplying malformed data may result in data corruption or a
          segmentation fault.

        :param double[::1] x_spline: A memory view to a double array containing monotonically increasing values.
        :param double[::1] y_spline: The desired spline points corresponding function returned values
        :param int index: The index of the lower spline point that the gradient is to be calculated for
        """
        # Calculate central difference method, but at the start of end of the array use the forward/back difference
        cdef double dfdx
        cdef double x_eff
        if index == 0:
            dfdx = (y_spline[index + 1] - y_spline[index])
        elif index == self._n - 1:
            dfdx = y_spline[index] - y_spline[index - 1]
        else:
            # if equally spaced this would be divided by 2. Not guaranteed so work out the total normalised distance
            x_eff = (x_spline[index + 1] - x_spline[index - 1])/(x_spline[index + 1] - x_spline[index])
            dfdx = (y_spline[index + 1]-y_spline[index - 1])/x_eff
        return dfdx

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double px, int idx) except? -1e999:
        cdef int index_use = find_index(self._x, px)
        # rescale x between 0 and 1
        cdef double x_scal
        cdef double x_bound = (self._x_mv[index_use + 1] - self._x_mv[index_use])
        if x_bound != 0:
            x_scal = (px - self._x_mv[index_use]) / x_bound
        else:
            raise ZeroDivisionError("Two adjacent spline points have the same x value!")

        # Calculate the coefficients (and gradients at each spline point) if they dont exist
        if not self._mask_a[index_use]:
            if not self._mask_dfdx[index_use]:
                self._dfdx[index_use] = self.get_gradient(self._x_mv, self._f_mv, index_use)
                self._mask_dfdx[index_use] = 1
            if not self._mask_dfdx[index_use + 1]:
                self._dfdx[index_use + 1] = self.get_gradient(self._x_mv, self._f_mv, index_use + 1)
                self._mask_dfdx[index_use + 1] = 1
            self._a_mv[index_use, :] = self.calc_coefficients_1d(self._f_mv[index_use], self._f_mv[index_use + 1], self._dfdx[index_use], self._dfdx[index_use + 1])
            self._mask_a[index_use] = 1
        return self._a_mv[index_use, 0] * x_scal ** 3 + self._a_mv[index_use, 1] * x_scal ** 2 + self._a_mv[index_use, 2] * x_scal + self._a_mv[index_use, 3]


    @cython.initializedcheck(False)
    cdef double[:] calc_coefficients_1d(self, double f1, double f2, double dfdx1, double dfdx2):
        """
        Calculate the cubic spline coefficients between 2 spline points.

        The gradient is pre-calculated before filling out the matrix which corresponds to the inverse of a matrix
        which constrains the cubic at x=0 and x=1 in row 1 and 2 respectively, followed by constraining the
        gradient of the cubic at x=0 and x=1 to the supplied gradient in rows 3-4.

        .. WARNING:: For speed, this function does not perform any initialization
          checking. Supplying malformed data may result in data corruption or a
          segmentation fault.

        :param double f1: The functional value at the first spline point (x=0)
        :param double f2: The functional value at the second spline point (x=1)
        :param double dfdx1: The gradient value at the first spline point (x=0 at respective index)
        :param double dfdx2: The gradient value at the second spline point (x=1 at respective index)
        """
        cdef ndarray a = np.zeros((4, ), dtype=np.float64)
        cdef double[:] a_mv = a
        a_mv[0] = 0.5 * f1 - 0.5 * f2 + 0.5 * dfdx2
        a_mv[1] = -1.5 * f1 + 1.5 * f2 - 1. * dfdx1 - 0.5 * dfdx2
        a_mv[2] = 1. * dfdx1
        a_mv[3] = 1. * f1
        return a_mv

    @property
    def domain(self):
        raise NotImplementedError(f"{self.__class__} not implemented")


cdef class _Extrapolator1D:
    """
    Base class for Function1D extrapolators.

    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    :param double extrapolation_range: Range covered by the extrapolator.
    Padded symmetrically to both ends of the input.
    """
    typename = NotImplemented

    def __init__(self, double[::1] x, double[::1] f, double range):
        self._range = range
        self._x = x
        self._f = f

    # @property
    # def range(self):
    #     """
    #     Range covered either by the original x input or by the extrapolator
    #     """
    #     min_range = self._x[0] - self._range
    #     max_range = self._x[self._x.shape[0] - 1] + self._range
    #     return min_range, max_range

    cdef double extrapolate(self, double px, int order, int index, double rx) except? -1e999:
        raise NotImplementedError(f"{self.__class__} not implemented")

    cdef double evaluate(self, double px, int idx) except? -1e999:
        raise NotImplementedError(f"{self.__class__} not implemented")


cdef class ExtrapolatorNone(_Extrapolator1D):
    """
    Extrapolator that does nothing.
    """
    cdef double extrapolate(self, double px, int order, int index, double rx) except? -1e999:
        raise ValueError(f"Extrapolation not available. Interpolate within function range {np.min(self._x)}-{np.max(self._x)}")

    cdef double evaluate(self, double px, int idx)  except? -1e999:
        raise ValueError("Extrapolation not available.")

cdef class Extrapolator1DNearest(_Extrapolator1D):
    """
    Extrapolator that returns nearest input value
    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    :param double extrapolation_range: Range covered by the extrapolator.
    """

    typename = 'nearest'

    def __init__(self, double [::1] x, double[::1] f, double extrapolation_range):
        self._x = x
        self._f = f
        self._range = extrapolation_range
        self._last_index = self._x.shape[0] -1

    cdef double evaluate(self, double px, int idx) except? -1e999:
        if px < self._x[0]:
            return self._f[0]
        elif px >= self._x[self._last_index]:
            return self._f[self._last_index]


cdef class Extrapolator1DLinear(_Extrapolator1D):
    """
    Extrapolator that extrapolates linearly
    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    :param double extrapolation_range: Range covered by the extrapolator.
    """

    typename = 'linear'

    def __init__(self, double [::1] x, double[::1] f, double extrapolation_range):
        self._x = x
        self._f = f
        self._range = extrapolation_range
        self._last_index = self._x.shape[0] -1

    cdef double evaluate(self, double px, int idx) except? -1e999:
        if idx == -1:
            idx += 1
        else:
            idx -= 1
        return lerp(self._x[idx], self._x[idx + 1], self._f[idx], self._f[idx + 1], px)
