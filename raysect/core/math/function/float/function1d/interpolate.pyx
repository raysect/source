import numpy as np
from numpy cimport ndarray

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

        x = np.array(x, dtype=np.float64)
        f = np.array(f, dtype=np.float64)

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
            extrapolator = ExtrapolatorNone(x, f, extrapolation_range)
        elif extrapolation_type == ExtrapType.NearestExt:
            extrapolator = Extrapolator1DNearest(x, f, extrapolation_range)
        elif extrapolation_type == ExtrapType.LinearExt:
            extrapolator = Extrapolator1DLinear(x, f, extrapolation_range)
        else:
            raise ValueError(f"Unsupported extrapolator type {extrapolation_type}")

        # create interpolator
        if interpolation_type == InterpType.LinearInt:
            self._impl = Interpolate1DLinear(x, f, extrapolator)
        elif interpolation_type == InterpType.CubicInt:
            self._impl = Interpolate1DCubic(x, f, extrapolator)
        else:
            raise ValueError(f"Interpolation type {interpolation_type} not supported")

    cdef double evaluate(self, double x) except? -1e999:
        """
        Evaluates the interpolating function.

        :param double x: x coordinate.
        :return: the interpolated value.
        """

        return self._impl.evaluate(x)

    @property
    def domain(self):
        """
        Returns bounding box of the provided inputs.
        Order: min(x), max(x), min(f), max(f)
        :warning: doesn't take extrapolator into account at the moment
        """
        return self._impl.domain

cdef class Interpolate1DLinear(Interpolate1D):
    """
    Linear interpolation of 1D function
    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    :param Extrapolator1D extrapolator: extrapolator object
    """
    def __init__(self, object x, object f, Extrapolator1D extrapolator):
        self._x = np.array(x, dtype=np.float64)
        self._f = np.array(f, dtype=np.float64)
        self._extrapolator = extrapolator

    cdef double evaluate(self, double x) except? -1e999:
        cdef int index = find_index(self._x, x)
        cdef int nx = self._x.shape[0]

        if index == -1:
            if x < self._extrapolator.range[0]:
                raise ValueError(f"The specified value (x={x}) is outside of extrapolation range {self._extrapolator.range}")
            return self._extrapolator.extrapolate(x, 0, 0, self._x[0])
        elif index == nx - 1:
            if x > self._extrapolator.range[1]:
                raise ValueError(f"The specified value (x={x}) is outside of extrapolation range {self._extrapolator.range}")
            return self._extrapolator.extrapolate(x, 0, nx - 2, self._x[nx - 1])
        else:
            return linear1d(self._x[index], self._x[index + 1], self._f[index], self._f[index + 1], x)

    @property
    def domain(self):
        return np.min(self._x), np.max(self._x), np.min(self._f), np.max(self._f)


cdef class Interpolate1DCubic(Interpolate1D):
    """
    Cubic interpolation of 1D function
    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    :param Extrapolator1D extrapolator: extrapolator object
    """
    def __init__(self, object x, object f, Extrapolator1D extrapolator):
        self._x = np.array(x, dtype=np.float64)
        self._f = np.array(f, dtype=np.float64)
        self._extrapolator = extrapolator
        raise NotImplementedError(f"{self.__class__} not implemented")

    cdef double evaluate(self, double x) except? -1e999:
        raise NotImplementedError(f"{self.__class__} not implemented")

    @property
    def domain(self):
        raise NotImplementedError(f"{self.__class__} not implemented")


cdef class Extrapolator1D:
    """
    Base class for Function1D extrapolators.

    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    :param double extrapolation_range: Range covered by the extrapolator.
    Padded symmetrically to both ends of the input.
    """
    def __init__(self, ndarray x, ndarray f, double extrapolation_range):
        self._range = extrapolation_range
        self._x = x
        self._f = f

    @property
    def range(self):
        """
        Range covered either by the original x input or by the extrapolator
        """
        min_range = self._x[0] - self._range
        max_range = self._x[self._x.shape[0] - 1] + self._range
        return min_range, max_range

    cdef double extrapolate(self, double px, int order, int index, double rx) except? -1e999:
        raise NotImplementedError(f"{self.__class__} not implemented")

cdef class ExtrapolatorNone(Extrapolator1D):
    """
    Extrapolator that does nothing.
    """
    cdef double extrapolate(self, double px, int order, int index, double rx) except? -1e999:
        raise ValueError(f"Extrapolation not available. Interpolate within function range {np.min(self._x)}-{np.max(self._x)}")

cdef class Extrapolator1DNearest(Extrapolator1D):
    """
    Extrapolator that returns nearest input value
    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    :param double extrapolation_range: Range covered by the extrapolator.
    """
    def __init__(self, ndarray x, ndarray f, double extrapolation_range):
        super(Extrapolator1DNearest, self).__init__(x, f, extrapolation_range)

    cdef double extrapolate(self, double px, int order, int index, double rx) except? -1e999:
        cdef int nx = self._x.shape[0]

        if px < self._x[0]:
            return self._f[0]
        elif px > self._x[nx-1]:
            return self._f[nx-1]

cdef class Extrapolator1DLinear(Extrapolator1D):
    """
    Extrapolator that extrapolates linearly
    :param object x: 1D array-like object of real values.
    :param object f: 1D array-like object of real values.
    :param double extrapolation_range: Range covered by the extrapolator.
    """
    def __init__(self, ndarray x, ndarray f, double extrapolation_range):
        super(Extrapolator1DLinear, self).__init__(x, f, extrapolation_range)

    cdef double extrapolate(self, double px, int order, int index, double rx) except? -1e999:
        return lerp(self._x[index], self._x[index + 1], self._f[index], self._f[index + 1], px)
