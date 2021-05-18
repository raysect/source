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
        raise NotImplementedError(f"{self.__class__} not implemented")

    cdef double evaluate(self, double px, int idx) except? -1e999:
        raise NotImplementedError(f"{self.__class__} not implemented")

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
        elif px > self._x[self._last_index]:
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
