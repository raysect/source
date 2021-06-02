import unittest
import numpy as np
from raysect.core.math.function.float.function1d.interpolate import Interpolate1D


X_LOWER = 0.0
X_UPPER = 1.0
X_EXTRAP_DELTA_MAX = 0.08
X_EXTRAP_DELTA_MIN = 0.04

NB_X = 10
NB_XSAMPLES = 30


class TestInterpolators1D(unittest.TestCase):
    def setUp(self) -> None:

        # self.data is a precalculated input values for testing. It's the result of applying function f on self.x
        # as in self.data = f(self.x), where self.x is linearly spaced between X_LOWER and X_UPPER

        #: x values used to obtain self.data
        self.x = np.linspace(X_LOWER, X_UPPER, NB_X)

        #: data array from a function sampled on self.x. dtype should be np.float64
        self.data: np.array = None

        #: precalculated result of sampling self.data on self.xsamples
        #   should be set in interpolator specific setup function.
        self.precalc_interpolation = None

        #: x values on which self.precalc_interpolation was samples on
        self.xsamples = np.linspace(X_LOWER, X_UPPER, NB_XSAMPLES)

        #: array holding precalculated linear extrapolation data
        #   to be set in interpolator specific setup_ method
        self.precalc_extrapolation_linear: np.array = None

        #: array holding precalculated cubic extrapolation data
        #   to be set in interpolator specific setup_ method
        self.precalc_extrapolation_cubic: np.array = None

        #: x values on which self.precalc_extrapolation_ arrays were sampled on
        self.xsamples_extrap = np.array(
            [
                X_LOWER - X_EXTRAP_DELTA_MAX,
                X_LOWER - X_EXTRAP_DELTA_MIN,
                X_UPPER + X_EXTRAP_DELTA_MIN,
                X_UPPER + X_EXTRAP_DELTA_MAX,
            ],
            dtype=np.float64,
        )

        #: the interpolator object that is being tested. Set in setup_ method
        self.interpolator: Interpolate1D = None


    def setup_linear(self, extrapolator_type: str, extrapolation_range: float) -> None:
        """
        Sets precalculated values for linear interpolator.
        Called in every test method that addresses linear interpolation.

        Once executed, self.precalc_NNN members variables will contain
        precalculated extrapolated / interpolated data. self.interpolator
        will hold Interpolate1D object that is being tested.

        :param extrapolator_type: type of extrapolator 'none', 'linear' or 'cubic'
        :param extrapolation_range: padding around interpolation range where extrapolation is possible
        """

        # set precalculated expected interpolation results
        # this is the result of sampling self.data on self.xsamples
        self.precalc_interpolation = None

        # set precalculated expected extrapolation results
        # this is the result of nearest neighbour extrapolation on self.xsamples_extrap
        self.precalc_extrapolation_nearest = None

        # set precalculated expected extrapolation results
        # this is the result of linear extrapolation on self.xsamples_extrap
        self.precalc_extrapolation_linear = None

        # set interpolator
        self.interpolator = Interpolate1D(self.x, self.data, "linear", extrapolator_type, extrapolation_range)

    def setup_cubic(self, extrapolator_type: str, extrapolation_range: float):
        """
        Sets precalculated values for cubic interpolator.
        Called in every test method that addresses cubic interpolation.

        Once executed, self.precalc_NNN members variables will contain
        precalculated extrapolated / interpolated data. self.interpolator
        will hold Interpolate1D object that is being tested.

        :param extrapolator_type: type of extrapolator 'none', 'linear' or 'cubic'
        :param extrapolation_range: padding around interpolation range where extrapolation is possible
        """

        # set precalculated expected interpolation results
        # this is the result of sampling self.data on self.xsamples
        self.precalc_interpolation = None

        # set precalculated expected extrapolation results
        # this is the result of nearest neighbour extrapolation on self.xsamples_extrap
        self.precalc_extrapolation_linear = None

        # set precalculated expected extrapolation results
        # this is the result of linear extrapolation on self.xsamples_extrap
        self.precalc_extrapolation_cubic = None

        # set interpolator
        self.interpolator = Interpolate1D(self.x, self.data, "linear", extrapolator_type, extrapolation_range)

    def test_linear_interpolation(self):
        self.setup_linear("none", 0.0)

    def test_linear_interpolation_extrapolators(self):
        self.setup_linear("nearest", 0.0)
        # test linear interpolation with 'nearest' extrapolator here

        self.setup_linear("linear", 0.0)
        # test linear interpolation with 'linear' extrapolator here

    def test_cubic_interpolation_extrapolators(self):
        self.setup_cubic("nearest", 0.0)
        # test cubic interpolation with 'nearest' extrapolator here

        self.setup_cubic("linear", 0.0)
        # test cubic interpolation with 'linear' extrapolator here
