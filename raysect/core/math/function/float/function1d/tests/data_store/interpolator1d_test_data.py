import numpy as np


class TestInterpolatorLoadValues:
    """
    Base class for loading values for a 1D sin function test.

    Storage for interpolation and extrapolation data to be test against.
    These data are saved to 12 significant figures. self.data is generated by applying the sin function to an
    (NB_X = 10) 1D data set, which is used as the spline knots. The precalc_interpolation
    is setup for cubic, and linear interpolation are compared to functions equivalent functions from scipy
    version 1.6.3. In 1D, all extrapolation values are independent of the interpolator types.
    """
    def __init__(self):
        # Define in setup_cubic or setup_linear.
        self.precalc_interpolation = None

        #: Array holding precalculated nearest neighbour extrapolation data.
        self.precalc_extrapolation_nearest: np.array = None

        #: Array holding precalculated linear extrapolation data.
        self.precalc_extrapolation_linear: np.array = None

        #: Array holding precalculated quadratic extrapolation data.
        self.precalc_extrapolation_quadratic: np.array = None


class TestInterpolatorLoadNormalValues(TestInterpolatorLoadValues):
    """
    Loading values for the original np.sin(x) tests.

    For description of data storage, see TestInterpolatorLoadValues.
    """
    def __init__(self):
        super().__init__()
        #: data array from a function sampled on self.x. dtype should be np.float64.
        self.data: np.array = np.array(
            [0.000000000000E+00, 1.108826285100E-01, 2.203977434561E-01, 3.271946967962E-01, 4.299563635284E-01,
             5.274153857719E-01, 6.183698030697E-01, 7.016978761467E-01, 7.763719213007E-01, 8.414709848079E-01],
            dtype=np.float64
        )

        #: array holding precalculated nearest neighbour extrapolation data.
        self.precalc_extrapolation_nearest: np.array = np.array(
            [0.000000000000E+00, 0.000000000000E+00, 8.414709848079E-01, 8.414709848079E-01], dtype=np.float64
        )

        #: array holding precalculated linear extrapolation data.
        self.precalc_extrapolation_linear: np.array = np.array(
            [-7.983549252717E-02, -3.991774626358E-02, 8.649066476705E-01, 8.883423105331E-01], dtype=np.float64
        )
        #: array holding precalculated quadratic extrapolation data.
        self.precalc_extrapolation_quadratic: np.array = np.array(
            [-8.001272228503E-02, -3.996205370305E-02, 8.645964182651E-01, 8.871013929117E-01], dtype=np.float64
        )

    def setup_cubic(self):
        self.precalc_interpolation = np.array(
            [0.000000000000E+00, 3.445726766897E-02, 6.892361882629E-02, 1.032764263792E-01,
             1.374723672762E-01, 1.715660220877E-01, 2.054427374305E-01, 2.390211362762E-01,
             2.723736467104E-01, 3.054074932289E-01, 3.380184236890E-01, 3.702697865700E-01,
             4.021030314403E-01, 4.334059817503E-01, 4.642087324270E-01, 4.944967727547E-01,
             5.241623210482E-01, 5.531868543036E-01, 5.816021511759E-01, 6.093090034050E-01,
             6.362532436612E-01, 6.624890560875E-01, 6.879357356812E-01, 7.125198048495E-01,
             7.362936780787E-01, 7.592034615909E-01, 7.811477131101E-01, 8.017642725164E-01,
             8.215858285263E-01, 8.414709848079E-01], dtype=np.float64
        )

    def setup_linear(self):
        self.precalc_interpolation = np.array(
            [0.000000000000E+00, 3.441185022723E-02, 6.882370045445E-02, 1.032355506817E-01,
             1.373173114280E-01, 1.713047608940E-01, 2.052922103601E-01, 2.388110112734E-01,
             2.719548933444E-01, 3.050987754155E-01, 3.378252140443E-01, 3.697167657888E-01,
             4.016083175333E-01, 4.333170194678E-01, 4.635629229227E-01, 4.938088263776E-01,
             5.240547298324E-01, 5.525062595092E-01, 5.807334924637E-01, 6.089607254182E-01,
             6.356100940512E-01, 6.614705305234E-01, 6.873309669955E-01, 7.119977444438E-01,
             7.351724481123E-01, 7.583471517807E-01, 7.808615118874E-01, 8.010646695275E-01,
             8.212678271677E-01, 8.414709848079E-01], dtype=np.float64
        )


class TestInterpolatorLoadBigValues(TestInterpolatorLoadValues):
    """
    Loading big values (10^20 times the original) instead of the original np.sin(x).

    For description of data storage, see TestInterpolatorLoadValues.
    """
    def __init__(self):
        super().__init__()
        #: data array from a function sampled on self.x. dtype should be np.float64.
        # self.data: np.array = np.sin(self.x).
        self.data: np.array = np.array(
            [0.000000000000E+00, 1.108826285100E+19, 2.203977434561E+19, 3.271946967962E+19,
             4.299563635284E+19, 5.274153857719E+19, 6.183698030697E+19, 7.016978761467E+19,
             7.763719213007E+19, 8.414709848079E+19], dtype=np.float64
        )
        #: precalculated result of the function used to calculate self.data on self.xsamples.
        #: array holding precalculated nearest neighbour extrapolation data.
        self.precalc_extrapolation_nearest: np.array = np.array(
            [0.000000000000e+00, 0.000000000000e+00, 8.414709848079e+19, 8.414709848079e+19], dtype=np.float64
        )

        #: array holding precalculated linear extrapolation data.
        self.precalc_extrapolation_linear: np.array = np.array(
            [-7.983549252717e+18, -3.991774626358e+18,  8.649066476705e+19,  8.883423105331e+19], dtype=np.float64
        )

        #: array holding precalculated quadratic extrapolation data.
        self.precalc_extrapolation_quadratic: np.array = np.array(
            [-8.001272228503E+18, -3.996205370305E+18, 8.645964182651E+19, 8.871013929117E+19], dtype=np.float64
        )

    def setup_cubic(self):
        self.precalc_interpolation = np.array(
            [0.000000000000e+00, 3.445726766897e+18, 6.892361882629e+18, 1.032764263792e+19,
             1.374723672762e+19, 1.715660220877e+19, 2.054427374305e+19, 2.390211362762e+19,
             2.723736467104e+19, 3.054074932289e+19, 3.380184236890e+19, 3.702697865700e+19,
             4.021030314403e+19, 4.334059817503e+19, 4.642087324270e+19, 4.944967727547e+19,
             5.241623210482e+19, 5.531868543036e+19, 5.816021511759e+19, 6.093090034050e+19,
             6.362532436612e+19, 6.624890560875e+19, 6.879357356812e+19, 7.125198048495e+19,
             7.362936780787e+19, 7.592034615909e+19, 7.811477131101e+19, 8.017642725164e+19,
             8.215858285263e+19, 8.414709848079e+19], dtype=np.float64)

    def setup_linear(self):
        self.precalc_interpolation = np.array(
            [0.000000000000e+00, 3.441185022723e+18, 6.882370045445e+18, 1.032355506817e+19,
             1.373173114280e+19, 1.713047608940e+19, 2.052922103601e+19, 2.388110112734e+19,
             2.719548933444e+19, 3.050987754155e+19, 3.378252140443e+19, 3.697167657888e+19,
             4.016083175333e+19, 4.333170194678e+19, 4.635629229227e+19, 4.938088263776e+19,
             5.240547298324e+19, 5.525062595092e+19, 5.807334924637e+19, 6.089607254182e+19,
             6.356100940512e+19, 6.614705305234e+19, 6.873309669955e+19, 7.119977444438e+19,
             7.351724481123e+19, 7.583471517807e+19, 7.808615118874e+19, 8.010646695275e+19,
             8.212678271677e+19, 8.414709848079e+19], dtype=np.float64)


class TestInterpolatorLoadSmallValues(TestInterpolatorLoadValues):
    """
    Loading small values (10^-20 times the original) instead of the original np.sin(x)

    For description of data storage, see TestInterpolatorLoadValues.
    """
    def __init__(self):
        super().__init__()
        #: data array from a function sampled on self.x. dtype should be np.float64.
        # self.data: np.array = np.sin(self.x).
        self.data: np.array = np.array(
            [0.000000000000E+00, 1.108826285100E-21, 2.203977434561E-21, 3.271946967962E-21,
             4.299563635284E-21, 5.274153857719E-21, 6.183698030697E-21, 7.016978761467E-21,
             7.763719213007E-21, 8.414709848079E-21], dtype=np.float64
        )

        #: precalculated result of the function used to calculate self.data on self.xsamples.
        #: array holding precalculated nearest neighbour extrapolation data.
        self.precalc_extrapolation_nearest: np.array = np.array(
            [0.000000000000e+00, 0.000000000000e+00, 8.414709848079e-21, 8.414709848079e-21], dtype=np.float64
        )

        #: array holding precalculated linear extrapolation data.
        self.precalc_extrapolation_linear: np.array = np.array(
            [-7.983549252717e-22, -3.991774626358e-22,  8.649066476705e-21,  8.883423105331e-21], dtype=np.float64
        )

        #: array holding precalculated quadratic extrapolation data.
        self.precalc_extrapolation_quadratic: np.array = np.array(
            [-8.001272228503E-22, -3.996205370305E-22, 8.645964182651E-21, 8.871013929117E-21], dtype=np.float64
        )

    def setup_cubic(self):
        self.precalc_interpolation = np.array(
            [0.000000000000e+00, 3.445726766897e-22, 6.892361882629e-22, 1.032764263792e-21,
             1.374723672762e-21, 1.715660220877e-21, 2.054427374305e-21, 2.390211362762e-21,
             2.723736467104e-21, 3.054074932289e-21, 3.380184236890e-21, 3.702697865700e-21,
             4.021030314403e-21, 4.334059817503e-21, 4.642087324270e-21, 4.944967727547e-21,
             5.241623210482e-21, 5.531868543036e-21, 5.816021511759e-21, 6.093090034050e-21,
             6.362532436612e-21, 6.624890560875e-21, 6.879357356812e-21, 7.125198048495e-21,
             7.362936780787e-21, 7.592034615909e-21, 7.811477131101e-21, 8.017642725164e-21,
             8.215858285263e-21, 8.414709848079e-21], dtype=np.float64
        )

    def setup_linear(self):
        self.precalc_interpolation = np.array(
            [0.000000000000e+00, 3.441185022723e-22, 6.882370045445e-22, 1.032355506817e-21,
             1.373173114280e-21, 1.713047608940e-21, 2.052922103601e-21, 2.388110112734e-21,
             2.719548933444e-21, 3.050987754155e-21, 3.378252140443e-21, 3.697167657888e-21,
             4.016083175333e-21, 4.333170194678e-21, 4.635629229227e-21, 4.938088263776e-21,
             5.240547298324e-21, 5.525062595092e-21, 5.807334924637e-21, 6.089607254182e-21,
             6.356100940512e-21, 6.614705305234e-21, 6.873309669955e-21, 7.119977444438e-21,
             7.351724481123e-21, 7.583471517807e-21, 7.808615118874e-21, 8.010646695275e-21,
             8.212678271677e-21, 8.414709848079e-21], dtype=np.float64
        )
