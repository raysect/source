
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
Unit tests for the Interpolator1DCubic class from within Interpolate1D,
including interaction with Extrapolator1DLinear and Extrapolator1DNearest.
"""
import unittest
import numpy as np
from raysect.core.math.function.float.function2d.interpolate.interpolator2dgrid import Interpolator2DGrid, \
    id_to_extrapolator, id_to_interpolator
from raysect.core.math.function.float.function2d.interpolate.tests.scipts.generate_2d_splines import X_LOWER, X_UPPER, \
    NB_XSAMPLES, NB_X, X_EXTRAP_DELTA_MAX, X_EXTRAP_DELTA_MIN, PRECISION, BIG_VALUE_FACTOR, SMALL_VALUE_FACTOR,\
    Y_LOWER, Y_UPPER, NB_YSAMPLES, NB_Y, Y_EXTRAP_DELTA_MAX, Y_EXTRAP_DELTA_MIN, EXTRAPOLATION_RANGE, \
    get_extrapolation_input_values


class TestInterpolatorLoadValues:
    def __init__(self):
        # Define in setup_cubic or setup_linear
        self.precalc_interpolation = None


class TestInterpolatorLoadNormalValues(TestInterpolatorLoadValues):
    """
    Loading normal sized values for a 2D sinc function test.

    These data are saved to 12 significant figures.
    """
    def __init__(self):
        super().__init__()
        #: data array from a function sampled on self.x. dtype should be np.float64
        # self.data: np.array = np.sin(self.x)
        self.data: np.array = np.array(
            [[7.049456954407E-02, -5.031133752816E-03, -8.474851229653E-02, -7.975908114097E-02,
              -4.876973734940E-02, -4.876973734940E-02, -7.975908114097E-02, -8.474851229653E-02,
              -5.031133752816E-03, 7.049456954407E-02],
             [-5.031133752816E-03, -9.121921863446E-02, -9.251264987298E-04, 1.052139178127E-01,
              1.283205555674E-01, 1.283205555674E-01, 1.052139178127E-01, -9.251264987298E-04,
              -9.121921863446E-02, -5.031133752816E-03],
             [-8.474851229653E-02, -9.251264987298E-04, 1.283205555674E-01, 1.734970013481E-02,
              -1.140407180451E-01, -1.140407180451E-01, 1.734970013481E-02, 1.283205555674E-01,
              -9.251264987298E-04, -8.474851229653E-02],
             [-7.975908114097E-02, 1.052139178127E-01, 1.734970013481E-02, -2.145503300375E-01,
              -9.241435356589E-02, -9.241435356589E-02, -2.145503300375E-01, 1.734970013480E-02,
              1.052139178127E-01, -7.975908114097E-02],
             [-4.876973734940E-02, 1.283205555674E-01, -1.140407180451E-01, -9.241435356589E-02,
              6.446759109720E-01, 6.446759109720E-01, -9.241435356589E-02, -1.140407180451E-01,
              1.283205555674E-01, -4.876973734940E-02],
             [-4.876973734940E-02, 1.283205555674E-01, -1.140407180451E-01, -9.241435356589E-02,
              6.446759109720E-01, 6.446759109720E-01, -9.241435356589E-02, -1.140407180451E-01,
              1.283205555674E-01, -4.876973734940E-02],
             [-7.975908114097E-02, 1.052139178127E-01, 1.734970013481E-02, -2.145503300375E-01,
              -9.241435356589E-02, -9.241435356589E-02, -2.145503300375E-01, 1.734970013480E-02,
              1.052139178127E-01, -7.975908114097E-02],
             [-8.474851229653E-02, -9.251264987298E-04, 1.283205555674E-01, 1.734970013480E-02,
              -1.140407180451E-01, -1.140407180451E-01, 1.734970013480E-02, 1.283205555674E-01,
              -9.251264987296E-04, -8.474851229653E-02],
             [-5.031133752816E-03, -9.121921863446E-02, -9.251264987298E-04, 1.052139178127E-01,
              1.283205555674E-01, 1.283205555674E-01, 1.052139178127E-01, -9.251264987296E-04,
              -9.121921863446E-02, -5.031133752816E-03],
             [7.049456954407E-02, -5.031133752816E-03, -8.474851229653E-02, -7.975908114097E-02,
              -4.876973734940E-02, -4.876973734940E-02, -7.975908114097E-02, -8.474851229653E-02,
              -5.031133752816E-03, 7.049456954407E-02]],
            dtype=np.float64
        )

        #: array holding precalculated nearest neighbour extrapolation data
        self.precalc_extrapolation_nearest: np.array = np.array(
            [], dtype=np.float64
        )

        #: array holding precalculated linear extrapolation data
        self.precalc_extrapolation_linear: np.array = np.array(
            [], dtype=np.float64
        )
        #: array holding precalculated quadratic extrapolation data
        self.precalc_extrapolation_quadratic: np.array = np.array(
            [], dtype=np.float64
        )

    def setup_cubic(self):
        self.precalc_interpolation = np.array(
            [], dtype=np.float64
        )

    def setup_linear(self):
        self.precalc_interpolation = np.array(
            [[7.049456954407E-02, 1.385029207141E-02, -4.488982302467E-02, -8.350115450764E-02,
              -7.975908114097E-02, -5.651707329729E-02, -4.876973734940E-02, -5.651707329729E-02,
              -7.975908114097E-02, -8.350115450764E-02, -4.488982302467E-02, 1.385029207141E-02],
             [1.385029207141E-02, -4.879157504269E-02, -4.577658518112E-02, -1.668062692568E-03,
              5.897066807427E-02, 7.777865377223E-02, 8.404798233821E-02, 7.777865377223E-02,
              5.897066807427E-02, -1.668062692568E-03, -4.577658518112E-02, -4.879157504269E-02],
             [-4.488982302467E-02, -4.577658518112E-02, 8.812770983874E-03, 6.309373814419E-02,
              6.128180897374E-02, 2.067539131431E-02, 7.139918761166E-03, 2.067539131431E-02,
              6.128180897374E-02, 6.309373814419E-02, 8.812770983874E-03, -4.577658518112E-02],
             [-8.350115450764E-02, -1.668062692568E-03, 6.309373814419E-02, 6.527705442988E-02,
              -4.062530740828E-02, -9.163192204604E-02, -1.086341269253E-01, -9.163192204604E-02,
              -4.062530740828E-02, 6.527705442988E-02, 6.309373814419E-02, -1.668062692568E-03],
             [-7.975908114097E-02, 5.897066807427E-02, 6.128180897374E-02, -4.062530740828E-02,
              -2.145503300375E-01, -1.229483476838E-01, -9.241435356589E-02, -1.229483476838E-01,
              -2.145503300375E-01, -4.062530740828E-02, 6.128180897374E-02, 5.897066807427E-02],
             [-5.651707329729E-02, 7.777865377223E-02, 2.067539131431E-02, -9.163192204604E-02,
              -1.229483476838E-01, 3.145654217072E-01, 4.604033448375E-01, 3.145654217072E-01,
              -1.229483476838E-01, -9.163192204604E-02, 2.067539131431E-02, 7.777865377223E-02],
             [-4.876973734940E-02, 8.404798233821E-02, 7.139918761166E-03, -1.086341269253E-01,
              -9.241435356589E-02, 4.604033448375E-01, 6.446759109720E-01, 4.604033448375E-01,
              -9.241435356589E-02, -1.086341269253E-01, 7.139918761166E-03, 8.404798233821E-02],
             [-5.651707329729E-02, 7.777865377223E-02, 2.067539131431E-02, -9.163192204604E-02,
              -1.229483476838E-01, 3.145654217072E-01, 4.604033448375E-01, 3.145654217072E-01,
              -1.229483476838E-01, -9.163192204604E-02, 2.067539131431E-02, 7.777865377223E-02],
             [-7.975908114097E-02, 5.897066807427E-02, 6.128180897374E-02, -4.062530740828E-02,
              -2.145503300375E-01, -1.229483476838E-01, -9.241435356589E-02, -1.229483476838E-01,
              -2.145503300375E-01, -4.062530740828E-02, 6.128180897374E-02, 5.897066807427E-02],
             [-8.350115450764E-02, -1.668062692568E-03, 6.309373814419E-02, 6.527705442988E-02,
              -4.062530740828E-02, -9.163192204604E-02, -1.086341269253E-01, -9.163192204604E-02,
              -4.062530740828E-02, 6.527705442988E-02, 6.309373814419E-02, -1.668062692568E-03],
             [-4.488982302467E-02, -4.577658518112E-02, 8.812770983874E-03, 6.309373814419E-02,
              6.128180897374E-02, 2.067539131431E-02, 7.139918761166E-03, 2.067539131431E-02,
              6.128180897374E-02, 6.309373814419E-02, 8.812770983874E-03, -4.577658518112E-02],
             [1.385029207141E-02, -4.879157504269E-02, -4.577658518112E-02, -1.668062692568E-03,
              5.897066807427E-02, 7.777865377223E-02, 8.404798233821E-02, 7.777865377223E-02,
              5.897066807427E-02, -1.668062692568E-03, -4.577658518112E-02, -4.879157504269E-02]]
            , dtype=np.float64
        )


class TestInterpolatorLoadBigValues(TestInterpolatorLoadValues):
    """
    Loading big values (10^20 times the original) instead of the original np.sin(x).

    These data are saved to 12 significant figures.
    """
    def __init__(self):
        super().__init__()
        #: data array from a function sampled on self.x. dtype should be np.float64
        # self.data: np.array = np.sin(self.x)
        self.data: np.array = np.array(
            [[7.049456954407E+18, -5.031133752816E+17, -8.474851229653E+18, -7.975908114097E+18,
              -4.876973734940E+18, -4.876973734940E+18, -7.975908114097E+18, -8.474851229653E+18,
              -5.031133752816E+17, 7.049456954407E+18],
             [-5.031133752816E+17, -9.121921863446E+18, -9.251264987298E+16, 1.052139178127E+19,
              1.283205555674E+19, 1.283205555674E+19, 1.052139178127E+19, -9.251264987298E+16,
              -9.121921863446E+18, -5.031133752816E+17],
             [-8.474851229653E+18, -9.251264987298E+16, 1.283205555674E+19, 1.734970013481E+18,
              -1.140407180451E+19, -1.140407180451E+19, 1.734970013481E+18, 1.283205555674E+19,
              -9.251264987298E+16, -8.474851229653E+18],
             [-7.975908114097E+18, 1.052139178127E+19, 1.734970013481E+18, -2.145503300375E+19,
              -9.241435356589E+18, -9.241435356589E+18, -2.145503300375E+19, 1.734970013480E+18,
              1.052139178127E+19, -7.975908114097E+18],
             [-4.876973734940E+18, 1.283205555674E+19, -1.140407180451E+19, -9.241435356589E+18,
              6.446759109720E+19, 6.446759109720E+19, -9.241435356589E+18, -1.140407180451E+19,
              1.283205555674E+19, -4.876973734940E+18],
             [-4.876973734940E+18, 1.283205555674E+19, -1.140407180451E+19, -9.241435356589E+18,
              6.446759109720E+19, 6.446759109720E+19, -9.241435356589E+18, -1.140407180451E+19,
              1.283205555674E+19, -4.876973734940E+18],
             [-7.975908114097E+18, 1.052139178127E+19, 1.734970013481E+18, -2.145503300375E+19,
              -9.241435356589E+18, -9.241435356589E+18, -2.145503300375E+19, 1.734970013480E+18,
              1.052139178127E+19, -7.975908114097E+18],
             [-8.474851229653E+18, -9.251264987298E+16, 1.283205555674E+19, 1.734970013480E+18,
              -1.140407180451E+19, -1.140407180451E+19, 1.734970013480E+18, 1.283205555674E+19,
              -9.251264987296E+16, -8.474851229653E+18],
             [-5.031133752816E+17, -9.121921863446E+18, -9.251264987298E+16, 1.052139178127E+19,
              1.283205555674E+19, 1.283205555674E+19, 1.052139178127E+19, -9.251264987296E+16,
              -9.121921863446E+18, -5.031133752816E+17],
             [7.049456954407E+18, -5.031133752816E+17, -8.474851229653E+18, -7.975908114097E+18,
              -4.876973734940E+18, -4.876973734940E+18, -7.975908114097E+18, -8.474851229653E+18,
              -5.031133752816E+17, 7.049456954407E+18]]
            , dtype=np.float64
        )
        #: precalculated result of the function used to calculate self.data on self.xsamples
        #: array holding precalculated nearest neighbour extrapolation data
        self.precalc_extrapolation_nearest: np.array = np.array(
            [], dtype=np.float64
        )

        #: array holding precalculated linear extrapolation data
        self.precalc_extrapolation_linear: np.array = np.array(
            [], dtype=np.float64
        )

        #: array holding precalculated quadratic extrapolation data
        self.precalc_extrapolation_quadratic: np.array = np.array(
            [], dtype=np.float64
        )

    def setup_cubic(self):
        self.precalc_interpolation = np.array(
            [], dtype=np.float64)

    def setup_linear(self):
        self.precalc_interpolation = np.array(
            [[7.049456954407E+18, 1.385029207141E+18, -4.488982302467E+18, -8.350115450764E+18,
              -7.975908114097E+18, -5.651707329729E+18, -4.876973734940E+18, -5.651707329729E+18,
              -7.975908114097E+18, -8.350115450764E+18, -4.488982302467E+18, 1.385029207141E+18],
             [1.385029207141E+18, -4.879157504269E+18, -4.577658518112E+18, -1.668062692568E+17,
              5.897066807427E+18, 7.777865377223E+18, 8.404798233821E+18, 7.777865377223E+18,
              5.897066807427E+18, -1.668062692568E+17, -4.577658518112E+18, -4.879157504269E+18],
             [-4.488982302467E+18, -4.577658518112E+18, 8.812770983874E+17, 6.309373814419E+18,
              6.128180897374E+18, 2.067539131431E+18, 7.139918761166E+17, 2.067539131431E+18,
              6.128180897374E+18, 6.309373814419E+18, 8.812770983874E+17, -4.577658518112E+18],
             [-8.350115450764E+18, -1.668062692568E+17, 6.309373814419E+18, 6.527705442988E+18,
              -4.062530740828E+18, -9.163192204604E+18, -1.086341269253E+19, -9.163192204604E+18,
              -4.062530740828E+18, 6.527705442988E+18, 6.309373814419E+18, -1.668062692568E+17],
             [-7.975908114097E+18, 5.897066807427E+18, 6.128180897374E+18, -4.062530740828E+18,
              -2.145503300375E+19, -1.229483476838E+19, -9.241435356589E+18, -1.229483476838E+19,
              -2.145503300375E+19, -4.062530740828E+18, 6.128180897374E+18, 5.897066807427E+18],
             [-5.651707329729E+18, 7.777865377223E+18, 2.067539131431E+18, -9.163192204604E+18,
              -1.229483476838E+19, 3.145654217072E+19, 4.604033448375E+19, 3.145654217072E+19,
              -1.229483476838E+19, -9.163192204604E+18, 2.067539131431E+18, 7.777865377223E+18],
             [-4.876973734940E+18, 8.404798233821E+18, 7.139918761166E+17, -1.086341269253E+19,
              -9.241435356589E+18, 4.604033448375E+19, 6.446759109720E+19, 4.604033448375E+19,
              -9.241435356589E+18, -1.086341269253E+19, 7.139918761166E+17, 8.404798233821E+18],
             [-5.651707329729E+18, 7.777865377223E+18, 2.067539131431E+18, -9.163192204604E+18,
              -1.229483476838E+19, 3.145654217072E+19, 4.604033448375E+19, 3.145654217072E+19,
              -1.229483476838E+19, -9.163192204604E+18, 2.067539131431E+18, 7.777865377223E+18],
             [-7.975908114097E+18, 5.897066807427E+18, 6.128180897374E+18, -4.062530740828E+18,
              -2.145503300375E+19, -1.229483476838E+19, -9.241435356589E+18, -1.229483476838E+19,
              -2.145503300375E+19, -4.062530740828E+18, 6.128180897374E+18, 5.897066807427E+18],
             [-8.350115450764E+18, -1.668062692568E+17, 6.309373814419E+18, 6.527705442988E+18,
              -4.062530740828E+18, -9.163192204604E+18, -1.086341269253E+19, -9.163192204604E+18,
              -4.062530740828E+18, 6.527705442988E+18, 6.309373814419E+18, -1.668062692568E+17],
             [-4.488982302467E+18, -4.577658518112E+18, 8.812770983874E+17, 6.309373814419E+18,
              6.128180897374E+18, 2.067539131431E+18, 7.139918761166E+17, 2.067539131431E+18,
              6.128180897374E+18, 6.309373814419E+18, 8.812770983874E+17, -4.577658518112E+18],
             [1.385029207141E+18, -4.879157504269E+18, -4.577658518112E+18, -1.668062692568E+17,
              5.897066807427E+18, 7.777865377223E+18, 8.404798233821E+18, 7.777865377223E+18,
              5.897066807427E+18, -1.668062692568E+17, -4.577658518112E+18, -4.879157504269E+18]], dtype=np.float64)


class TestInterpolatorLoadSmallValues(TestInterpolatorLoadValues):
    """
    Loading small values (10^-20 times the original) instead of the original np.sin(x)

    These data are saved to 12 significant figures.
    """
    def __init__(self):
        super().__init__()
        #: data array from a function sampled on self.x. dtype should be np.float64
        # self.data: np.array = np.sin(self.x)
        self.data: np.array = np.array(
            [[7.049456954407E-22, -5.031133752816E-23, -8.474851229653E-22, -7.975908114097E-22,
              -4.876973734940E-22, -4.876973734940E-22, -7.975908114097E-22, -8.474851229653E-22,
              -5.031133752816E-23, 7.049456954407E-22],
             [-5.031133752816E-23, -9.121921863446E-22, -9.251264987298E-24, 1.052139178127E-21,
              1.283205555674E-21, 1.283205555674E-21, 1.052139178127E-21, -9.251264987298E-24,
              -9.121921863446E-22, -5.031133752816E-23],
             [-8.474851229653E-22, -9.251264987298E-24, 1.283205555674E-21, 1.734970013481E-22,
              -1.140407180451E-21, -1.140407180451E-21, 1.734970013481E-22, 1.283205555674E-21,
              -9.251264987298E-24, -8.474851229653E-22],
             [-7.975908114097E-22, 1.052139178127E-21, 1.734970013481E-22, -2.145503300375E-21,
              -9.241435356589E-22, -9.241435356589E-22, -2.145503300375E-21, 1.734970013480E-22,
              1.052139178127E-21, -7.975908114097E-22],
             [-4.876973734940E-22, 1.283205555674E-21, -1.140407180451E-21, -9.241435356589E-22,
              6.446759109720E-21, 6.446759109720E-21, -9.241435356589E-22, -1.140407180451E-21,
              1.283205555674E-21, -4.876973734940E-22],
             [-4.876973734940E-22, 1.283205555674E-21, -1.140407180451E-21, -9.241435356589E-22,
              6.446759109720E-21, 6.446759109720E-21, -9.241435356589E-22, -1.140407180451E-21,
              1.283205555674E-21, -4.876973734940E-22],
             [-7.975908114097E-22, 1.052139178127E-21, 1.734970013481E-22, -2.145503300375E-21,
              -9.241435356589E-22, -9.241435356589E-22, -2.145503300375E-21, 1.734970013480E-22,
              1.052139178127E-21, -7.975908114097E-22],
             [-8.474851229653E-22, -9.251264987298E-24, 1.283205555674E-21, 1.734970013480E-22,
              -1.140407180451E-21, -1.140407180451E-21, 1.734970013480E-22, 1.283205555674E-21,
              -9.251264987296E-24, -8.474851229653E-22],
             [-5.031133752816E-23, -9.121921863446E-22, -9.251264987298E-24, 1.052139178127E-21,
              1.283205555674E-21, 1.283205555674E-21, 1.052139178127E-21, -9.251264987296E-24,
              -9.121921863446E-22, -5.031133752816E-23],
             [7.049456954407E-22, -5.031133752816E-23, -8.474851229653E-22, -7.975908114097E-22,
              -4.876973734940E-22, -4.876973734940E-22, -7.975908114097E-22, -8.474851229653E-22,
              -5.031133752816E-23, 7.049456954407E-22]], dtype=np.float64
        )

        #: precalculated result of the function used to calculate self.data on self.xsamples
        # self.precalc_function = np.array()
        #: array holding precalculated nearest neighbour extrapolation data
        self.precalc_extrapolation_nearest: np.array = np.array(
            [], dtype=np.float64
        )

        #: array holding precalculated linear extrapolation data
        self.precalc_extrapolation_linear: np.array = np.array(
            [], dtype=np.float64
        )

        #: array holding precalculated quadratic extrapolation data
        self.precalc_extrapolation_quadratic: np.array = np.array(
            [], dtype=np.float64
        )

    def setup_cubic(self):
        self.precalc_interpolation = np.array(
            [], dtype=np.float64
        )

    def setup_linear(self):
        self.precalc_interpolation = np.array(
            [[7.049456954407E-22, 1.385029207141E-22, -4.488982302467E-22, -8.350115450764E-22,
              -7.975908114097E-22, -5.651707329729E-22, -4.876973734940E-22, -5.651707329729E-22,
              -7.975908114097E-22, -8.350115450764E-22, -4.488982302467E-22, 1.385029207141E-22],
             [1.385029207141E-22, -4.879157504269E-22, -4.577658518112E-22, -1.668062692568E-23,
              5.897066807427E-22, 7.777865377223E-22, 8.404798233821E-22, 7.777865377223E-22,
              5.897066807427E-22, -1.668062692568E-23, -4.577658518112E-22, -4.879157504269E-22],
             [-4.488982302467E-22, -4.577658518112E-22, 8.812770983874E-23, 6.309373814419E-22,
              6.128180897374E-22, 2.067539131431E-22, 7.139918761166E-23, 2.067539131431E-22,
              6.128180897374E-22, 6.309373814419E-22, 8.812770983874E-23, -4.577658518112E-22],
             [-8.350115450764E-22, -1.668062692568E-23, 6.309373814419E-22, 6.527705442988E-22,
              -4.062530740828E-22, -9.163192204604E-22, -1.086341269253E-21, -9.163192204604E-22,
              -4.062530740828E-22, 6.527705442988E-22, 6.309373814419E-22, -1.668062692568E-23],
             [-7.975908114097E-22, 5.897066807427E-22, 6.128180897374E-22, -4.062530740828E-22,
              -2.145503300375E-21, -1.229483476838E-21, -9.241435356589E-22, -1.229483476838E-21,
              -2.145503300375E-21, -4.062530740828E-22, 6.128180897374E-22, 5.897066807427E-22],
             [-5.651707329729E-22, 7.777865377223E-22, 2.067539131431E-22, -9.163192204604E-22,
              -1.229483476838E-21, 3.145654217072E-21, 4.604033448375E-21, 3.145654217072E-21,
              -1.229483476838E-21, -9.163192204604E-22, 2.067539131431E-22, 7.777865377223E-22],
             [-4.876973734940E-22, 8.404798233821E-22, 7.139918761166E-23, -1.086341269253E-21,
              -9.241435356589E-22, 4.604033448375E-21, 6.446759109720E-21, 4.604033448375E-21,
              -9.241435356589E-22, -1.086341269253E-21, 7.139918761166E-23, 8.404798233821E-22],
             [-5.651707329729E-22, 7.777865377223E-22, 2.067539131431E-22, -9.163192204604E-22,
              -1.229483476838E-21, 3.145654217072E-21, 4.604033448375E-21, 3.145654217072E-21,
              -1.229483476838E-21, -9.163192204604E-22, 2.067539131431E-22, 7.777865377223E-22],
             [-7.975908114097E-22, 5.897066807427E-22, 6.128180897374E-22, -4.062530740828E-22,
              -2.145503300375E-21, -1.229483476838E-21, -9.241435356589E-22, -1.229483476838E-21,
              -2.145503300375E-21, -4.062530740828E-22, 6.128180897374E-22, 5.897066807427E-22],
             [-8.350115450764E-22, -1.668062692568E-23, 6.309373814419E-22, 6.527705442988E-22,
              -4.062530740828E-22, -9.163192204604E-22, -1.086341269253E-21, -9.163192204604E-22,
              -4.062530740828E-22, 6.527705442988E-22, 6.309373814419E-22, -1.668062692568E-23],
             [-4.488982302467E-22, -4.577658518112E-22, 8.812770983874E-23, 6.309373814419E-22,
              6.128180897374E-22, 2.067539131431E-22, 7.139918761166E-23, 2.067539131431E-22,
              6.128180897374E-22, 6.309373814419E-22, 8.812770983874E-23, -4.577658518112E-22],
             [1.385029207141E-22, -4.879157504269E-22, -4.577658518112E-22, -1.668062692568E-23,
              5.897066807427E-22, 7.777865377223E-22, 8.404798233821E-22, 7.777865377223E-22,
              5.897066807427E-22, -1.668062692568E-23, -4.577658518112E-22, -4.879157504269E-22]], dtype=np.float64
        )


class TestInterpolators2D(unittest.TestCase):
    def setUp(self) -> None:

        # self.data is a precalculated input values for testing. It's the result of applying function f on self.x
        # as in self.data = f(self.x), where self.x is linearly spaced between X_LOWER and X_UPPER

        #: x and y values used to obtain self.data
        x_in = np.linspace(X_LOWER, X_UPPER, NB_X)
        y_in = np.linspace(Y_LOWER, Y_UPPER, NB_Y)
        x_in_full, y_in_full = np.meshgrid(x_in, y_in)
        self.x = x_in
        self.y = y_in

        self.test_loaded_values = TestInterpolatorLoadNormalValues()
        self.test_loaded_big_values = TestInterpolatorLoadBigValues()
        self.test_loaded_small_values = TestInterpolatorLoadSmallValues()

        #: precalculated result of sampling self.data on self.xsamples
        #   should be set in interpolator specific setup function.
        self.precalc_interpolation = None

        #: x values on which self.precalc_interpolation was samples on
        xsamples = np.linspace(X_LOWER, X_UPPER, NB_XSAMPLES)
        ysamples = np.linspace(Y_LOWER, Y_UPPER, NB_YSAMPLES)
        # Temporary measure for not extrapolating
        self.xsamples = xsamples[:-1]
        self.ysamples = ysamples[:-1]

        #: x values on which self.precalc_extrapolation_ arrays were sampled on
        # Extrapolation x and y values
        self.xsamples_out_of_bounds, self.ysamples_out_of_bounds, self.xsamples_in_bounds, self.ysamples_in_bounds = \
            get_extrapolation_input_values(
                X_LOWER, X_UPPER, Y_LOWER, Y_UPPER, X_EXTRAP_DELTA_MAX, Y_EXTRAP_DELTA_MAX, X_EXTRAP_DELTA_MIN,
                Y_EXTRAP_DELTA_MIN
            )

        #: set precalculated expected extrapolation results  Set in setup_ method
        self.precalc_extrapolation = None

        #: the interpolator object that is being tested. Set in setup_ method
        self.interpolator: Interpolator2DGrid = None

    def setup_linear(
            self, extrapolator_type: str, extrapolation_range: float, big_values: bool, small_values: bool) -> None:
        """
        Sets precalculated values for linear interpolator.
        Called in every test method that addresses linear interpolation.

        Once executed, self.precalc_NNN members variables will contain
        precalculated extrapolated / interpolated data. self.interpolator
        will hold Interpolate1D object that is being tested. Precalculated interpolation using
        scipy.interpolate.interp1d(kind=linear), generated using scipy version 1.6.3

        :param extrapolator_type: type of extrapolator 'none', 'linear' or 'cubic'
        :param extrapolation_range: padding around interpolation range where extrapolation is possible
        :param big_values: For loading and testing big value saved data
        :param small_values: For loading and testing small value saved data
        """

        # set precalculated expected interpolation results  using scipy.interpolate.interp1d(kind=linear)
        # this is the result of sampling self.data on self.xsamples

        if big_values:
            self.value_storage_obj = self.test_loaded_big_values
        elif small_values:
            self.value_storage_obj = self.test_loaded_small_values
        else:
            self.value_storage_obj = self.test_loaded_values

        self.value_storage_obj.setup_linear()
        self.data = self.value_storage_obj.data
        self.precalc_interpolation = self.value_storage_obj.precalc_interpolation
        # set precalculated expected extrapolation results
        # this is the result of the type of extrapolation on self.xsamples_extrap
        self.setup_extrpolation_type(extrapolator_type)

        # set interpolator
        self.interpolator = Interpolator2DGrid(self.x, self.y, self.data, 'linear', extrapolator_type, extrapolation_range)

    def setup_cubic(self, extrapolator_type: str, extrapolation_range: float, big_values: bool, small_values: bool):
        """
        Sets precalculated values for cubic interpolator.
        Called in every test method that addresses cubic interpolation.

        Once executed, self.precalc_NNN members variables will contain
        precalculated extrapolated / interpolated data. self.interpolator
        will hold Interpolate1D object that is being tested. Generated using scipy
        version 1.6.3 scipy.interpolate.CubicHermiteSpline, with input gradients.

        :param extrapolator_type: type of extrapolator 'none', 'linear' or 'cubic'
        :param extrapolation_range: padding around interpolation range where extrapolation is possible
        :param big_values: For loading and testing big value saved data
        :param small_values: For loading and testing small value saved data
        """

        # set precalculated expected interpolation results
        # this is the result of sampling self.data on self.xsamples
        if big_values:
            self.value_storage_obj = self.test_loaded_big_values
        elif small_values:
            self.value_storage_obj = self.test_loaded_small_values
        else:
            self.value_storage_obj = self.test_loaded_values

        self.value_storage_obj.setup_cubic()
        self.data = self.value_storage_obj.data
        self.precalc_interpolation = self.value_storage_obj.precalc_interpolation

        self.setup_extrpolation_type(extrapolator_type)
        # set interpolator
        self.interpolator = Interpolator2DGrid(self.x, self.y, self.data, 'cubic', extrapolator_type, extrapolation_range)

    def setup_extrpolation_type(self, extrapolator_type: str):
        if extrapolator_type == 'linear':
            self.precalc_extrapolation = np.copy(self.value_storage_obj.precalc_extrapolation_linear)
        elif extrapolator_type == 'nearest':
            self.precalc_extrapolation = np.copy(self.value_storage_obj.precalc_extrapolation_nearest)
        elif extrapolator_type == 'none':
            self.precalc_extrapolation = None
        elif extrapolator_type == 'quadratic':
            self.precalc_extrapolation = np.copy(self.value_storage_obj.precalc_extrapolation_quadratic)
        else:
            raise ValueError(
                f'Extrapolation type {extrapolator_type} not found or no test. options are {id_to_extrapolator.keys()}'
            )

    def test_extrapolation_none(self):
        self.setup_linear('none', EXTRAPOLATION_RANGE, big_values=False, small_values=False)
        for i in range(len(self.xsamples_in_bounds)):
            self.assertRaises(ValueError, self.interpolator, **{'x':self.xsamples_in_bounds[i], 'y':self.ysamples_in_bounds[i]})

    def test_linear_interpolation_extrapolators(self):
        for extrapolator_type in id_to_extrapolator.keys():
            self.setup_linear(extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=False)
            if extrapolator_type != 'none':
                if extrapolator_type == 'nearest':
                    gradient_continuity = False
                else:
                    gradient_continuity = True
                self.run_general_extrapolation_tests(gradient_continuity=gradient_continuity)
            self.run_general_interpolation_tests()

        # Tests for big values
        for extrapolator_type in id_to_extrapolator.keys():
            self.setup_linear(extrapolator_type, EXTRAPOLATION_RANGE, big_values=True, small_values=False)
            if extrapolator_type != 'none':
                if extrapolator_type == 'nearest':
                    gradient_continuity = False
                else:
                    gradient_continuity = True
                self.run_general_extrapolation_tests(gradient_continuity=gradient_continuity)
            self.run_general_interpolation_tests()

        # Tests for small values
        for extrapolator_type in id_to_extrapolator.keys():
            self.setup_linear(extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=True)
            if extrapolator_type != 'none':
                if extrapolator_type == 'nearest':
                    gradient_continuity = False
                else:
                    gradient_continuity = True
                self.run_general_extrapolation_tests(gradient_continuity=gradient_continuity)

            self.run_general_interpolation_tests()

    def test_cubic_interpolation_extrapolators(self):
        """
        Testing against scipy.interpolate.CubicHermiteSpline with the same gradient calculations
        """
        test_on = False
        # Temporarily turn off cubic 2D tests
        if test_on:
            for extrapolator_type in id_to_extrapolator.keys():
                self.setup_cubic(extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=False)
                if extrapolator_type != 'none':
                    if extrapolator_type == 'nearest':
                        gradient_continuity = False
                    else:
                        gradient_continuity = True
                    self.run_general_extrapolation_tests(gradient_continuity=gradient_continuity)
                self.run_general_interpolation_tests()

            # Tests for big values
            for extrapolator_type in id_to_extrapolator.keys():
                self.setup_cubic(extrapolator_type, EXTRAPOLATION_RANGE, big_values=True, small_values=False)
                if extrapolator_type != 'none':
                    if extrapolator_type == 'nearest':
                        gradient_continuity = False
                    else:
                        gradient_continuity = True
                    self.run_general_extrapolation_tests(gradient_continuity=gradient_continuity)
                self.run_general_interpolation_tests()

            # Tests for small values
            for extrapolator_type in id_to_extrapolator.keys():
                self.setup_cubic(extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=True)
                if extrapolator_type != 'none':
                    if extrapolator_type == 'nearest':
                        gradient_continuity = False
                    else:
                        gradient_continuity = True
                    self.run_general_extrapolation_tests(gradient_continuity=gradient_continuity)

                self.run_general_interpolation_tests()

    def run_general_extrapolation_tests(self, gradient_continuity=True):
        # Test extrapolator out of range, there should be an error raised
        for i in range(len(self.xsamples_out_of_bounds)):
            self.assertRaises(
                ValueError, self.interpolator, (self.xsamples_out_of_bounds[i], self.ysamples_out_of_bounds[i])
            )

        # Test extrapolation inside extrapolation range matches the predefined values
        for i in range(len(self.xsamples_in_bounds)):
            delta_max = np.abs(self.precalc_extrapolation[i]/np.power(10., PRECISION - 1))
            self.assertAlmostEqual(
                self.interpolator(self.xsamples_in_bounds[i], self.ysamples_in_bounds[i]), self.precalc_extrapolation[i]
                , delta=delta_max
            )

        # Turned off gradient testing for now
        test_on = False
        if test_on:
            # Test gradient continuity between interpolation and extrapolation
            delta_max_lower = np.abs(self.precalc_interpolation[0] / np.power(10., PRECISION - 1))
            delta_max_upper = np.abs(self.precalc_interpolation[-1] / np.power(10., PRECISION - 1))
            if gradient_continuity:
                gradients_lower, gradients_upper = self.interpolator.test_edge_gradients()
                self.assertAlmostEqual(gradients_lower[0], gradients_lower[1], delta=delta_max_lower)
                self.assertAlmostEqual(gradients_upper[0], gradients_upper[1], delta=delta_max_upper)

    def run_general_interpolation_tests(self):
        # Test interpolation against xsample
        for i in range(len(self.xsamples)):
            for j in range(len(self.ysamples)):
                delta_max = np.abs(self.precalc_interpolation[i, j] / np.power(10., PRECISION - 1))
                self.assertAlmostEqual(
                    self.interpolator(self.xsamples[i], self.ysamples[j]), self.precalc_interpolation[i, j],
                    delta=delta_max
                )

    def initialise_tests_on_interpolators(self, x_values, y_values, f_values):
        # Test for all combinations
        for extrapolator_type in id_to_extrapolator.keys():
            for interpolator_type in id_to_interpolator.keys():
                dict_kwargs_interpolators = {
                    'x': x_values, 'y': y_values, 'f': f_values, 'interpolation_type': interpolator_type,
                    'extrapolation_type': extrapolator_type, 'extrapolation_range': 2.0
                }
                self.assertRaises(ValueError, Interpolator2DGrid, **dict_kwargs_interpolators)

    def test_initialisation_errors(self):
        # monotonicity x
        x_wrong = np.copy(self.x)
        x_wrong[0] = self.x[1]
        x_wrong[1] = self.x[0]
        self.initialise_tests_on_interpolators(x_wrong, self.y, self.test_loaded_values.data)

        # monotonicity y
        y_wrong = np.copy(self.y)
        y_wrong[0] = self.y[1]
        y_wrong[1] = self.y[0]
        self.initialise_tests_on_interpolators(self.x, y_wrong, self.test_loaded_values.data)

        # test repeated coordinate x
        x_wrong = np.copy(self.x)
        x_wrong[0] = x_wrong[1]
        self.initialise_tests_on_interpolators(x_wrong, self.y, self.test_loaded_values.data)

        # test repeated coordinate y
        y_wrong = np.copy(self.y)
        y_wrong[0] = y_wrong[1]
        self.initialise_tests_on_interpolators(self.x, y_wrong, self.test_loaded_values.data)

        # mismatch array size between x and data
        x_wrong = np.copy(self.x)
        x_wrong = x_wrong[:-1]
        self.initialise_tests_on_interpolators(x_wrong, self.y, self.test_loaded_values.data)

        # mismatch array size between y and data
        y_wrong = np.copy(self.y)
        y_wrong = y_wrong[:-1]
        self.initialise_tests_on_interpolators(self.x, y_wrong, self.test_loaded_values.data)

        # Todo self._x_mv = x and self._f_mv = f need to be initialised after array checks
        # Test array length 1
        test_on = False
        if test_on:
            # Arrays are too short
            x_wrong = np.copy(self.x)
            y_wrong = np.copy(self.y)
            f_wrong = np.copy(self.data)
            x_wrong = x_wrong[0]
            y_wrong = y_wrong[0]
            f_wrong = f_wrong[0, 0]
            self.initialise_tests_on_interpolators(x_wrong, y_wrong, f_wrong)

        # Incorrect dimension (1D data)
        x_wrong = np.array(np.concatenate((np.copy(self.x)[:, np.newaxis], np.copy(self.x)[:, np.newaxis]), axis=1))
        f_wrong = np.array(np.concatenate((
            np.copy(self.test_loaded_values.data)[:, np.newaxis], np.copy(self.test_loaded_values.data)[:, np.newaxis]
        ), axis=1)
        )
        self.initialise_tests_on_interpolators(x_wrong, y_wrong, f_wrong)
