# cython: language_level=3

# Copyright (c) 2014-2017, Dr Alex Meakins, Raysect Project
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

from raysect.optical.spectralfunction import InterpolatedSF

yellow = InterpolatedSF([100, 569.6, 570.6, 575.6, 576.6, 800], [0, 0, 1, 1, 0, 0], normalise=True)

orange = InterpolatedSF([100, 581.1, 582.1, 587.1, 588.1, 800], [0, 0, 1, 1, 0, 0], normalise=True)

red_orange = InterpolatedSF([100, 592.6, 593.6, 598.6, 599.6, 800], [0, 0, 1, 1, 0, 0], normalise=True)

red = InterpolatedSF([100, 627.3, 628.3, 633.3, 634.3, 800], [0, 0, 1, 1, 0, 0], normalise=True)

maroon = InterpolatedSF([100, 673.4, 674.4, 679.4, 680.4, 800], [0, 0, 1, 1, 0, 0], normalise=True)

purple = InterpolatedSF([100, 419.6, 420.6, 425.6, 426.6, 800], [0, 0, 1, 1, 0, 0], normalise=True)

blue = InterpolatedSF([100, 465.7, 466.7, 471.7, 472.7, 800], [0, 0, 1, 1, 0, 0], normalise=True)

light_blue = InterpolatedSF([100, 475.3, 476.3, 481.3, 482.3, 800], [0, 0, 1, 1, 0, 0], normalise=True)

cyan = InterpolatedSF([100, 488.8, 489.8, 494.8, 495.8, 800], [0, 0, 1, 1, 0, 0], normalise=True)

green = InterpolatedSF([100, 534.96, 535.96, 540.96, 541.96, 800], [0, 0, 1, 1, 0, 0], normalise=True)
