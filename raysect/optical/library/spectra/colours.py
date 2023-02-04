# cython: language_level=3

# Copyright (c) 2014-2023, Dr Alex Meakins, Raysect Project
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


def _top_hat_spectralfn(center, width, rolloff):

    start = 0
    end = 1000
    half_width = width / 2
    top_min = center - half_width
    top_max = center + half_width
    base_min = top_min - rolloff
    base_max = top_max + rolloff

    return InterpolatedSF([start, base_min, top_min, top_max, base_max, end], [0, 0, 1, 1, 0, 0], normalise=True)


purple = _top_hat_spectralfn(423.1, 5.0, 1.0)
blue = _top_hat_spectralfn(469.2, 5.0, 1.0)
light_blue = _top_hat_spectralfn(478.8, 5.0, 1.0)
cyan = _top_hat_spectralfn(492.3, 5.0, 1.0)
green = _top_hat_spectralfn(538.5, 5.0, 1.0)
yellow = _top_hat_spectralfn(571.1, 5.0, 1.0)
orange = _top_hat_spectralfn(584.6, 5.0, 1.0)
red_orange = _top_hat_spectralfn(596.1, 5.0, 1.0)
red = _top_hat_spectralfn(630.8, 5.0, 1.0)
maroon = _top_hat_spectralfn(676.9, 5.0, 1.0)
