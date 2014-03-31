# cython: language_level=3

# Copyright (c) 2014, Dr Alex Meakins, Raysect Project
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

# CIE 1931 Standard Colorimetric Observer
CIEXYZ_WAVELENGTH = [380, 385, 390, 395, 400, 405, 410, 415, 420, 425, 430, 435,
                     440, 445, 450, 455, 460, 465, 470, 475, 480, 485, 490, 495,
                     500, 505, 510, 515, 520, 525, 530, 535, 540, 545, 550, 555,
                     560, 565, 570, 575, 580, 585, 590, 595, 600, 605, 610, 615,
                     620, 625, 630, 635, 640, 645, 650, 655, 660, 665, 670, 675,
                     680, 685, 690, 695, 700, 705, 710, 715, 720, 725, 730, 735,
                     740, 745, 750, 755, 760, 765, 770, 775, 780]

CIEXYZ_X = [0.001368, 0.002236, 0.004243, 0.007650, 0.014310, 0.023190, 0.043510,
            0.077630, 0.134380, 0.214770, 0.283900, 0.328500, 0.348280, 0.348060,
            0.336200, 0.318700, 0.290800, 0.251100, 0.195360, 0.142100, 0.095640,
            0.057950, 0.032010, 0.014700, 0.004900, 0.002400, 0.009300, 0.029100,
            0.063270, 0.109600, 0.165500, 0.225750, 0.290400, 0.359700, 0.433450,
            0.512050, 0.594500, 0.678400, 0.762100, 0.842500, 0.916300, 0.978600,
            1.026300, 1.056700, 1.062200, 1.045600, 1.002600, 0.938400, 0.854450,
            0.751400, 0.642400, 0.541900, 0.447900, 0.360800, 0.283500, 0.218700,
            0.164900, 0.121200, 0.087400, 0.063600, 0.046770, 0.032900, 0.022700,
            0.015840, 0.011359, 0.008111, 0.005790, 0.004109, 0.002899, 0.002049,
            0.001440, 0.001000, 0.000690, 0.000476, 0.000332, 0.000235, 0.000166,
            0.000117, 0.000083, 0.000059, 0.000042]


CIEXYZ_Y = [0.000039, 0.000064, 0.000120, 0.000217, 0.000396, 0.000640, 0.001210,
            0.002180, 0.004000, 0.007300, 0.011600, 0.016840, 0.023000, 0.029800,
            0.038000, 0.048000, 0.060000, 0.073900, 0.090980, 0.112600, 0.139020,
            0.169300, 0.208020, 0.258600, 0.323000, 0.407300, 0.503000, 0.608200,
            0.710000, 0.793200, 0.862000, 0.914850, 0.954000, 0.980300, 0.994950,
            1.000000, 0.995000, 0.978600, 0.952000, 0.915400, 0.870000, 0.816300,
            0.757000, 0.694900, 0.631000, 0.566800, 0.503000, 0.441200, 0.381000,
            0.321000, 0.265000, 0.217000, 0.175000, 0.138200, 0.107000, 0.081600,
            0.061000, 0.044580, 0.032000, 0.023200, 0.017000, 0.011920, 0.008210,
            0.005723, 0.004102, 0.002929, 0.002091, 0.001484, 0.001047, 0.000740,
            0.000520, 0.000361, 0.000249, 0.000172, 0.000120, 0.000085, 0.000060,
            0.000042, 0.000030, 0.000021, 0.000015]

CIEXYZ_Z = [0.006450, 0.010550, 0.020050, 0.036210, 0.067850, 0.110200, 0.207400,
            0.371300, 0.645600, 1.039050, 1.385600, 1.622960, 1.747060, 1.782600,
            1.772110, 1.744100, 1.669200, 1.528100, 1.287640, 1.041900, 0.812950,
            0.616200, 0.465180, 0.353300, 0.272000, 0.212300, 0.158200, 0.111700,
            0.078250, 0.057250, 0.042160, 0.029840, 0.020300, 0.013400, 0.008750,
            0.005750, 0.003900, 0.002750, 0.002100, 0.001800, 0.001650, 0.001400,
            0.001100, 0.001000, 0.000800, 0.000600, 0.000340, 0.000240, 0.000190,
            0.000100, 0.000050, 0.000030, 0.000020, 0.000010, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
            0.000000, 0.000000, 0.000000, 0.000000]


def _clamp(v):

    if v < 0:

        return 0

    if v > 1:

        return 1

    return v


def _srgb_transfer_function(v):

    if v <= 0.0031308:

        return 12.92 * v

    else:

        return 1.055 * v**(1 / 2.4) - 0.055


def _srgb_transfer_function_inverse(v):

    if v <= 0.04045:

        return v / 12.92

    else:

        return ((v + 0.055) / 1.055)**2.4


def ciexyz_to_srgb(x, y, z):
    """
    sRGB specified as per IEC 61966-2-1:1999.

    (x, y, z) in range [0, 1]
    (r, g, b) in range [0, 1]
    """

    # convert from CIE XYZ (D65) to sRGB (linear)
    r = 3.2406 * x - 1.5372 * y - 0.4986 * z
    g = -0.9689 * x + 1.8758 * y + 0.0415 * z
    b = 0.0557 * x - 0.2040 * y + 1.0570 * z

    # apply sRGB transfer function
    r = _srgb_transfer_function(r)
    g = _srgb_transfer_function(g)
    b = _srgb_transfer_function(b)

    # restrict to [0, 1]
    r = _clamp(r)
    g = _clamp(g)
    b = _clamp(b)

    return (r, g, b)


def srgb_to_ciexyz(r, g, b):
    """
    sRGB specified as per IEC 61966-2-1:1999.

    (r, g, b) in range [0, 1]
    (x, y, z) in range [0, 1]
    """

    # apply inverse sRGB transfer function
    r = _srgb_transfer_function_inverse(r)
    g = _srgb_transfer_function_inverse(g)
    b = _srgb_transfer_function_inverse(b)

    # convert from sRGB (linear) to CIE XYZ (D65)
    x = 0.4124 * r + 0.3576 * g + 0.1805 * b
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = 0.0193 * r + 0.1192 * g + 0.9502 * b

    # restrict to [0, 1]
    x = _clamp(x)
    y = _clamp(y)
    z = _clamp(z)

    return (x, y, z)
