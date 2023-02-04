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

from raysect.core.math.cython cimport clamp
from raysect.optical.spectralfunction cimport InterpolatedSF
from numpy import array, float64, zeros, linspace
from numpy cimport ndarray
cimport cython

# CIE 1931 Standard Colorimetric Observer (normalised)
ciexyz_wavelength_samples = array([
    375, 380, 385, 390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440,
    445, 450, 455, 460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510,
    515, 520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580,
    585, 590, 595, 600, 605, 610, 615, 620, 625, 630, 635, 640, 645, 650,
    655, 660, 665, 670, 675, 680, 685, 690, 695, 700, 705, 710, 715, 720,
    725, 730, 735, 740, 745, 750, 755, 760, 765, 770, 775, 780, 785],
    dtype=float64)

ciexyz_x_samples = array([
    0.000000, 0.001368, 0.002236, 0.004243, 0.007650, 0.014310, 0.023190,
    0.043510, 0.077630, 0.134380, 0.214770, 0.283900, 0.328500, 0.348280,
    0.348060, 0.336200, 0.318700, 0.290800, 0.251100, 0.195360, 0.142100,
    0.095640, 0.057950, 0.032010, 0.014700, 0.004900, 0.002400, 0.009300,
    0.029100, 0.063270, 0.109600, 0.165500, 0.225750, 0.290400, 0.359700,
    0.433450, 0.512050, 0.594500, 0.678400, 0.762100, 0.842500, 0.916300,
    0.978600, 1.026300, 1.056700, 1.062200, 1.045600, 1.002600, 0.938400,
    0.854450, 0.751400, 0.642400, 0.541900, 0.447900, 0.360800, 0.283500,
    0.218700, 0.164900, 0.121200, 0.087400, 0.063600, 0.046770, 0.032900,
    0.022700, 0.015840, 0.011359, 0.008111, 0.005790, 0.004109, 0.002899,
    0.002049, 0.001440, 0.001000, 0.000690, 0.000476, 0.000332, 0.000235,
    0.000166, 0.000117, 0.000083, 0.000059, 0.000042, 0.000000]) / 106.8566

ciexyz_y_samples = array([
    0.000000, 0.000039, 0.000064, 0.000120, 0.000217, 0.000396, 0.000640,
    0.001210, 0.002180, 0.004000, 0.007300, 0.011600, 0.016840, 0.023000,
    0.029800, 0.038000, 0.048000, 0.060000, 0.073900, 0.090980, 0.112600,
    0.139020, 0.169300, 0.208020, 0.258600, 0.323000, 0.407300, 0.503000,
    0.608200, 0.710000, 0.793200, 0.862000, 0.914850, 0.954000, 0.980300,
    0.994950, 1.000000, 0.995000, 0.978600, 0.952000, 0.915400, 0.870000,
    0.816300, 0.757000, 0.694900, 0.631000, 0.566800, 0.503000, 0.441200,
    0.381000, 0.321000, 0.265000, 0.217000, 0.175000, 0.138200, 0.107000,
    0.081600, 0.061000, 0.044580, 0.032000, 0.023200, 0.017000, 0.011920,
    0.008210, 0.005723, 0.004102, 0.002929, 0.002091, 0.001484, 0.001047,
    0.000740, 0.000520, 0.000361, 0.000249, 0.000172, 0.000120, 0.000085,
    0.000060, 0.000042, 0.000030, 0.000021, 0.000015, 0.000000]) / 106.8566

ciexyz_z_samples = array([
    0.000000, 0.006450, 0.010550, 0.020050, 0.036210, 0.067850, 0.110200,
    0.207400, 0.371300, 0.645600, 1.039050, 1.385600, 1.622960, 1.747060,
    1.782600, 1.772110, 1.744100, 1.669200, 1.528100, 1.287640, 1.041900,
    0.812950, 0.616200, 0.465180, 0.353300, 0.272000, 0.212300, 0.158200,
    0.111700, 0.078250, 0.057250, 0.042160, 0.029840, 0.020300, 0.013400,
    0.008750, 0.005750, 0.003900, 0.002750, 0.002100, 0.001800, 0.001650,
    0.001400, 0.001100, 0.001000, 0.000800, 0.000600, 0.000340, 0.000240,
    0.000190, 0.000100, 0.000050, 0.000030, 0.000020, 0.000010, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]) / 106.8566

ciexyz_x = InterpolatedSF(ciexyz_wavelength_samples, ciexyz_x_samples)
ciexyz_y = InterpolatedSF(ciexyz_wavelength_samples, ciexyz_y_samples)
ciexyz_z = InterpolatedSF(ciexyz_wavelength_samples, ciexyz_z_samples)

# CIE D65 standard illuminant, normalised to 1.0 over visual range 375-785 nm
d65_wavelength_samples = linspace(300.0, 830.0, 107)

d65_white_samples = array([
      0.034100,   1.664300,   3.294500,  11.765200,  20.236000,  28.644700,
     37.053500,  38.501100,  39.948800,  42.430200,  44.911700,  45.775000,
     46.638300,  49.363700,  52.089100,  51.032300,  49.975500,  52.311800,
     54.648200,  68.701500,  82.754900,  87.120400,  91.486000,  92.458900,
     93.431800,  90.057000,  86.682300,  95.773600, 104.865000, 110.936000,
    117.008000, 117.410000, 117.812000, 116.336000, 114.861000, 115.392000,
    115.923000, 112.367000, 108.811000, 109.082000, 109.354000, 108.578000,
    107.802000, 106.296000, 104.790000, 106.239000, 107.689000, 106.047000,
    104.405000, 104.225000, 104.046000, 102.023000, 100.000000,  98.167100,
     96.334200,  96.061100,  95.788000,  92.236800,  88.685600,  89.345900,
     90.006200,  89.802600,  89.599100,  88.648900,  87.698700,  85.493600,
     83.288600,  83.493900,  83.699200,  81.863000,  80.026800,  80.120700,
     80.214600,  81.246200,  82.277800,  80.281000,  78.284200,  74.002700,
     69.721300,  70.665200,  71.609100,  72.979000,  74.349000,  67.976500,
     61.604000,  65.744800,  69.885600,  72.486300,  75.087000,  69.339800,
     63.592700,  55.005400,  46.418200,  56.611800,  66.805400,  65.094100,
     63.382800,  63.843400,  64.304000,  61.877900,  59.451900,  55.705400,
     51.959000,  54.699800,  57.440600,  58.876500,  60.312500]) / 87.1971

#: CIE D65 standard illuminant, normalised to 1.0 over visual range 375-785 nm
d65_white = InterpolatedSF(d65_wavelength_samples, d65_white_samples)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,::1] resample_ciexyz(double min_wavelength, double max_wavelength, int bins):
    """
    Pre-calculates samples of XYZ sensitivity curves over desired spectral range.

    Returns ndarray of shape [N, 3] where the last dimension (0, 1, 2) corresponds
    to (X, Y, Z).

    :param float min_wavelength: Lower wavelength bound on spectrum
    :param float max_wavelength: Upper wavelength bound on spectrum
    :param int bins: Number of spectral bins in spectrum
    :rtype: memoryview
    """

    cdef ndarray xyz

    if bins < 1:
        raise("Number of samples can not be less than 1.")

    if min_wavelength <= 0.0 or max_wavelength <= 0.0:
        raise ValueError("Wavelength can not be less than or equal to zero.")

    if min_wavelength >= max_wavelength:
        raise ValueError("Minimum wavelength can not be greater or equal to the maximum wavelength.")

    xyz = zeros((bins, 3))
    xyz[:, 0] = ciexyz_x.sample(min_wavelength, max_wavelength, bins)
    xyz[:, 1] = ciexyz_y.sample(min_wavelength, max_wavelength, bins)
    xyz[:, 2] = ciexyz_z.sample(min_wavelength, max_wavelength, bins)

    return xyz


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
cpdef (double, double, double) spectrum_to_ciexyz(Spectrum spectrum, double[:,::1] resampled_xyz=None):
    """
    Calculates a tuple of CIE X, Y, Z values from an input spectrum

    :param Spectrum spectrum: Spectrum to process
    :param memoryview resampled_xyz: Pre-calculated XYZ sensitivity curves optimised
      for this spectral range (default=None).
    :rtype: tuple
    """

    cdef:
        double x, y, z
        int index

    if resampled_xyz is None:
        resampled_xyz = resample_ciexyz(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.bins)

    if resampled_xyz.shape[0] != spectrum.bins or resampled_xyz.shape[1] != 3:
        raise ValueError("The supplied resampled_xyz array size is inconsistent with the number of spectral bins or channel count.")

    x = 0
    y = 0
    z = 0
    for index in range(spectrum.bins):
        x += spectrum.delta_wavelength * spectrum.samples_mv[index] * resampled_xyz[index, 0]
        y += spectrum.delta_wavelength * spectrum.samples_mv[index] * resampled_xyz[index, 1]
        z += spectrum.delta_wavelength * spectrum.samples_mv[index] * resampled_xyz[index, 2]

    return x, y, z


@cython.cdivision(True)
cpdef (double, double, double) ciexyy_to_ciexyz(double cx, double cy, double y):
    """
    Performs conversion from CIE xyY to CIE XYZ colour space

    Returns a tuple of (X, Y, Z)

    :param float cx: chromaticity x
    :param float cy: chromaticity y
    :param float y: tristimulus Y
    :rtype: tuple
    """
    return y / cy * cx, y, y / cy * (1 - cx - cy)


@cython.cdivision(True)
cpdef (double, double, double) ciexyz_to_ciexyy(double x, double y, double z):
    """
    Performs conversion from CIE XYZ to CIE xyY colour space

    Returns a tuple of (cx, cy, Y)

    :param float x: tristimulus X
    :param float y: tristimulus y
    :param float z: tristimulus Z
    :rtype: tuple
    """

    cdef double n

    n = x + y + z
    return x / n, y / n, y


cdef double srgb_transfer_function(double v):

    if v <= 0.0031308:

        return 12.92 * v

    else:

        # optimised: return 1.055 * v**(1 / 2.4) - 0.055
        return 1.055 * v**0.4166666666666667 - 0.055


cpdef (double, double, double) ciexyz_to_srgb(double x, double y, double z):
    """
    Convert CIE XYZ values to sRGB colour space.

    x, y, z in range [0, 1]
    r, g, b in range [0, 1]

    :param float x: tristimulus X
    :param float y: tristimulus y
    :param float z: tristimulus Z
    :rtype: tuple
    """

    cdef:
        double r, g, b

    # convert from CIE XYZ (D65) to sRGB (linear)
    r =  3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b =  0.0556434 * x - 0.2040259 * y + 1.0572252 * z

    # apply sRGB transfer function
    r = srgb_transfer_function(r)
    g = srgb_transfer_function(g)
    b = srgb_transfer_function(b)

    # restrict to [0, 1]
    r = clamp(r, 0, 1)
    g = clamp(g, 0, 1)
    b = clamp(b, 0, 1)

    return r, g, b


cdef double srgb_transfer_function_inverse(double v):

    if v <= 0.04045:

        # optimised: return v / 12.92
        return 0.07739938080495357 * v

    else:

        # optimised: return ((v + 0.055) / 1.055)**2.4
        return (0.9478672985781991 * (v + 0.055))**2.4


cpdef (double, double, double) srgb_to_ciexyz(double r, double g, double b):
    """
    Convert sRGB values to CIE XYZ colour space.

    r, g, b in range [0, 1]
    x, y, z in range [0, 1]

    :param float r: Red value
    :param float g: Green value
    :param float b: Blue value
    :rtype: tuple
    """

    cdef:
        double x, y, z

    # apply inverse sRGB transfer function
    r = srgb_transfer_function_inverse(r)
    g = srgb_transfer_function_inverse(g)
    b = srgb_transfer_function_inverse(b)

    # convert from sRGB (linear) to CIE XYZ (D65)
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    return x, y, z
