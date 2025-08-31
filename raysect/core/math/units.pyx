# cython: language_level=3

# Copyright (c) 2014-2025, Dr Alex Meakins, Raysect Project
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

from libc.math cimport M_1_PI


cpdef double km(double v) noexcept:
    """
    Converts kilometers to meters.

    :param float v: Length in kilometers.
    :return: Length in meters.
    """
    return v * 1e3


cpdef double cm(double v) noexcept:
    """
    Converts centimeters to meters.

    :param float v: Length in centimeters.
    :return: Length in meters.
    """
    return v * 1e-2


cpdef double mm(double v) noexcept:
    """
    Converts millimeters to meters.

    :param float v: Length in millimeters.
    :return: Length in meters.
    """
    return v * 1e-3


cpdef double um(double v) noexcept:
    """
    Converts micrometers to meters.

    :param float v: Length in micrometers.
    :return: Length in meters.
    """
    return v * 1e-6


cpdef double nm(double v) noexcept:
    """
    Converts nanometers to meters.

    :param float v: Length in nanometers.
    :return: Length in meters.
    """
    return v * 1e-9


cpdef double mile(double v) noexcept:
    """
    Converts miles to meters.

    :param float v: Length in miles.
    :return: Length in meters.
    """
    return v * 1609.34


cpdef double yard(double v) noexcept:
    """
    Converts yards to meters.

    :param float v: Length in yards.
    :return: Length in meters.
    """
    return v * 0.9144


cpdef double foot(double v) noexcept:
    """
    Converts feet to meters.

    :param float v: Length in feet.
    :return: Length in meters.
    """
    return v * 0.3048


cpdef double inch(double v) noexcept:
    """
    Converts inches to meters.

    :param float v: Length in inches.
    :return: Length in meters.
    """
    return v * 0.0254


cpdef double mil(double v) noexcept:
    """
    Converts mils (thousandths of an inch) to meters.

    :param float v: Length in mils.
    :return: Length in meters.
    """
    return v * 2.54e-5


cpdef double radian(double v) noexcept:
    """
    Converts radians to degrees.

    :param float v: Angle in radians.
    :return: Angle in degrees.
    """
    return v * 180 * M_1_PI






