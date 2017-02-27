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

cdef class RayGenerator:
    """
    An immutable class that generates rays.

    When implementing a new observer a ray generator is provided to generate
    new rays for sampling over an observer collection area. The object is
    immutable once constructed, preventing observer developers from
    accidentally modifying the ray template held in the observer.

    A ray instance, appropriately configured, is used as the template for
    new rays. When a new ray is requested, a copy of the template, with
    spectral properties, origin and direction appropriately adjusted, will
    be returned.

    :param template: An optical ray instance.
    :param min_wavelength: Lower end of spectral wavelength range.
    :param max_wavelength: Upper end of spectral wavelength range.
    :param bins: Number of spectral bins across wavelength range.
    """

    def __init__(self, Ray template not None, double min_wavelength, double max_wavelength, int bins):

        # copy template ray and configure spectral properties
        template = template.copy()
        template.wavelength_range = (min_wavelength, max_wavelength)
        template.bins = bins
        self._template = template

    cpdef Ray new_ray(self, Point3D origin, Vector3D direction):
        """
        Returns a new ray with the specified origin and direction.

        :param origin: The ray origin point.
        :param direction: The ray direction vector.
        :return: An optical ray instance.
        """

        return self._template.copy(origin, direction)