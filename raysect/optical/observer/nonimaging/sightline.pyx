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

from raysect.optical.observer import PowerPipeline0D

from raysect.optical cimport Ray, new_point3d, new_vector3d
from raysect.optical.observer.base cimport Observer0D


# TODO: complete docstrings
cdef class SightLine(Observer0D):
    """
    An observer that fires rays along the observers z axis.
    Inherits arguments and attributes from the base NonImaging sensor class. Fires a single ray oriented along the
    observer's z axis in world space.
    """

    cdef double _etendue

    def __init__(self, etendue=None, pipelines=None, parent=None, transform=None, name=None):

        pipelines = pipelines or [PowerPipeline0D()]
        super().__init__(pipelines, parent=parent, transform=transform, name=name)
        self.etendue = etendue or 1.0

    @property
    def etendue(self):
        return self._etendue

    @etendue.setter
    def etendue(self, value):
        if value <= 0:
            raise ValueError('Etendue must be greater than zero.')
        self._etendue = value

    cpdef list _generate_rays(self, Ray template, int ray_count):

        cdef:
            list rays
            int n

        rays = []
        for n in range(ray_count):
            rays.append((template.copy(new_point3d(0, 0, 0), new_vector3d(0, 0, 1)), 1.0))
        return rays

    cpdef double _pixel_etendue(self):
        return self._etendue
