# Copyright (c) 2016, Dr Alex Meakins, Raysect Project
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

from raysect.optical.observer.old.point_generator import Rectangle
from raysect.optical.observer.sampler2d import FullFrameSampler2D
from raysect.optical.observer.pipeline import RGBPipeline2D

from raysect.optical.observer.old.point_generator cimport PointGenerator
from raysect.core cimport Point3D, new_point3d, Vector3D, new_vector3d
from raysect.optical cimport Ray
from libc.math cimport M_PI as pi, tan
from raysect.optical.observer.base cimport Observer2D


cdef class VectorCamera(Observer2D):
    """
    An observer that uses a specified set of pixel vectors.

    A simple camera that uses calibrated vectors for each pixel to sample the scene.
    Arguments and attributes are inherited from the base Observer2D sensor class.

    :param double fov: The field of view of the camera in degrees (default is 90 degrees).
    """

    cdef:
        double image_delta, image_start_x, image_start_y
        double[:,::1] pixel_origins, pixel_directions
        PointGenerator point_generator

    def __init__(self, pixel_origins, pixel_directions, pixels, parent=None, transform=None, name=None, pipelines=None):

        pipelines = pipelines or [RGBPipeline2D()]

        super().__init__(pixels, FullFrameSampler2D(), pipelines,
                         parent=parent, transform=transform, name=name)

        # camera configuration
        self.pixel_origins = pixel_origins
        self.pixel_directions = pixel_directions

    cpdef list _generate_rays(self, tuple pixel_id, Ray template, int ray_count):

        cdef:
            int ix, iy
            list rays
            Point3D origin
            Vector3D direction
            Ray ray

        # unpack
        ix, iy = pixel_id

        # assemble rays
        origin = self.pixel_origins[ix, iy]
        direction = self.pixel_directions[ix, iy]

        # assemble rays
        rays = []
        for i in range(self._pixel_samples):

            ray = template.copy(origin, direction)

            # projected area weight is normal.incident which simplifies
            # to incident.z here as the normal is (0, 0 ,1)
            rays.append((ray, 1.0))

        return rays

    cpdef double _pixel_etendue(self, tuple pixel_id):
        return 1.0
