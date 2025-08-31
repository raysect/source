# cython: language_level=3

# Copyright (c) 2014-2018, Dr Alex Meakins, Raysect Project
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

import numpy as np
cimport numpy as np

from raysect.optical.observer.sampler2d import FullFrameSampler2D
from raysect.optical.observer.pipeline import RGBPipeline2D
from raysect.optical.observer.sampler2d import RGBAdaptiveSampler2D

from raysect.core cimport Point3D, new_point3d, Vector3D, new_vector3d, RectangleSampler3D
from raysect.optical cimport Ray
from libc.math cimport M_PI as pi, sin, cos
from raysect.optical.observer.base cimport Observer2D


cdef class FovCamera(Observer2D):
    """
    Field-of-view observer. A camera that launches rays from the observer's origin point
    over a specified field of view in spherical coordinates. Each pixel of the final image
    represents a solid angle of collection inside an azimuth-altitude rectangle.

    :param tuple pixels: A tuple of pixel dimensions for the camera, i.e. (512, 512).
    :param tuple fov: The field of view of the camera in degrees in horizontal (azimuthal) and
        vertical (altitudinal) directions (default=(90, 90) degrees).
    :param float sensitivity: The sensitivity of each pixel (default=1.0)
    :param FrameSampler2D frame_sampler: The frame sampling strategy, defaults to adaptive
        sampling (i.e. extra samples for noisier pixels).
    :param list pipelines: The list of pipelines that will process the spectrum measured
        at each pixel by the camera (default=RGBPipeline2D()).
    :param kwargs: **kwargs and properties from Observer2D and _ObserverBase.

    :ivar np.ndarray azimuth: Azimuthal angles (in degrees) corresponding to pixels' centres
        in x-direction.
    :ivar np.ndarray altitude: Altitudinal angles (in degrees) corresponding to pixels' centres
        in y-direction.
    :ivar np.ndarray angle_h: Same as `azimuth`.
    :ivar np.ndarray angle_v: Same as `altitude`.

    .. code-block:: pycon

        >>> from raysect.core import translate
        >>> from raysect.optical import World
        >>> from raysect.optical.observer import FovCamera, PowerPipeline2D
        >>>
        >>> power = PowerPipeline2D(display_unsaturated_fraction=0.96, name="Unfiltered")
        >>>
        >>> camera = FovCamera((512, 256), fov=(70, 35), parent=world, pipelines=[power])
        >>> camera.transform = translate(0, 0, -3.3)
        >>> camera.pixel_samples = 250
        >>> camera.spectral_bins = 15
        >>>
        >>> camera.observe()
    """
    cdef:
        double _sensitivity, azimuth_delta, altitude_delta
        np.ndarray _azimuth, _altitude
        tuple _fov
        RectangleSampler3D point_sampler

    def __init__(self, pixels, fov=(90., 90.), sensitivity=1.0, frame_sampler=None,
                 pipelines=None, parent=None, transform=None, name=None):

        # defaults to an adaptively sampled RGB pipeline
        if not pipelines and not frame_sampler:
            rgb = RGBPipeline2D()
            pipelines = [rgb]
            frame_sampler = RGBAdaptiveSampler2D(rgb)
        else:
            pipelines = pipelines or [RGBPipeline2D()]
            frame_sampler = frame_sampler or FullFrameSampler2D()

        self._fov = (90., 90.)
        super().__init__(pixels, frame_sampler, pipelines, parent=parent, transform=transform, name=name)

        # note that the fov property triggers a call to _update_image_geometry()
        self.fov = fov
        self.sensitivity = sensitivity

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, value):
        value = tuple(value)
        if len(value) != 2:
            raise ValueError("Fov must be a 2 element tuple defining the field-of-view in horizontal and vertical directions")
        if value[0] <= 0 or value[0] >= 180 or value[1] <= 0 or value[1] >= 180:
            raise ValueError("The field-of-view angle must lie in the range (0, 180).")
        self._fov = value
        self._update_image_geometry()

    @property
    def pixels(self):
        return self._pixels

    @pixels.setter
    def pixels(self, value):
        pixels = tuple(value)
        if len(pixels) != 2:
            raise ValueError("Pixels must be a 2 element tuple defining the x and y resolution.")
        x, y = pixels
        if x <= 0:
            raise ValueError("Number of x pixels must be greater than 0.")
        if y <= 0:
            raise ValueError("Number of y pixels must be greater than 0.")
        self._pixels = pixels
        self._update_image_geometry()

    @property
    def sensitivity(self):
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value):
        if value <= 0:
            raise ValueError("Sensitivity must be greater than zero.")
        self._sensitivity = value

    @property
    def azimuth(self):
        return self._azimuth

    @property
    def altitude(self):
        return self._altitude

    @property
    def angle_h(self):
        return self._azimuth

    @property
    def angle_v(self):
        return self._altitude

    cpdef np.ndarray solid_angle(self):
        """
        Returns solid angles of collection (in steradians) for all pixels.
        """

        cdef:
            np.ndarray solid_angle1d

        solid_angle1d = (pi / 180.) * self.azimuth_delta * (np.sin((pi / 180.) * (self._altitude + 0.5 * self.altitude_delta)) -
                                                            np.sin((pi / 180.) * (self._altitude - 0.5 * self.altitude_delta)))

        return solid_angle1d[None, :] * np.ones(self._azimuth.size)[:, None]

    cdef object _update_image_geometry(self):
        self.azimuth_delta = self._fov[0] / self._pixels[0]
        self.altitude_delta = self._fov[1] / self._pixels[1]

        self._azimuth = np.linspace(-0.5 * (self._fov[0] - self.azimuth_delta),
                                    0.5 * (self._fov[0] - self.azimuth_delta),
                                    self._pixels[0])
        self._altitude = np.linspace(0.5 * (self._fov[1] - self.altitude_delta),
                                     -0.5 * (self._fov[1] - self.altitude_delta),
                                     self._pixels[1])

        self.point_sampler = RectangleSampler3D(self.azimuth_delta * pi / 180., self.altitude_delta * pi / 180.)

    cpdef list _generate_rays(self, int x, int y, Ray template, int ray_count):

        cdef:
            double azimuth_c, altitude_c, azimuth_p, altitude_p
            list points, rays
            Point3D origin
            Vector3D direction
            Ray ray

        azimuth_c = self._azimuth[x] * pi / 180.
        altitude_c = self._altitude[y] * pi / 180.
        points = self.point_sampler(ray_count)

        # assemble rays
        rays = []
        for point in points:
            azimuth_p = azimuth_c + point.x
            altitude_p = altitude_c + point.y

            origin = new_point3d(0, 0, 0)
            direction = new_vector3d(
                -sin(azimuth_p) * cos(altitude_p),
                sin(altitude_p),
                cos(azimuth_p) * cos(altitude_p)
            ).normalise()

            ray = template.copy(origin, direction)

            rays.append((ray, 1.0))

        return rays

    cpdef double _pixel_sensitivity(self, int x, int y):
        return self._sensitivity
