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

import numpy as np
cimport numpy as np
import cv2

from raysect.core cimport Point3D, Vector3D, RectangleSampler3D
from raysect.optical cimport Ray
from raysect.optical.observer.base cimport Observer2D
from raysect.optical.observer.sampler2d import FullFrameSampler2D, RGBAdaptiveSampler2D
from raysect.optical.observer.pipeline import RGBPipeline2D


cdef class OpenCVCamera(Observer2D):
    """
    An observer based on the OpenCV camera model.

    A simple analytic camera that uses calibrated camera parameters to re-generate the
    pixel vectors. The following parameters need to be supplied.

    * pinhole and barrel distortion terms :math:`(k_1, k_2, p_1, p_2, k_3)`.
    * camera matrix describing the focal lengths :math:`(f_x, f_y)` and
      optical centres :math:`(c_x, c_y)` in pixel coordinates.
    * R and T coordinate vectors defining the transformation coordinates.
    * pixel dimensions of the camera.

    See the OpenCV documentation `here
    <https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html>`_ and `here
    <https://docs.opencv.org/3.4.0/d9/d0c/group__calib3d.html>`_ for more details.

    Arguments and attributes are inherited from the base Observer2D sensor class.

    :param ndarray camera_matrix: focal lengths :math:`(f_x, f_y)` and optical centres :math:`(c_x, c_y)`
      in pixel coordinates.
    :param tuple distortion: tuple/list/array of pinhole and barrel distortion terms :math:`(k_1, k_2, p_1, p_2, k_3)`.
    :param tuple r_vector: R coordinate vector.
    :param tuple t_vector: T coordinate vector.
    :param tuple pixels: The pixel dimensions of the camera.
    :param FrameSampler2D frame_sampler: The frame sampling strategy (default=FullFrameSampler2D()).
    :param list pipelines: The list of pipelines that will process the spectrum measured
      at each pixel by the camera (default=RGBPipeline2D()).
    :param float etendue: The constant etendue factor applied to each pixel (default=1).
    :param kwargs: **kwargs and properties from Observer2D and _ObserverBase.
    """

    def __init__(self, camera_matrix, distortion, r_vector, t_vector, pixels,
                 frame_sampler=None, pipelines=None, etendue=None, parent=None, transform=None, name=None):

        camera_matrix = np.matrix(camera_matrix, dtype=np.float64)
        if not camera_matrix.shape == (3, 3):
            raise TypeError("The OpenCV camera matrix must be 3x3 matrix or numpy array.")

        distortion = np.array(distortion, dtype=np.float64)
        if not distortion.shape == (5,):
            raise TypeError("The OpenCV camera's distortion array must have 5 distortion terms.")

        r_vector = np.array(r_vector, dtype=np.float64)
        if not r_vector.shape == (3,):
            raise TypeError("The OpenCV camera's r vector must be described by 3 coordinate values.")

        t_vector = np.array(t_vector, dtype=np.float64)
        if not t_vector.shape == (3,):
            raise TypeError("The OpenCV camera's t vector must be described by 3 coordinate values.")

        if not isinstance(pixels, tuple) or not len(pixels) == 2 or not any([isinstance(pi, int) for pi in pixels]):
            raise TypeError("The OpenCV camera's pixel shape must a tuple, e.g. (512, 512).")

        # save OpenCV camera attributes
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.r_vector = r_vector
        self.t_vector = t_vector

        # Get camera pupil position as a point 3D
        self._rotation_matrix = cv2.Rodrigues(self.r_vector)[0].transpose()
        camera_position = np.matrix(self.t_vector).transpose()
        camera_position = - (np.matrix(self._rotation_matrix) * camera_position)
        self._origin = Point3D(camera_position[0][0], camera_position[1][0], camera_position[2][0])

        # defaults to an adaptively sampled RGB pipeline
        if not pipelines and not frame_sampler:
            rgb = RGBPipeline2D()
            pipelines = [rgb]
            frame_sampler = RGBAdaptiveSampler2D(rgb)
        else:
            pipelines = pipelines or [RGBPipeline2D()]
            frame_sampler = frame_sampler or FullFrameSampler2D()

        super().__init__(pixels, frame_sampler, pipelines, parent=parent, transform=transform, name=name)

        self._sensitivity = etendue or 1.0

        self.point_sampler = RectangleSampler3D(1, 1)

    @property
    def sensitivity(self):
        """
        The sensitivity applied to each pixel.

        If sensitivity=1.0 all spectral units are in radiance.

        :rtype: float
        """
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value):
        if value <= 0:
            raise ValueError("Sensitivity must be greater than zero.")
        self._sensitivity = value

    cpdef list _generate_rays(self, int x, int y, Ray template, int ray_count):

        cdef:
            list rays, pixel_points, directions
            Point3D pixel_point
            Vector3D local_direction, world_direction
            Ray ray
            int i
            np.ndarray input_points, undistorted, rotation_matrix

        rotation_matrix = self._rotation_matrix

        # sample the start positions inside pixel space (u->[-0.5, 0.5], v->[-0.5, 0.5]).
        pixel_points = self.point_sampler.samples(ray_count)

        # move the sample points from coordinates inside local pixel, to pixel coordinates in camera space
        # use OpenCV to transform / undistort the points using the camera matrix and distortion coefficients
        input_points = np.zeros([len(pixel_points), 1, 2])
        for i in range(len(pixel_points)):
            pixel_point = pixel_points[i]
            input_points[i, 0, 0] = pixel_point.x + x
            input_points[i, 0, 1] = pixel_point.y + y
        undistorted = cv2.undistortPoints(input_points, self.camera_matrix, self.distortion)

        directions = []
        for i in range(len(undistorted)):
            local_direction = Vector3D(undistorted[i, 0, 0], undistorted[i, 0, 1], 1).normalise()
            world_direction = Vector3D(
                local_direction.x * rotation_matrix[0, 0] + local_direction.y * rotation_matrix[0, 1] + local_direction.z * rotation_matrix[0, 2],
                local_direction.x * rotation_matrix[1, 0] + local_direction.y * rotation_matrix[1, 1] + local_direction.z * rotation_matrix[1, 2],
                local_direction.x * rotation_matrix[2, 0] + local_direction.y * rotation_matrix[2, 1] + local_direction.z * rotation_matrix[2, 2],
            )
            directions.append(world_direction)

        # assemble rays
        rays = []
        for i in range(ray_count):

            ray = template.copy(self._origin, directions[i])

            # projected area weight is normal.incident which simplifies
            # to incident.z here as the normal is (0, 0 ,1)
            rays.append((ray, 1.0))

        return rays

    cpdef double _pixel_sensitivity(self, int x, int y):
        return self._sensitivity
