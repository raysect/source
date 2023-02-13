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

from raysect.core.math.random cimport probability
from raysect.optical.observer.sampler2d import FullFrameSampler2D
from raysect.optical.observer.pipeline import RGBPipeline2D

from raysect.core.math.sampler cimport RectangleSampler3D, HemisphereCosineSampler, TargettedHemisphereSampler
from raysect.optical cimport Primitive, BoundingSphere3D, Ray, AffineMatrix3D, Point3D, Vector3D, translate
from libc.math cimport M_PI
from raysect.optical.observer.base cimport Observer2D
cimport cython


DEF R_2_PI = 0.15915494309189535  # 1 / (2 * pi)


cdef class TargettedCCDArray(Observer2D):
    """
    An ideal CCD-like imaging sensor that preferentially targets a given list of primitives.

    The targetted CCD is a regular array of square pixels. Each pixel samples red, green
    and blue channels (behaves like a Foveon imaging sensor). The CCD sensor
    width is specified with the width parameter. The CCD height is calculated
    from the width and the number of vertical and horizontal pixels. The
    default width and sensor ratio approximates a 35mm camera sensor.

    The targetted CCD takes a list of target primitives. Each pixel will target the
    bounding spheres that encompass each target primitive. Therefore, for best performance,
    the target primitives should be split up such that their surfaces are closely wrapped
    by the bounding sphere.

    The sampling algorithm fires a proportion of rays at the targets, and a portion sampled
    from the full hemisphere. The proportion that is fired towards the targets is controlled
    with the targetted_path_prob attribute. By default this attribute is set to 0.9, i.e.
    90% of the rays are fired towards the targets.

    .. Warning..
       If the target probability is set to 1, rays will only be fired directly towards the
       targets. The user must ensure there are no sources of radiance outside of the
       targeted directions, otherwise they will not be sampled and the result will be biased.

    :param list targets: The list of primitives for targetted sampling.
    :param tuple pixels: A tuple of pixel dimensions for the camera (default=(720, 480)).
    :param float width: The CCD sensor x-width in metres (default=35mm).
    :param list pipelines: The list of pipelines that will process the spectrum measured
      at each pixel by the camera (default=RGBPipeline2D()).
    :param kwargs: **kwargs and properties from Observer2D and _ObserverBase.
    """

    def __init__(self, targets, pixels=(720, 480), width=0.035, targetted_path_prob=None, parent=None, transform=None, name=None, pipelines=None):

        # initial values to prevent undefined behaviour when setting via self.width
        self._width = 0.035
        self._pixels = (720, 480)

        pipelines = pipelines or [RGBPipeline2D()]

        super().__init__(pixels, FullFrameSampler2D(), pipelines,
                         parent=parent, transform=transform, name=name)

        self._cosine_sampler = HemisphereCosineSampler()
        self._targetted_sampler = None

        # setting width triggers calculation of image geometry calculations
        self.width = width
        self.targets = targets
        self.targetted_path_prob = targetted_path_prob or 0.9

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
    def width(self):
        """
        The CCD sensor x-width in metres.

        :rtype: float
        """
        return self._width

    @width.setter
    def width(self, width):
        if width <= 0:
            raise ValueError("width can not be less than or equal to 0 meters.")
        self._width = width
        self._update_image_geometry()

    @property
    def targets(self):
        """
        The list of primitives this pixel will target for sampling.

        :rtype: list
        """
        return self._targets

    @targets.setter
    def targets(self, value):

        # No targets?
        if value is None:
            raise ValueError("Targets must be a list of primitives.")

        value = tuple(value)

        # Empty tuple?
        if not value:
            raise ValueError("Targets list cannot be empty.")

        # List must contain only primitives
        for target in value:
            if not isinstance(target, Primitive):
                raise TypeError("Target list must contain only primitives.")

        self._targets = value

    @property
    def targetted_path_prob(self):
        """
        The probability that an individual sample will be fired at a target instead of a sample from the whole hemisphere.

        .. Warning..
           If the target probability is set to 1, rays will only be fired directly towards the targets. The user must
           ensure there are now sources of radiance outside of the targeted directions, otherwise they will not be
           sampled and the result will be biased.

        :rtype: float
        """
        return self._targetted_path_prob

    @targetted_path_prob.setter
    def targetted_path_prob(self, double value):

        if value < 0 or value > 1:
            raise ValueError("Targeted path probability must lie in the range [0, 1].")
        self._targetted_path_prob = value

    cdef object _update_image_geometry(self):

        self._image_delta = self._width / self._pixels[0]
        self._image_start_x = 0.5 * self._pixels[0] * self._image_delta
        self._image_start_y = 0.5 * self._pixels[1] * self._image_delta
        self._point_sampler = RectangleSampler3D(self._image_delta, self._image_delta)
        self._pixel_area = (self._width / self._pixels[0])**2

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef list _generate_rays(self, int ix, int iy, Ray template, int ray_count):

        cdef:
            double pixel_x, pixel_y
            list origins, rays, spheres
            Point3D origin
            Vector3D direction
            Ray ray
            AffineMatrix3D pixel_to_local
            BoundingSphere3D sphere
            Primitive target

        # test target primitives are in the same scene-graph as the observer
        for target in self._targets:
            if target.root is not self.root:
                raise ValueError("Target primitive is not in the same scenegraph as the TargetedPixel observer.")

        # generate bounding spheres and convert to local coordinate system
        spheres = []
        for target in self._targets:
            sphere = target.bounding_sphere()
            spheres.append((sphere.centre.transform(self.to_local()), sphere.radius, 1.0))

        # instance targetted pixel sampler
        self._targetted_sampler = TargettedHemisphereSampler(spheres)

        # generate pixel transform
        pixel_x = self._image_start_x - self._image_delta * (ix + 0.5)
        pixel_y = self._image_start_y - self._image_delta * (iy + 0.5)
        pixel_to_local = translate(pixel_x, pixel_y, 0)

        # generate origin points in pixel space
        origins = self._point_sampler.samples(ray_count)

        # assemble rays
        rays = []
        for origin in origins:

            # transform to local space from pixel space
            origin = origin.transform(pixel_to_local)

            if probability(self._targetted_path_prob):
                # obtain targetted vector sample
                direction = self._targetted_sampler.sample(origin)

            else:
                # obtain cosine weighted hemisphere sample
                direction = self._cosine_sampler.sample()

            # calculate combined pdf and ray weight
            pdf = self._targetted_path_prob * self._targetted_sampler.pdf(origin, direction) + \
                  (1-self._targetted_path_prob) * self._cosine_sampler.pdf(direction)

            if pdf <= 0:
                raise ValueError('Ray direction probability is zero. The target object extends beyond the pixel horizon.')

            # weight = 1 / (2 * pi) * cos(theta) * 1/pdf
            weight = R_2_PI * direction.z / pdf

            rays.append((template.copy(origin, direction), weight))

        return rays

    cpdef double _pixel_sensitivity(self, int x, int y):
        return self._pixel_area * 2 * M_PI
