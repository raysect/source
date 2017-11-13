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

from libc.math cimport cos, M_PI as PI

from raysect.core.math.random cimport probability
from raysect.core.math.sampler cimport RectangleSampler3D, HemisphereCosineSampler, TargettedHemisphereSampler
from raysect.optical cimport Ray, Primitive, Point3D, Vector3D, BoundingSphere3D
from raysect.optical.observer.base cimport Observer0D
from raysect.optical.observer.pipeline.spectral import SpectralPowerPipeline0D
cimport cython


DEF R_2_PI = 0.15915494309189535  # 1 / (2 * pi)


cdef class TargettedPixel(Observer0D):
    """
    A pixel observer that samples rays from a hemisphere and rectangular area.

    :param list pipelines: The list of pipelines that will process the spectrum measured
      by this pixel (default=SpectralPipeline0D()).
    :param float x_width: The rectangular collection area's width along the
      x-axis in local coordinates (default=1cm).
    :param float y_width: The rectangular collection area's width along the
      y-axis in local coordinates (default=1cm).
    :param kwargs: **kwargs from Observer0D and _ObserverBase
    """

    cdef:
        double _x_width, _y_width, _solid_angle, _collection_area, _targetted_path_prob
        tuple _targets
        RectangleSampler3D _point_sampler
        HemisphereCosineSampler _cosine_sampler
        TargettedHemisphereSampler _targetted_sampler

    def __init__(self, pipelines=None, x_width=None, y_width=None, parent=None, transform=None, name=None, targets=None,
                 targetted_path_prob = None,
                 render_engine=None, pixel_samples=None, samples_per_task=None, spectral_rays=None, spectral_bins=None,
                 min_wavelength=None, max_wavelength=None, ray_extinction_prob=None, ray_extinction_min_depth=None,
                 ray_max_depth=None, ray_importance_sampling=None, ray_important_path_weight=None, quiet=False):

        pipelines = pipelines or [SpectralPowerPipeline0D()]

        super().__init__(pipelines, parent=parent, transform=transform, name=name, render_engine=render_engine,
                         pixel_samples=pixel_samples, samples_per_task=samples_per_task, spectral_rays=spectral_rays,
                         spectral_bins=spectral_bins, min_wavelength=min_wavelength, max_wavelength=max_wavelength,
                         ray_extinction_prob=ray_extinction_prob, ray_extinction_min_depth=ray_extinction_min_depth,
                         ray_max_depth=ray_max_depth, ray_importance_sampling=ray_importance_sampling,
                         ray_important_path_weight=ray_important_path_weight, quiet=quiet)

        self._x_width = 0.01
        self._y_width = 0.01
        self._cosine_sampler = HemisphereCosineSampler()
        self._targetted_sampler = None
        self._solid_angle = 2 * PI

        self.x_width = x_width or 0.01
        self.y_width = y_width or 0.01

        self.targets = targets
        self.targetted_path_prob = targetted_path_prob or 0.9

    @property
    def x_width(self):
        """
        The rectangular collection area's width along the x-axis in local coordinates.

        :rtype: float
        """
        return self._x_width

    @x_width.setter
    def x_width(self, value):
        if value <= 0:
            raise RuntimeError("Pixel x-width must be greater than zero.")
        self._x_width = value
        self._point_sampler = RectangleSampler3D(width=self._x_width, height=self._y_width)
        self._collection_area = self._x_width * self._y_width

    @property
    def y_width(self):
        """
        The rectangular collection area's width along the y-axis in local coordinates.

        :rtype: float
        """
        return self._y_width

    @y_width.setter
    def y_width(self, value):
        if value <= 0:
            raise RuntimeError("Pixel y-width must be greater than zero.")
        self._y_width = value
        self._point_sampler = RectangleSampler3D(width=self._x_width, height=self._y_width)
        self._collection_area = self._x_width * self._y_width

    @property
    def collection_area(self):
        """
        The pixel's collection area in m^2.

        :rtype: float
        """
        return self._collection_area

    @property
    def solid_angle(self):
        """
        The pixel's solid angle in steradians str.

        :rtype: float
        """
        return self._solid_angle

    @property
    def etendue(self):
        """
        The pixel's etendue measured in units of per area per solid angle (m^-2 str^-1).

        :rtype: float
        """
        return self._pixel_etendue()

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, value):

        # No targets?
        if value is None:
            self._targets = None
            return

        value = tuple(value)

        # Empty tuple?
        if not value:
            self._targets = None
            return

        # List must contain only primitives
        for target in value:
            if not isinstance(target, Primitive):
                raise TypeError("Target list must contain only primitives.")

        self._targets = value

    @property
    def targetted_path_prob(self):
        return self._targetted_path_prob

    @targetted_path_prob.setter
    def targetted_path_prob(self, double value):

        if value < 0 or value > 1:
            raise ValueError("Targeted path probability must lie in the range [0, 1].")
        self._targetted_path_prob = value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef list _generate_rays(self, Ray template, int ray_count):

        if self._targets is None:
            return self._generate_rays_untargetted(template, ray_count)
        else:
            return self._generate_rays_targetted(template, ray_count)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef list _generate_rays_untargetted(self, Ray template, int ray_count):

        cdef:
            list rays, origins, directions
            int n
            double weight

        origins = self._point_sampler.samples(ray_count)
        directions = self._cosine_sampler.samples(ray_count)

        rays = []
        for n in range(ray_count):

            # cosine weighted distribution
            # projected area cosine is implicit in distribution
            # weight = 1 / (2 * pi) * (pi / cos(theta)) * cos(theta) = 0.5
            rays.append((template.copy(origins[n], directions[n]), 0.5))

        return rays

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef list _generate_rays_targetted(self, Ray template, int ray_count):

        cdef:
            list rays, origins, spheres
            int n
            double weight, distance, sphere_radius
            Point3D sphere_centre, ray_origin
            Vector3D sphere_direction, ray_direction
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

        # sample pixel origins
        origins = self._point_sampler.samples(ray_count)

        rays = []
        for origin in origins:

            if probability(self._targetted_path_prob):
                # obtain targetted vector sample
                direction = self._targetted_sampler.sample(origin)

            else:
                # obtain cosine weighted hemisphere sample
                direction = self._cosine_sampler.sample()

            # calculate combined pdf and ray weight
            pdf = self._targetted_path_prob * self._targetted_sampler.pdf(origin, direction) + \
                  (1-self._targetted_path_prob) * self._cosine_sampler.pdf(direction)

            # weight = 1 / (2 * pi) * cos(theta) * 1/pdf
            weight = R_2_PI * direction.z / pdf

            rays.append((template.copy(origin, direction), weight))

        return rays


        # cdef:
        #     list rays, origins
        #     int n
        #     double weight, distance, sphere_radius
        #     Point3D sphere_centre, ray_origin
        #     Vector3D sphere_direction, ray_direction
        #     BoundingSphere3D sphere
        #
        # # test target primitive is in the same scenegraph as the observer
        # if self.target.root is not self.root:
        #     raise ValueError("Target primitive is not in the same scenegraph as the TargetedPixel observer.")
        #
        # origins = self._point_sampler.samples(ray_count)
        #
        # # obtain bounding sphere and convert position to local coordinate space
        # sphere = self.target.bounding_sphere()
        # sphere_centre = sphere.centre.transform(self.to_local())
        # sphere_radius = sphere.radius
        #
        # rays = []
        # for n in range(ray_count):
        #
        #     ray_origin = origins[n]
        #
        #     # are we importance sample - yes or no
        #     # if importance sampling enabled (targets exist) and probability of importance sample == True:
        #     if probability(self._targetted_path_prob):
        #
        #         # calculating the PDF
        #         self._add_targetted_sample(template, ray_origin, sphere_centre, sphere_radius, rays)
        #
        #     else:
        #         # cosine weighted hemisphere sampling
        #         self._add_hemisphere_sample(template, ray_origin, 1 - self._targetted_path_prob, rays)
        #
        # return rays



    cpdef double _pixel_etendue(self):
        return self._solid_angle * self._collection_area
