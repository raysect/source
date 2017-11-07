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

from raysect.core.math.sampler cimport RectangleSampler3D, HemisphereCosineSampler
from raysect.core.boundingsphere cimport BoundingSphere3D
from raysect.optical cimport Primitive, Ray, Point3D, Vector3D, AffineMatrix3D
from raysect.optical.observer.base cimport Observer0D
from raysect.optical.observer.pipeline.spectral import SpectralRadiancePipeline0D
from raysect.core.math.random cimport probability, vector_cone_uniform
from raysect.core.math.cython cimport rotate_basis

from libc.math cimport M_PI as PI, asin, sqrt

cimport cython

# TODO - the pdf calcualtion is horrifically broken, do not user until this is fixed!!!

# TODO - make sure the cached bounding sphere is reset when a change to the scenegraph occurs
cdef class TargetedPixel(Observer0D):
    """
    A pixel observer that samples are targetted solid angle and rectangular area.

    If the user wants to target and empty space or a hole in a structure. They should
    create a primitive that occupies the space and give it a Null material.

    .. Warning..
       This should only be used when the only source of light lies in the solid angle
       of the targetted primitive as observed by the observer. If this is not the case any
       power calculations will be invalid.

    :param Primitive target: The primitive that this observer will target.
    :param list pipelines: The list of pipelines that will process the spectrum measured
      by this pixel (default=SpectralPipeline0D()).
    :param float x_width: The rectangular collection area's width along the
      x-axis in local coordinates (default=1cm).
    :param float y_width: The rectangular collection area's width along the
      y-axis in local coordinates (default=1cm).
    :param kwargs: **kwargs from Observer0D and _ObserverBase
    """

    cdef:
        double _x_width, _y_width, _solid_angle, _collection_area
        double _targetted_path_prob
        Primitive target
        RectangleSampler3D _point_sampler
        HemisphereCosineSampler _fallback_vector_sampler

    def __init__(self, target, pipelines=None, x_width=None, y_width=None, parent=None, transform=None, name=None,
                 render_engine=None, pixel_samples=None, samples_per_task=None, spectral_rays=None, spectral_bins=None,
                 min_wavelength=None, max_wavelength=None, ray_extinction_prob=None, ray_extinction_min_depth=None,
                 ray_max_depth=None, ray_importance_sampling=None, ray_important_path_weight=None,
                 targetted_path_prob=None, quiet=False):

        pipelines = pipelines or [SpectralRadiancePipeline0D()]

        super().__init__(pipelines, parent=parent, transform=transform, name=name, render_engine=render_engine,
                         pixel_samples=pixel_samples, samples_per_task=samples_per_task, spectral_rays=spectral_rays,
                         spectral_bins=spectral_bins, min_wavelength=min_wavelength, max_wavelength=max_wavelength,
                         ray_extinction_prob=ray_extinction_prob, ray_extinction_min_depth=ray_extinction_min_depth,
                         ray_max_depth=ray_max_depth, ray_importance_sampling=ray_importance_sampling,
                         ray_important_path_weight=ray_important_path_weight, quiet=quiet)

        self.target = target
        self.targetted_path_prob = targetted_path_prob or 0.9

        self._x_width = 0.01
        self._y_width = 0.01
        self._fallback_vector_sampler = HemisphereCosineSampler()
        self._solid_angle = 2 * PI

        self.x_width = x_width or 0.01
        self.y_width = y_width or 0.01

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
    def targetted_path_prob(self):
        return self._targetted_path_prob

    @targetted_path_prob.setter
    def targetted_path_prob(self, double value):

        if value < 0 or value > 1:
            raise ValueError("Targeted path probability must lie in the range [0, 1].")
        self._targetted_path_prob = value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef list _generate_rays(self, Ray template, int ray_count):

        cdef:
            list rays, origins
            int n
            double weight, distance, sphere_radius
            Point3D sphere_centre, ray_origin
            Vector3D sphere_direction, ray_direction
            BoundingSphere3D sphere

        # test target primitive is in the same scenegraph as the observer
        if self.target.root is not self.root:
            raise ValueError("Target primitive is not in the same scenegraph as the TargetedPixel observer.")

        origins = self._point_sampler.samples(ray_count)

        # obtain bounding sphere and convert position to local coordinate space
        sphere = self.target.bounding_sphere()
        sphere_centre = sphere.centre.transform(self.to_local())
        sphere_radius = sphere.radius

        rays = []
        for n in range(ray_count):

            ray_origin = origins[n]

            # are we importance sample - yes or no
            # if importance sampling enabled (targets exist) and probability of importance sample == True:
            if probability(self._targetted_path_prob):

                # calculating the PDF
                self._add_targetted_sample(template, ray_origin, sphere_centre, sphere_radius, rays)

            else:
                # cosine weighted hemisphere sampling
                self._add_hemisphere_sample(template, ray_origin, 1 - self._targetted_path_prob, rays)

        return rays

    @cython.cdivision(True)
    cdef _add_targetted_sample(self, Ray template, Point3D ray_origin, Point3D sphere_centre, double sphere_radius, list rays):

        cdef:
            double weight, distance, t, angular_radius, angular_radius_cos, sphere_angle
            Vector3D sphere_direction, ray_direction
            AffineMatrix3D rotation

        # TODO - select target

        sphere_direction = ray_origin.vector_to(sphere_centre)
        distance = sphere_direction.get_length()
        sphere_direction = sphere_direction.normalise()

        # is point inside sphere?
        if distance == 0 or distance < sphere_radius:
            self._add_hemisphere_sample(template, ray_origin, self._targetted_path_prob, rays)
            return

        # calculate the angular radius
        t = sphere_radius / distance
        angular_radius = asin(t)
        angular_radius_cos = sqrt(1 - t * t)

        # Tests direction angle is always above cone angle:
        # We have yet to derive the partial projected area of a sphere intersecting the horizon
        # for now we will just fall back to hemisphere sampling, even if this is inefficient.
        # This test also implicitly checks if the sphere direction lies in the hemisphere as the
        # angular_radius cannot be less than zero
        # todo: calculate projected area of a cut sphere to improve sampling
        sphere_angle = asin(sphere_direction.z)
        if angular_radius >= sphere_angle:
            self._add_hemisphere_sample(template, ray_origin, self._targetted_path_prob, rays)
            return

        # sample a vector from a cone of half angle equal to the angular radius
        ray_direction = vector_cone_uniform(angular_radius * 180 / PI)

        # rotate cone to lie along vector from observation point to sphere centre
        rotation = rotate_basis(sphere_direction, sphere_direction.orthogonal())
        ray_direction =  ray_direction.transform(rotation)

        # calculate sampling weight
        # pdf = 1 / solid_angle = 1 / (2 * pi * (1 - cos(angular_radius)) * target_path_prob
        # weight = 1 / (2 * pi) * cos(theta) * 1 / pdf
        weight = ray_direction.z * (1 - angular_radius_cos) / self._targetted_path_prob

        rays.append((template.copy(ray_origin, ray_direction), weight))

    @cython.cdivision(True)
    cdef void _add_hemisphere_sample(self, Ray template, Point3D origin, double selection_prob, list rays):

        cdef Vector3D direction

        # perform hemisphere sampling
        direction = self._fallback_vector_sampler.sample()

        # cosine weighted distribution
        # projected area cosine is implicit in distribution
        # weight = 1 / (2 * pi) * (pi / cos(theta)) * cos(theta) = 0.5
        # multiplying by 1 / selection probability to correct the pdf given there
        # are two possible code paths.
        rays.append((template.copy(origin, direction), 0.5 / selection_prob))

    cpdef double _pixel_etendue(self):
        return self._solid_angle * self._collection_area

