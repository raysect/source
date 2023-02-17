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

import unittest

from raysect.core import Point3D, Vector3D
from raysect.primitive.lens.spherical import BiConvex, BiConcave, PlanoConvex, PlanoConcave, Meniscus
from raysect.core.ray import Ray as CoreRay

import numpy as np
from math import sqrt, cos, sin, asin, fabs, pi


class TestSphericalLens(unittest.TestCase):
    """
    Tests for spherical lenses.

    Based on comparison of predicted position of intersection point and angle of incidence of a ray launched
    from outside against the lens surface. Lens front and back surface and barrel surface are tested.
    """

    pad_constant = 1e-6  # offset ratio to construct inside and outside points
    tolerance_distance = 10
    tolerance_angle = 10
    n_testpoints = 10  # points to test lens surfaces at
    test_altitudes = np.linspace(1 - pad_constant, 0, 10, endpoint=True)  # ratio of the lens radius (x-y plane) to test at
    test_azimuths = np.linspace(0, 2 * pi, 20)  # angles in xy plane to test the surface at
    test_barrel_z_points = np.linspace(0, 1, endpoint=True)  # points along lens barrel

    def test_biconvex(self):
        """
        Test biconvex lens for a range of curvatures and center thickness combinations.
        """

        # combinations of lens parameters to test
        diameter = 75
        front_curvatures = np.array([0.5, 0.75, 1, 1.5, 5]) * diameter
        back_curvatures = np.array([0.5, 0.75, 1, 1.5, 5]) * diameter
        threshold_ratios = np.array([1 + self.pad_constant, 1.05, 2, 3, 5, 10])

        radius = diameter / 2
        radius2 = radius ** 2

        for front_curvature in front_curvatures:
            for back_curvature in back_curvatures:
                for threshold_ratio in threshold_ratios:

                    # center thickness calculated from the minimum threshold
                    front_thickness = front_curvature - sqrt(front_curvature ** 2 - radius2)
                    back_thickness = back_curvature - sqrt(back_curvature ** 2 - radius2)
                    center_thickness = threshold_ratio * (front_thickness + back_thickness)

                    lens = BiConvex(diameter, center_thickness, front_curvature, back_curvature)

                    # calculate coordinates of center of curvatures of the lens front and back surfaces
                    center_back = Point3D(0, 0, back_curvature)
                    center_front = Point3D(0, 0, center_thickness - front_curvature)

                    # check lens front and back surface by inside and outside envelopes
                    azimuths = np.linspace(0, 2 * pi, self.n_testpoints, endpoint=False)
                    min_altitude = self.pad_constant

                    max_altitude = asin(radius / back_curvature) - self.pad_constant
                    altitude = np.linspace(max_altitude, min_altitude, self.n_testpoints, endpoint=True)
                    radii = back_curvature * np.sin(altitude)
                    self._check_spherical_surface(back_curvature, lens, center_back, False, False, azimuths, radii)

                    max_altitude = asin(radius / front_curvature) - self.pad_constant
                    altitude = np.linspace(max_altitude, min_altitude, self.n_testpoints, endpoint=True)
                    radii = front_curvature * np.sin(altitude)
                    self._check_spherical_surface(front_curvature, lens, center_front, False, True, azimuths, radii)

                    # check lens barrel surface by inside and outside envelope
                    min_z = back_thickness
                    max_z = center_thickness - front_thickness
                    if max_z - min_z <= 3 * self.pad_constant:
                        barrel_z = np.array(((min_z + max_z) / 2), ndmin=1)
                    else:
                        barrel_z = np.linspace(min_z + self.pad_constant, max_z - self.pad_constant, self.n_testpoints, endpoint=True)
                    self._check_barrel_surface(lens, azimuths, barrel_z)

    def test_biconcave(self):
        """
        Test biconvex lens for a range of curvatures and center thickness combinations.
        """

        # combinations of lens parameters to test
        diameter = 75
        front_curvatures = np.array([0.5, 0.75, 1, 1.5, 5]) * diameter
        back_curvatures = np.array([0.5, 0.75, 1, 1.5, 5]) * diameter
        threshold_ratios = np.array([0, 0.25, 0.5, 1, 2, 5, 10])

        radius = diameter / 2
        radius2 = radius ** 2

        for front_curvature in front_curvatures:
            for back_curvature in back_curvatures:
                for threshold_ratio in threshold_ratios:

                    # center thickness calculated from the minimum threshold
                    front_thickness = front_curvature - sqrt(front_curvature ** 2 - radius2)
                    back_thickness = back_curvature - sqrt(back_curvature ** 2 - radius2)
                    center_thickness = threshold_ratio * radius + self.pad_constant

                    lens = BiConcave(diameter, center_thickness, front_curvature, back_curvature)

                    # calculate coordinates of center of curvatures of the lens front and back surfaces
                    center_back = Point3D(0, 0, -back_curvature)
                    center_front = Point3D(0, 0, center_thickness + front_curvature)

                    # check lens front and back surface by inside and outside envelopes
                    azimuths = np.linspace(0, 2 * pi, self.n_testpoints, endpoint=False)
                    min_altitude = self.pad_constant

                    max_altitude = asin(radius / back_curvature) - self.pad_constant
                    altitude = np.linspace(max_altitude, min_altitude, self.n_testpoints, endpoint=True)
                    radii = back_curvature * np.sin(altitude)
                    self._check_spherical_surface(back_curvature, lens, center_back, True, True, azimuths, radii)

                    max_altitude = asin(radius / front_curvature) - self.pad_constant
                    altitude = np.linspace(max_altitude, min_altitude, self.n_testpoints, endpoint=True)
                    radii = front_curvature * np.sin(altitude)
                    self._check_spherical_surface(front_curvature, lens, center_front, True, False, azimuths, radii)

                    # check lens barrel surface by inside and outside envelope
                    min_z = -back_thickness
                    max_z = center_thickness + front_thickness
                    if max_z - min_z <= 3 * self.pad_constant:
                        barrel_z = np.array(((min_z + max_z) / 2), ndmin=1)
                    else:
                        barrel_z = np.linspace(min_z + self.pad_constant, max_z - self.pad_constant, self.n_testpoints, endpoint=True)
                    self._check_barrel_surface(lens, azimuths, barrel_z)

    def test_planoconvex(self):
        """
        Test planoconvex lens for a range of curvatures and center thickness combinations.
        """

        # combinations of lens parameters to test
        diameter = 75
        curvatures = np.array([0.5, 0.75, 1, 1.5, 5]) * diameter
        threshold_ratios = np.array([1 + self.pad_constant, 1.05, 2, 3, 5, 10])

        radius = diameter / 2
        radius2 = radius ** 2

        for curvature in curvatures:
            for threshold_ratio in threshold_ratios:

                # center thickness calculated from the minimum threshold
                front_thickness = curvature - sqrt(curvature ** 2 - radius2)
                center_thickness = threshold_ratio * front_thickness

                lens = PlanoConvex(diameter, center_thickness, curvature)

                # calculate coordinates of center of curvature of the lens front surface
                center_front = Point3D(0, 0, center_thickness - curvature)

                # check lens front and back surface by inside and outside envelopes
                azimuths = np.linspace(0, 2 * pi, self.n_testpoints, endpoint=False)
                min_altitude = self.pad_constant

                max_altitude = asin(radius / curvature) - self.pad_constant
                altitude = np.linspace(max_altitude, min_altitude, self.n_testpoints, endpoint=True)
                radii = curvature * np.sin(altitude)
                self._check_spherical_surface(curvature, lens, center_front, False, True,
                                              azimuths, radii)

                radii = np.linspace(radius - self.pad_constant, self.pad_constant, self.n_testpoints)
                self._check_plane_surface(lens, azimuths, radii)

                # check lens barrel surface by inside and outside envelope
                min_z = 0
                max_z = center_thickness - front_thickness
                if max_z - min_z <= 3 * self.pad_constant:
                    barrel_z = np.array(((min_z + max_z) / 2), ndmin=1)
                else:
                    barrel_z = np.linspace(min_z + self.pad_constant, max_z - self.pad_constant, self.n_testpoints, endpoint=True)
                self._check_barrel_surface(lens, azimuths, barrel_z)

    def test_planoconcave(self):
        """
        Test planoconcave lens for a range of curvatures and center thickness combinations.
        """

        # combinations of lens parameters to test
        diameter = 75
        curvatures = np.array([0.5, 0.75, 1, 1.5, 5]) * diameter
        threshold_ratios = np.array([1, 1.05, 2, 3, 5, 10])

        radius = diameter / 2
        radius2 = radius ** 2

        for curvature in curvatures:
            for threshold_ratio in threshold_ratios:

                # center thickness calculated from the minimum threshold
                front_thickness = curvature - sqrt(curvature ** 2 - radius2)
                center_thickness = threshold_ratio * front_thickness

                lens = PlanoConcave(diameter, center_thickness, curvature)

                # calculate coordinates of center of curvature of the lens front surface
                center_front = Point3D(0, 0, center_thickness + curvature)

                # check lens front and back surface by inside and outside envelopes
                azimuths = np.linspace(0, 2 * pi, self.n_testpoints, endpoint=False)
                min_altitude = self.pad_constant

                max_altitude = asin(radius / curvature) - self.pad_constant
                altitude = np.linspace(max_altitude, min_altitude, self.n_testpoints, endpoint=True)
                radii = curvature * np.sin(altitude)
                self._check_spherical_surface(curvature, lens, center_front, True, False, azimuths, radii)

                radii = np.linspace(radius - self.pad_constant, self.pad_constant, self.n_testpoints)
                self._check_plane_surface(lens, azimuths, radii)

                # check lens barrel surface by inside and outside envelope
                min_z = 0
                max_z = center_thickness + front_thickness
                if max_z - min_z <= 3 * self.pad_constant:
                    barrel_z = np.array(((min_z + max_z) / 2), ndmin=1)
                else:
                    barrel_z = np.linspace(min_z + self.pad_constant, max_z - self.pad_constant, self.n_testpoints, endpoint=True)
                self._check_barrel_surface(lens, azimuths, barrel_z)

    def test_meniscus(self):
        """
        Test meniscus lens for a range of curvatures and center thickness combinations.
        """

        # define range of lens parameter combinations to test
        diameter = 75

        # combinations of lens parameters to test
        front_curvatures = np.array([0.5, 0.75, 1, 1.5, 5]) * diameter
        back_curvatures = np.array([0.5, 0.75, 1, 1.5, 5]) * diameter
        threshold_ratios = np.array([1.01, 1.1, 1.5, 2, 5, 10])

        radius = diameter / 2
        radius2 = radius ** 2

        for front_curvature in front_curvatures:
            for back_curvature in back_curvatures:
                for threshold_ratio in threshold_ratios:

                    # center thickness calculated from the minimum threshold
                    front_thickness = front_curvature - sqrt(front_curvature ** 2 - radius2)
                    back_thickness = back_curvature - sqrt(back_curvature ** 2 - radius2)
                    threshold = fabs(back_thickness - front_thickness) + 1e-3
                    center_thickness = threshold * threshold_ratio

                    lens = Meniscus(diameter, center_thickness, front_curvature, back_curvature)

                    # calculate coordinates of center of curvatures of the lens front and back surfaces
                    center_back = Point3D(0, 0, -back_curvature)
                    center_front = Point3D(0, 0, center_thickness - front_curvature)

                    # check lens front and back surface by inside and outside envelopes
                    azimuths = np.linspace(0, 2 * pi, self.n_testpoints, endpoint=False)
                    min_altitude = self.pad_constant

                    max_altitude = asin(radius / back_curvature) - self.pad_constant
                    altitude = np.linspace(max_altitude, min_altitude, self.n_testpoints, endpoint=True)
                    radii = back_curvature * np.sin(altitude)
                    self._check_spherical_surface(back_curvature, lens, center_back, True, True,
                                                  azimuths, radii)

                    max_altitude = asin(radius / front_curvature) - self.pad_constant
                    altitude = np.sin(np.linspace(max_altitude, min_altitude, self.n_testpoints))
                    radii = front_curvature * np.sin(altitude)
                    self._check_spherical_surface(front_curvature, lens, center_front, False, True,
                                                  azimuths, radii)

                    # check lens barrel surface by inside and outside envelope
                    # skip barrel test if the edge thickness is too small to avoid rays missing the lense
                    min_z = -back_thickness
                    max_z = center_thickness - front_thickness
                    if max_z - min_z <= 3 * self.pad_constant:
                        barrel_z = np.array(((min_z + max_z) / 2), ndmin=1)
                    else:
                        barrel_z = np.linspace(min_z + self.pad_constant, max_z - self.pad_constant, self.n_testpoints,
                                               endpoint=True)

                    self._check_barrel_surface(lens, azimuths, barrel_z)

    def _check_spherical_surface(self, curvature, lens, center_curvature, is_inside, positive_curvature, azimuths, radii):
        """
        Checks barrel surface of a lens by calculating ray-lens intersection. Hit point position and angle of incidence
        are compared to predicted ones.

        :param curvature: Curvature radius of the surface
        :param lens: Spherical lens object to test plane surface of.
        :param center_curvature: Point3D with center of curvature coordinates.
        :param is_inside: If True, the lens body is within the curvature sphere.
        :param positive_curvature: Orientation of the lens surface with respect to the center of curvature. If positive,
        :param azimuths: Azimuth angles to test lens surface at.
        :param radii: Radii to test lens surface at.
        """

        # set the direction of the test ray
        if is_inside is True:
            ray_direction = 1
        elif is_inside is False:
            ray_direction = -1

        # set the sphere direction
        if positive_curvature is True:
            hemisphere = 1
        elif positive_curvature is False:
            hemisphere = -1

        curvature2 = curvature ** 2

        for radius in radii:
            z = sqrt(curvature2 - radius ** 2)
            for ta in azimuths:
                # calculate position vector pointing from the curvature center to the surface point
                x = radius * cos(ta)
                y = radius * sin(ta)
                position_vector = Vector3D(x, y, hemisphere * z)

                # construct origin by surface point offset and calculate ray direction
                surface_point = center_curvature + position_vector
                origin = center_curvature + position_vector * (1 - 0.1 * ray_direction)
                direction = ray_direction * position_vector

                # calculate ray-lens intersection
                intersection = lens.hit(CoreRay(origin, direction))
                hit_point = intersection.hit_point.transform(intersection.primitive_to_world)

                # distance of expected surface point and the ray hit point
                distance = hit_point.vector_to(surface_point).length
                self.assertAlmostEqual(distance, 0, self.tolerance_distance,
                                       msg="Ray-curved surface hit point and predicted surface point difference"
                                           " is larger than tolerance.")

                # angle of incidence on the sphere surface should be perpendicular
                cos_angle_incidence = intersection.normal.dot(intersection.ray.direction.normalise())

                self.assertAlmostEqual(fabs(cos_angle_incidence), 1, self.tolerance_angle,
                                       msg="Angle of incidence differs from perpendicular.")

    def _check_barrel_surface(self, lens, azimuths, barrel_z):
        """
        Checks barrel surface of a lens by calculating ray-lens intersection. Hit point position and angle of incidence
        are compared to predicted ones.

        :param lens: Spherical lens object to test plane surface of.
        :param azimuths: Azimuth angles to test lens surface at.
        :param barrel_z:
        """

        lens_radius = lens.diameter / 2

        for z in barrel_z:
            for ta in azimuths:
                # get x-y coordinates of the surface point from azimuth
                x = lens_radius * cos(ta)
                y = lens_radius * sin(ta)

                # get origin by surface point offset and calculate ray direction
                surface_point = Point3D(x, y, z)
                direction = Vector3D(-x, -y, 0)
                origin = Point3D(1.1 * x, 1.1 * y, z)

                # calculate ray-lens intersection
                intersection = lens.hit(CoreRay(origin, direction))
                hit_point = intersection.hit_point.transform(intersection.primitive_to_world)

                # distance of expected surface point and the ray hit point
                distance = hit_point.vector_to(surface_point).length
                self.assertAlmostEqual(distance, 0, self.tolerance_distance,
                                       msg="Ray-curved surface hit point and predicted surface point difference"
                                           " is larger than tolerance.")

                # angle of incidence on the sphere surface should be perpendicular
                cos_angle_incidence = intersection.normal.dot(intersection.ray.direction.normalise())
                self.assertAlmostEqual(fabs(cos_angle_incidence), 1, self.tolerance_angle,
                                       msg="Angle of incidence differs from perpendicular.")

    def _check_plane_surface(self, lens, azimuths, radii):
        """
        Checks plane surface of a lens by calculating ray-lens intersection. Hit point position and angle of incidence
        are compared to predicted ones.

        :param lens: Spherical lens object to test plane surface of.
        :param azimuths: Azimuth angles to test lens surface at.
        :param radii: Radii to test plane surface at.
        """

        for radius in radii:
            for ta in azimuths:
                # get coordinates of the surface point from azimuth
                x = radius * cos(ta)
                y = radius * sin(ta)
                z = 0  # plane surface is always at z = 0

                # get origin by surface point offset and calculate ray direction
                surface_point = Point3D(x, y, z)
                origin = Point3D(x, y, z - lens.diameter)
                direction = Vector3D(0, 0, 1)

                # calculate ray-lens intersection
                intersection = lens.hit(CoreRay(origin, direction))
                hit_point = intersection.hit_point.transform(intersection.primitive_to_world)

                # distance of expected surface point and the ray hit point
                distance = hit_point.vector_to(surface_point).length
                self.assertAlmostEqual(distance, 0, self.tolerance_distance,
                                       msg="Ray-curved surface hit point and predicted surface point difference"
                                           " is larger than tolerance.")

                # angle of incidence on the sphere surface should be perpendicular
                cos_angle_incidence = intersection.normal.dot(intersection.ray.direction.normalise())
                self.assertAlmostEqual(fabs(cos_angle_incidence), 1, self.tolerance_angle,
                                       msg="Angle of incidence differs from perpendicular.")
