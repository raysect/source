import unittest

from raysect.core import Point3D, Vector3D
from raysect.primitive.lens.spherical import BiConvex, BiConcave, PlanoConvex, PlanoConcave, Meniscus
from raysect.core.ray import Ray as CoreRay
from raysect.optical import World
import numpy as np
from math import sqrt, cos, sin, asin, fabs, pi


class TestSphericalLense(unittest.TestCase):
    """
    The idea is to create a set of points making an inside and outside envelope of the lens surface
    and test if they are correctly contained by the lens primitive.
    """

    pad_constant = 1e-6  # offset ratio to construct inside and outside points
    tolerance_distance = 12
    tolerance_angle = 10
    # points to test lens surfaces at
    test_altitudes = np.linspace(1 - pad_constant, 0, 10,
                                     endpoint=True)  # ratio of the lens radius (x-y plane) to test at
    test_azimuths = np.linspace(0, 2 * pi, 20)  # angles in xy plane to test the surface at
    test_barrel_z_points = np.linspace(0, 1, endpoint=True)  # points along lens barrel

    def test_biconvex(self):

        # define range of lens parameter combinations to test
        diameter = 75

        front_curvatures = np.array([0.5, 0.75, 1, 1.5, 5]) * diameter
        back_curvatures = np.array([0.5, 0.75, 1, 1.5, 5]) * diameter
        threshold_ratios = np.array([1, 1.05, 2, 3, 5, 10])

        radius = diameter / 2
        radius2 = radius ** 2

        for _, front_curvature in enumerate(front_curvatures):
            for _, back_curvature in enumerate(back_curvatures):
                for _, threshold_ratio in enumerate(threshold_ratios):

                    # center thickness calculated from the minimum threshold
                    front_thickness = front_curvature - sqrt(front_curvature ** 2 - radius2)
                    back_thickness = back_curvature - sqrt(back_curvature ** 2 - radius2)
                    center_thickness = threshold_ratio * (front_thickness + back_thickness)

                    lens = BiConvex(diameter, center_thickness, front_curvature, back_curvature)

                    # calculate coordinates of center of curvatures of the lens front and back surfaces
                    center_back = Point3D(0, 0, back_curvature)
                    center_front = Point3D(0, 0, center_thickness - front_curvature)

                    # check lens front and back surface by inside and outside envelopes
                    self._check_spherical_surface(back_curvature, lens, center_back, False, "negative")
                    self._check_spherical_surface(front_curvature, lens, center_front, False, "positive")

                    # check lens barrel surface by inside and outside envelope
                    self._check_barrel_surface(back_thickness, center_thickness - front_thickness, lens)

    def test_biconcave(self):

        # define range of lens parameter combinations to test
        diameter = 75

        front_curvatures = np.array([0.5, 0.75, 1, 1.5, 5]) * diameter
        back_curvatures = np.array([0.5, 0.75, 1, 1.5, 5]) * diameter
        threshold_ratios = np.array([0, 0.25, 0.5, 1, 2, 5, 10])

        radius = diameter / 2
        radius2 = radius ** 2

        for _, front_curvature in enumerate(front_curvatures):
            for _, back_curvature in enumerate(back_curvatures):
                for _, threshold_ratio in enumerate(threshold_ratios):

                    # center thickness calculated from the minimum threshold
                    front_thickness = front_curvature - sqrt(front_curvature ** 2 - radius2)
                    back_thickness = back_curvature - sqrt(back_curvature ** 2 - radius2)
                    center_thickness = threshold_ratio * radius + self.pad_constant

                    lens = BiConcave(diameter, center_thickness, front_curvature, back_curvature)

                    # calculate coordinates of center of curvatures of the lens front and back surfaces
                    center_back = Point3D(0, 0, -back_curvature)
                    center_front = Point3D(0, 0, center_thickness + front_curvature)

                    # check lens front and back surface by inside and outside envelopes
                    self._check_spherical_surface(back_curvature, lens, center_back, True, "positive")
                    self._check_spherical_surface(front_curvature, lens, center_front, True, "negative")

                    # check lens barrel surface by inside and outside envelope
                    self._check_barrel_surface(-back_thickness, center_thickness + front_thickness, lens)

    def test_planoconvex(self):

        # define lens properties and initialise lens
        # define range of lens parameter combinations to test
        diameter = 75

        curvatures = np.array([0.5, 0.75, 1, 1.5, 5]) * diameter
        threshold_ratios = np.array([1, 1.05, 2, 3, 5, 10])

        radius = diameter / 2
        radius2 = radius ** 2

        for _, curvature in enumerate(curvatures):
            for _, threshold_ratio in enumerate(threshold_ratios):

                # center thickness calculated from the minimum threshold
                front_thickness = curvature - sqrt(curvature ** 2 - radius2)
                center_thickness = threshold_ratio * front_thickness

                lens = PlanoConvex(diameter, center_thickness, curvature)

                # calculate coordinates of center of curvature of the lens front surface
                center_front = Point3D(0, 0, center_thickness - curvature)

                # check lens front and back surface by inside and outside envelopes
                self._check_spherical_surface(curvature, lens, center_front, False, "positive")
                self._check_plane_surface(lens)

                # check lens barrel surface by inside and outside envelope
                self._check_barrel_surface(0, center_thickness - front_thickness, lens)

    def test_planoconcave(self):

        # define range of lens parameter combinations to test
        diameter = 75

        curvatures = np.array([0.5, 0.75, 1, 1.5, 5]) * diameter
        threshold_ratios = np.array([1, 1.05, 2, 3, 5, 10])

        radius = diameter / 2
        radius2 = radius ** 2

        for _, curvature in enumerate(curvatures):
            for _, threshold_ratio in enumerate(threshold_ratios):

                # center thickness calculated from the minimum threshold
                front_thickness = curvature - sqrt(curvature ** 2 - radius2)
                center_thickness = threshold_ratio * front_thickness

                lens = PlanoConcave(diameter, center_thickness, curvature)

                # calculate coordinates of center of curvature of the lens front surface
                center_front = Point3D(0, 0, center_thickness + curvature)

                # check lens front and back surface by inside and outside envelopes
                self._check_spherical_surface(curvature, lens, center_front, True, "negative")
                self._check_plane_surface(lens)

                # check lens barrel surface by inside and outside envelope
                self._check_barrel_surface(0, center_thickness + front_thickness, lens)

    def test_meniscus(self):

        # define range of lens parameter combinations to test
        diameter = 75

        front_curvatures = np.array([0.5, 0.75, 1, 1.5, 5]) * diameter
        back_curvatures = np.array([0.5, 0.75, 1, 1.5, 5]) * diameter
        threshold_ratios = np.array([1.01, 1.1, 1.5, 2, 5, 10])

        radius = diameter / 2
        radius2 = radius ** 2

        for _, front_curvature in enumerate(front_curvatures):
            for _, back_curvature in enumerate(back_curvatures):
                for _, threshold_ratio in enumerate(threshold_ratios):

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
                    #self._check_spherical_surface(back_curvature, lens, center_back, True, "positive")
                    self._check_spherical_surface(front_curvature, lens, center_front, False, "positive")

                    # check lens barrel surface by inside and outside envelope
                    # skip barrel test if the edge thickness is too small to avoid rays missing the lense
                    span = center_thickness + front_thickness - back_thickness
                    if span > 1e-6:
                        start = -back_thickness + span / 10
                        stop = center_thickness - front_thickness - span / 10
                        self._check_barrel_surface(-back_thickness, center_thickness - front_thickness, lens)

    def _check_spherical_surface(self, curvature, lens, center_curvature, inside, orientation):
        """

        :param curvature: Radius of the sphere forming the lens surface
        :param lens: Raysect lens object
        :param center_curvature: Position of the center of curvature.
        :param inside: Sets the lens outside region. Is True for test rays being shot from within the surface sphere..
        :param orientation: Sets which hemisphere is forming the surface. For positive the hemisphere used to
               construct the lens surface is in the direction of the z axis from the sphere center.
        :return:
        """
        # set the direction of the test ray
        if inside:
            ray_direction = 1
        else:
            ray_direction = -1

        if orientation == "positive":
            hemisphere = 1
        if orientation == "negative":
            hemisphere = -1

        lens_radius = lens.diameter / 2
        curvature2 = curvature ** 2

        # generate test-point on the lens front/back surface
        test_radii = curvature * np.sin(self.test_altitudes * asin(lens_radius / curvature))
        # test back surface
        for _, radius in enumerate(test_radii):
            z = sqrt(curvature2 - radius ** 2)
            for _, ta in enumerate(self.test_azimuths):
                # calculate position vector pointing from the curvature center to the surface point
                x = radius * cos(ta)
                y = radius * sin(ta)
                position_vector = Vector3D(x, y, hemisphere * z)

                # construct inside point by offsetting the surface point and test it
                surface_point = center_curvature + position_vector
                origin = center_curvature + position_vector * (1 - 0.1 * ray_direction)
                direction = ray_direction * position_vector

                intersection = lens.hit(CoreRay(origin, direction))
                hit_point = intersection.hit_point.transform(intersection.primitive_to_world)
                # distance of expected surface point and the ray hit point
                distance = hit_point.vector_to(surface_point).length
                # angle of incidence on the sphere surface should be perpendicular
                cos_angle_incidence = intersection.normal.dot(intersection.ray.direction.normalise())

                self.assertAlmostEqual(distance, 0, self.tolerance_distance, msg="Ray-curved surface hit point and predicted surface point difference"
                                                            " is larger than tolerance.")

                self.assertAlmostEqual(fabs(cos_angle_incidence), 1, self.tolerance_angle,
                                       msg="Angle of incidence differs from perpendicular.")

    def _check_barrel_surface(self, start, stop, lens):
        """
        Cylindrical surface with radius equal to lens radius stretching in between front and back lens edges
        """

        lens_radius = lens.diameter / 2

        # calculate z range for barrel points
        span = stop - start
        barrel_z = ((start + self.pad_constant * span) +
                    self.test_barrel_z_points * (stop - start - 2 * self.pad_constant * span))

        for _, z in enumerate(barrel_z):
            for _, ta in enumerate(self.test_azimuths):
                # construct and test inside point
                x = lens_radius * cos(ta)
                y = lens_radius * sin(ta)
                surface_point = Point3D(x, y, z)
                direction = Vector3D(-x, -y, 0)
                origin = Point3D(1.1 * x, 1.1 * y, z)

                intersection = lens.hit(CoreRay(origin, direction))
                if intersection is None:
                    print("c")
                hit_point = intersection.hit_point.transform(intersection.primitive_to_world)
                # distance of expected surface point and the ray hit point
                distance = hit_point.vector_to(surface_point).length
                # angle of incidence on the sphere surface should be perpendicular
                cos_angle_incidence = intersection.normal.dot(intersection.ray.direction.normalise())

                self.assertAlmostEqual(distance, 0, self.tolerance_distance,
                                       msg="Ray-curved surface hit point and predicted surface point difference"
                                           " is larger than tolerance.")
                #self.assertAlmostEqual(fabs(cos_angle_incidence), 1, self.tolerance_angle,
                #                       msg="Angle of incidence differs from perpendicular.")

    def _check_plane_surface(self, lens):

        # generate test-point on the lens front/back surface
        test_radii = self.test_altitudes * lens.diameter / 2

        for _, radius in enumerate(test_radii):
            for _, ta in enumerate(self.test_azimuths):
                x = radius * cos(ta)
                y = radius * sin(ta)
                z = 0

                surface_point = Point3D(x, y, z)
                origin = Point3D(x, y, z - lens.diameter)
                direction = Vector3D(0, 0, 1)

                intersection = lens.hit(CoreRay(origin, direction))
                hit_point = intersection.hit_point.transform(intersection.primitive_to_world)

                # distance of expected surface point and the ray hit point
                distance = hit_point.vector_to(surface_point).length
                # angle of incidence on the sphere surface should be perpendicular
                cos_angle_incidence = intersection.normal.dot(intersection.ray.direction.normalise())

                self.assertAlmostEqual(distance, 0, self.tolerance_distance,
                                       msg="Ray-curved surface hit point and predicted surface point difference"
                                           " is larger than tolerance.")
                self.assertAlmostEqual(fabs(cos_angle_incidence), 1, self.tolerance_angle,
                                       msg="Angle of incidence differs from perpendicular.")
