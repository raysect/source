import unittest

from raysect.core import translate, Point3D, Vector3D
from raysect.primitive.lens.spherical import BiConvex, BiConcave, PlanoConvex, PlanoConcave, Meniscus

import numpy as np
from math import sqrt, cos, sin, pi


class TestSphericalLense(unittest.TestCase):
    """
    The idea is to create a set of points making an inside and outside envelope of the lens surface
    and test if they are correctly contained by the lens primitive.
    """

    pad_constant = 1e-6  # offset ratio to construct inside and outside points

    # points to test lens surfaces at
    test_radius_ratios = np.linspace(1, pad_constant, 10,
                                     endpoint=True)  # ratio of the lens radius (x-y plane) to test at
    test_azimuths = np.linspace(0, 2 * pi, 20)  # angles in xy plane to test the surface at
    test_barrel_z_points = np.linspace(0, 1, endpoint=True)  # points along lens barrel

    def test_biconvex(self):

        # define lens properties and initialise lens
        diameter = 75
        center_thickness = 10
        front_curvature = 210
        back_curvature = 200
        lens = BiConvex(diameter, center_thickness, front_curvature, back_curvature)

        # calculate coordinates of center of curvatures of the lens front and back surfaces
        center_back = Point3D(0, 0, back_curvature)
        center_front = Point3D(0, 0, center_thickness - front_curvature)

        # check lens front and back surface by inside and outside envelopes
        self._check_convex_surface(back_curvature, lens, center_back, "smaller")
        self._check_concave_surface(front_curvature, lens, center_front, "smaller")

        # calculate range of lens barrel surface
        radius2 = (diameter / 2) ** 2
        back_thickness = back_curvature - sqrt(back_curvature ** 2 - radius2)
        front_thickness = front_curvature - sqrt(front_curvature ** 2 - radius2)

        # check lens barrel surface by inside and outside envelope
        self._check_barrel_surface(back_thickness, center_thickness - front_thickness, lens)

    def test_biconcave(self):

        # define lens properties and initialise lens
        diameter = 75
        center_thickness = 10
        front_curvature = 210
        back_curvature = 200
        lens = BiConcave(diameter, center_thickness, front_curvature, back_curvature)

        # calculate coordinates of center of curvatures of the lens front and back surfaces
        center_back = Point3D(0, 0, -back_curvature)
        center_front = Point3D(0, 0, center_thickness + front_curvature)  # center of the front curvature

        # check lens front and back surface by inside and outside envelopes
        self._check_concave_surface(back_curvature, lens, center_back, "larger")
        self._check_convex_surface(front_curvature, lens, center_front, "larger")

        # calculate range of lens barrel surface
        radius2 = (diameter / 2) ** 2
        back_thickness = back_curvature - sqrt(back_curvature ** 2 - radius2)
        front_thickness = front_curvature - sqrt(front_curvature ** 2 - radius2)

        self._check_barrel_surface(-back_thickness, center_thickness + front_thickness, lens)

    def test_planoconvex(self):

        # define lens properties and initialise lens
        diameter = 75
        centre_thickness = 10
        curvature = 200
        lens = PlanoConvex(diameter, centre_thickness, curvature)

        # calculate coordinates of center of curvature of the lens front surface
        center_front = Point3D(0, 0, centre_thickness - curvature)

        # check lens front and back surface by inside and outside envelopes
        self._check_concave_surface(curvature, lens, center_front, "smaller")
        self._check_plane_surface(0, lens, "larger")

        # calculate range of lens barrel surface
        radius2 = (diameter / 2) ** 2
        front_thickness = curvature - sqrt(curvature ** 2 - radius2)
        self._check_barrel_surface(0, centre_thickness - front_thickness, lens)

    def test_planoconcave(self):

        # define lens properties and initialise lens
        diameter = 75
        centre_thickness = 10
        curvature = 200
        lens = PlanoConcave(diameter, centre_thickness, curvature)

        # calculate coordinates of center of curvature of the lens front surface
        center_front = Point3D(0, 0, centre_thickness + curvature)

        # check lens front and back surface by inside and outside envelopes
        self._check_convex_surface(curvature, lens, center_front, "larger")
        self._check_plane_surface(0, lens, "larger")

        # calculate range of lens barrel surface
        radius2 = (diameter / 2) ** 2
        front_thickness = curvature - sqrt(curvature ** 2 - radius2)
        self._check_barrel_surface(0, centre_thickness + front_thickness, lens)

    def test_meniscus(self):

        # define lens properties and initialise lens
        diameter = 75
        center_thickness = 10
        front_curvature = 210
        back_curvature = 200

        lens = Meniscus(diameter, center_thickness, front_curvature, back_curvature)

        # calculate coordinates of center of curvatures of the lens front and back surfaces
        center_back = Point3D(0, 0, -back_curvature)
        center_front = Point3D(0, 0, center_thickness - front_curvature)

        # check lens front and back surface by inside and outside envelopes
        self._check_concave_surface(back_curvature, lens, center_back, "larger")
        self._check_concave_surface(front_curvature, lens, center_front, "smaller")

        # calculate range of lens barrel surface
        radius2 = (diameter / 2) ** 2
        back_thickness = back_curvature - sqrt(back_curvature ** 2 - radius2)
        front_thickness = front_curvature - sqrt(front_curvature ** 2 - radius2)
        self._check_barrel_surface(-back_thickness, center_thickness - front_thickness, lens)

    def _check_convex_surface(self, curvature, lens, center_curvature, inside_direction):
        """
        Concave surface is the lens surface which has z coordinate of the center of curvature larger than the
        intersection of the surface with z axis.
        :return:
        """

        if inside_direction == 'smaller':
            sign = -1
        elif inside_direction == 'larger':
            sign = 1
        else:
            raise ValueError("insde_direction can be either 'larger' or 'smaller'.")

        lens_radius = lens.diameter / 2
        curvature2 = curvature ** 2

        # generate test-point on the lens front/back surface
        test_radii = self.test_radius_ratios * (lens_radius - self.pad_constant)

        # test back surface
        for _, radius in enumerate(test_radii):
            z = sqrt(curvature2 - radius ** 2)
            for _, ta in enumerate(self.test_azimuths):
                # calculate position vector pointing from the curvature center to the surface point
                x = radius * cos(ta)
                y = radius * sin(ta)
                position_vector = Vector3D(x, y, -z).normalise()

                # construct inside point by offsetting the surface point and test it
                position_inside = position_vector * (curvature + (sign * self.pad_constant))
                inside = Point3D(center_curvature.x + position_inside.x,
                                 center_curvature.y + position_inside.y,
                                 center_curvature.z + position_inside.z)

                self.assertTrue(lens.contains(inside), msg="Lens inside point is not contained by the lens primitive.")

                # construct inside point by offsetting the surface point and test it
                position_outside = position_vector * (curvature - (sign * self.pad_constant))
                outside = Point3D(center_curvature.x + position_outside.x,
                                  center_curvature.y + position_outside.y,
                                  center_curvature.z + position_outside.z)
                self.assertFalse(lens.contains(outside), msg="Lens outside point of back surface is contained"
                                                             " by the lens primitive. {}".format(position_outside))

    def _check_concave_surface(self, curvature, lens, center_curvature, inside_direction):
        """
        Concave surface is the lens surface which has z coordinate of the center of curvature smaller than the
        intersection of the surface with z axis.
        :return:
        """

        if inside_direction == 'smaller':
            sign = -1
        elif inside_direction == 'larger':
            sign = 1
        else:
            raise ValueError("insde_direction can be either 'larger' or 'smaller'.")

        lens_radius = lens.diameter / 2
        curvature2 = curvature ** 2

        # generate test-point on the lens front/back surface
        test_radii = self.test_radius_ratios * (lens_radius - self.pad_constant)

        # test front surface
        for _, radius in enumerate(test_radii):
            z = sqrt(curvature2 - radius ** 2)
            for _, ta in enumerate(self.test_azimuths):
                # calculate position vector pointing from the curvature center to the surface point
                x = radius * cos(ta)
                y = radius * sin(ta)
                position_vector = Vector3D(x, y, z).normalise()

                # construct inside point by offsetting the surface point and test it
                position_inside = position_vector * (curvature + (sign * self.pad_constant))
                inside = Point3D(center_curvature.x + position_inside.x,
                                 center_curvature.y + position_inside.y,
                                 center_curvature.z + position_inside.z)

                self.assertTrue(lens.contains(inside), msg="Lens inside point is not contained by the lens primitive.")

                # construct outside point by offsetting the surface point and test it
                position_outside = position_vector * (curvature - (sign * self.pad_constant))
                outside = Point3D(center_curvature.x + position_outside.x,
                                  center_curvature.y + position_outside.y,
                                  center_curvature.z + position_outside.z)
                self.assertFalse(lens.contains(outside), msg="Lens outside point is contained by the lens primitive.")

    def _check_barrel_surface(self, z_start, z_stop, lens):
        """
        Cylindrical surface with radius equal to lens radius stretching in between front and back lens edges
        """

        lens_radius = lens.diameter / 2

        # calculate z range for barrel points
        start = z_start + self.pad_constant
        stop = z_stop - self.pad_constant
        barrel_z = start + self.test_barrel_z_points * (stop - start)

        for _, z in enumerate(barrel_z):
            for _, ta in enumerate(self.test_azimuths):
                # construct and test inside point
                x = (lens_radius - self.pad_constant) * cos(ta)
                y = (lens_radius - self.pad_constant) * sin(ta)
                inside = Point3D(x, y, z)
                self.assertTrue(lens.contains(inside), msg="Lens inside point is not contained by the lens primitive.")

                # construct and test outside point
                x = (lens_radius + self.pad_constant) * cos(ta)
                y = (lens_radius + self.pad_constant) * sin(ta)
                outside = Point3D(x, y, z)
                self.assertFalse(lens.contains(outside), msg="Lens outside point is contained by the lens primitive.")

    def _check_plane_surface(self, z_plane, lens, inside_direction):

        if inside_direction == 'smaller':
            sign = -1
        elif inside_direction == 'larger':
            sign = 1
        else:
            raise ValueError('Padding can be either "positive" or "negative" ')

        # generate test-point on the lens front/back surface
        test_radii = self.test_radius_ratios * lens.diameter / 2

        # test inside points
        for _, radius in enumerate(test_radii):
            for _, ta in enumerate(self.test_azimuths):
                x = (radius - self.pad_constant) * cos(ta)
                y = (radius - self.pad_constant) * sin(ta)
                z = z_plane + (sign * self.pad_constant)

                inside = Point3D(x, y, z)
                self.assertTrue(lens.contains(inside), msg="Lens inside point is not contained by the lens primitive.")

        # test outside points
        for _, radius in enumerate(test_radii):
            for _, ta in enumerate(self.test_azimuths):
                x = (radius + self.pad_constant) * cos(ta)
                y = (radius + self.pad_constant) * sin(ta)
                z = z_plane - (sign * self.pad_constant)

                inside = Point3D(x, y, z)
                self.assertFalse(lens.contains(inside), msg="Lens outside point is contained by the lens primitive.")
