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

from raysect.core cimport new_point3d, Point3D, new_normal3d, AffineMatrix3D, Material, new_intersection, BoundingBox3D
from raysect.core.math.cython cimport solve_quadratic, swap_double, swap_int
from libc.math cimport sqrt
cimport cython

# cython doesn't have a built-in infinity constant, this compiles to +infinity
DEF INFINITY = 1e999

# bounding box is padded by a small amount to avoid numerical accuracy issues
DEF BOX_PADDING = 1e-9

# TODO - Perhaps should be calculated based on primitive scale
# additional ray distance to avoid re-hitting the same surface point
DEF EPSILON = 1e-9

# object type enumeration
DEF NO_TYPE = -1
DEF PARABOLA = 0
DEF BASE = 1


cdef class Parabola(Primitive):
    """
    A parabola primitive.

    The parabola is defined by a radius and height. It lies along the z-axis
    and extends over the z range [0, height]. The base of the parabola is
    capped with a disk forming a closed surface. The base of the parabola lies
    on the x-y plane, the parabola vertex (tip) lies at z=height.

    :param float radius: Radius of the parabola in meters (default = 0.5).
    :param float height: Height of the parabola in meters (default = 1.0).
    :param Node parent: Scene-graph parent node or None (default = None).
    :param AffineMatrix3D transform: An AffineMatrix3D defining the local co-ordinate system relative to the scene-graph parent (default = identity matrix).
    :param Material material: A Material object defining the parabola's material (default = None).
    :param str name: A string specifying a user-friendly name for the parabola (default = "").

    .. code-block:: pycon

        >>> from raysect.core import translate
        >>> from raysect.primitive import Parabola
        >>> from raysect.optical import World
        >>> from raysect.optical.material import UniformSurfaceEmitter
        >>> from raysect.optical.library.spectra.colours import yellow
        >>>
        >>> world = World()
        >>>
        >>> parabola = Parabola(0.5, 2.0, parent=world, transform=translate(0, 0, 1),
                                material=UniformSurfaceEmitter(yellow), name="yellow parabola")
    """

    def __init__(self, double radius=0.5, double height=1.0, object parent=None,
                 AffineMatrix3D transform=None, Material material=None, str name=None):

        super().__init__(parent, transform, material, name)

        # validate radius and height values
        if radius <= 0.0:
            raise ValueError("Parabola radius cannot be less than or equal to zero.")
        if height <= 0.0:
            raise ValueError("Parabola height cannot be less than or equal to zero.")
        self._radius = radius
        self._height = height

        # initialise next intersection caching and control attributes
        self._further_intersection = False
        self._next_t = 0.0
        self._cached_origin = None
        self._cached_direction = None
        self._cached_ray = None
        self._cached_type = NO_TYPE

    @property
    def radius(self):
        """
        Radius of the parabola base in x-y plane.
        """
        return self._radius

    @radius.setter
    def radius(self, double value):
        if value <= 0.0:
            raise ValueError("Parabola radius cannot be less than or equal to zero.")
        self._radius = value

        # the next intersection cache has been invalidated by the geometry change
        self._further_intersection = False

        # any geometry caching in the root node is now invalid, inform root
        self.notify_geometry_change()

    @property
    def height(self):
        """
        The parabola's extent along the z-axis [0, height].
        """
        return self._height

    @height.setter
    def height(self, double value):
        if value <= 0.0:
            raise ValueError("Parabola height cannot be less than or equal to zero.")
        self._height = value

        # the next intersection cache has been invalidated by the geometry change
        self._further_intersection = False

        # any geometry caching in the root node is now invalid, inform root
        self.notify_geometry_change()

    @cython.cdivision(True)
    cpdef Intersection hit(self, Ray ray):

        cdef:
            Point3D origin
            Vector3D direction
            double radius, height
            double a, b, c, k, t0, t1, t0_z, t1_z
            int t0_type, t1_type
            bint t0_outside, t1_outside
            double closest_intersection
            int closest_type

        # reset the next intersection cache
        self._further_intersection = False

        # convert ray origin and direction to local space
        origin = ray.origin.transform(self.to_local())
        direction = ray.direction.transform(self.to_local())

        radius = self._radius
        height = self._height

        # Compute quadratic parabola coefficients
        # based on math from "Physically Based Rendering - 2nd Edition", Elsevier 2010
        k = height / (radius * radius)
        a = k * (direction.x * direction.x + direction.y * direction.y)
        b = 2 * k * (direction.x * origin.x + direction.y * origin.y) + direction.z
        c = k * (origin.x * origin.x + origin.y * origin.y) - (height - origin.z)

        # calculate intersection distances by solving the quadratic equation
        # ray misses if there are no real roots of the quadratic
        if not solve_quadratic(a, b, c, &t0, &t1):
            return None

        if t0 == t1:

            # ray intersects the tip of the parabola
            t0 = -b / (2.0 * a)
            t0_type = PARABOLA

            # does ray also intersect the base?
            k = -origin.z / direction.z
            r2 = (origin.x + k * direction.x)**2 + (origin.y + k * direction.y**2)
            if r2 <= (self._radius * self._radius):
                t1 = k
                t1_type = BASE
            else:
                t1 = t0
                t1_type = t0_type

        else:

            # calculate z height of intersection points
            t0_z = origin.z + t0 * direction.z
            t1_z = origin.z + t1 * direction.z

            t0_outside = t0_z < 0
            t1_outside = t1_z < 0

            if t0_outside and t1_outside:

                # ray intersects parabola outside of height range
                return None

            elif not t0_outside and t1_outside:

                # t0 is inside, t1 is outside
                t0_type = PARABOLA

                t1 = -origin.z / direction.z
                t1_type = BASE

            elif t0_outside and not t1_outside:

                # t0 is outside, t1 is inside
                t0_type = BASE
                t0 = -origin.z / direction.z

                t1_type = PARABOLA

            else:

                # both intersections are valid and within the parabola body
                t0_type = PARABOLA
                t1_type = PARABOLA

        # ensure t0 is always smaller (closer) than t1
        if t0 > t1:
            swap_double(&t0, &t1)
            swap_int(&t0_type, &t1_type)

        # are there any intersections inside the ray search range?
        if t0 > ray.max_distance or t1 < 0.0:
            return None

        # identify closest intersection
        if t0 >= 0.0:
            closest_intersection = t0
            closest_type = t0_type

            # If there is a further intersection, setup values for next calculation.
            if t1 <= ray.max_distance:
                self._further_intersection = True
                self._next_t = t1
                self._cached_origin = origin
                self._cached_direction = direction
                self._cached_ray = ray
                self._cached_type = t1_type

        elif t1 <= ray.max_distance:
            closest_intersection = t1
            closest_type = t1_type

        else:
            return None

        return self._generate_intersection(ray, origin, direction, closest_intersection, closest_type)

    cpdef Intersection next_intersection(self):

        if not self._further_intersection:
            return None

        # this is the 2nd and therefore last intersection
        self._further_intersection = False

        return self._generate_intersection(self._cached_ray, self._cached_origin, self._cached_direction, self._next_t, self._cached_type)

    @cython.cdivision(True)
    cdef Intersection _generate_intersection(self, Ray ray, Point3D origin, Vector3D direction, double ray_distance, int type):

        cdef:
            Point3D hit_point, inside_point, outside_point
            Normal3D normal
            double k
            bint exiting

        # point of surface intersection in local space
        hit_point = new_point3d(
            origin.x + ray_distance * direction.x,
            origin.y + ray_distance * direction.y,
            origin.z + ray_distance * direction.z
        )

        # calculate surface normal in local space
        if type == BASE:

            # parabola base
            normal = new_normal3d(0, 0, -1)

        else:

            # in implicit form F(x,y,z) = z - f(x,y) = 0, normal is given by grad(F(x, y, z))
            k = 2 * self._height / (self._radius * self._radius)
            normal = new_normal3d(k * hit_point.x, k * hit_point.y, 1)
            normal = normal.normalise()

        # displace hit_point away from surface to generate inner and outer points
        inside_point = self._interior_point(hit_point, normal, type)

        outside_point = new_point3d(
            hit_point.x + EPSILON * normal.x,
            hit_point.y + EPSILON * normal.y,
            hit_point.z + EPSILON * normal.z
        )

        # is ray exiting surface
        exiting = direction.dot(normal) >= 0.0

        return new_intersection(ray, ray_distance, self, hit_point, inside_point, outside_point,
                                normal, exiting, self.to_local(), self.to_root())

    @cython.cdivision(True)
    cdef Point3D _interior_point(self, Point3D hit_point, Normal3D normal, int type):

        cdef:
            double x, y, z
            double scale
            double inner_radius, hit_radius_sqr

        x = hit_point.x
        y = hit_point.y
        z = hit_point.z

        # todo: fix next time through the code as this is not 100% robust, but probability of inner point leaving volume is however vanishingly low
        inner_radius = self._radius - EPSILON
        hit_radius_sqr = hit_point.x * hit_point.x + hit_point.y * hit_point.y
        if hit_radius_sqr > (inner_radius * inner_radius):
            scale = inner_radius / sqrt(hit_radius_sqr)
            x = scale * hit_point.x
            y = scale * hit_point.y

        if hit_point.z < EPSILON:
            z = EPSILON

        else:
            x = hit_point.x - normal.x * EPSILON
            y = hit_point.y - normal.y * EPSILON
            z = hit_point.z - normal.z * EPSILON

        return new_point3d(x, y, z)

    @cython.cdivision(True)
    cpdef bint contains(self, Point3D point) except -1:

        cdef:
            double parabola_radius, point_radius

        # convert point to local object space
        point = point.transform(self.to_local())

        # reject points that are outside the parabola's height range
        if point.z < 0 or point.z > self._height:
            return False

        # calculate the parabola radius at that point along the height axis:
        parabola_radius = self._radius * sqrt((self._height - point.z) / self._height)

        # calculate the point's orthogonal distance from the axis to compare against the parabola radius:
        point_radius = sqrt(point.x * point.x + point.y * point.y)

        # Points distance from axis must be less than parabola radius at that height
        return point_radius <= parabola_radius

    cpdef BoundingBox3D bounding_box(self):

        cdef:
            list points
            Point3D point
            BoundingBox3D box

        box = BoundingBox3D()

        # calculate local bounds
        box.lower = new_point3d(-self._radius, -self._radius, 0.0)
        box.upper = new_point3d(self._radius, self._radius, self._height)

        # obtain local space vertices
        points = box.vertices()

        # convert points to world space and build an enclosing world space bounding box
        # a small degree of padding is added to avoid potential numerical accuracy issues
        box = BoundingBox3D()
        for point in points:
            box.extend(point.transform(self.to_root()), BOX_PADDING)

        return box

    cpdef object instance(self, object parent=None, AffineMatrix3D transform=None, Material material=None, str name=None):
        return Parabola(self._radius, self._height, parent, transform, material, name)