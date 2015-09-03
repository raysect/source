# cython: language_level=3

# Copyright (c) 2014, Dr Alex Meakins, Raysect Project
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

from raysect.core.math.affinematrix cimport AffineMatrix
from raysect.core.math.normal cimport new_normal
from raysect.core.math.point cimport new_point, Point
from raysect.core.math.vector cimport new_vector
from raysect.core.classes cimport Material, new_intersection
from raysect.core.acceleration.boundingbox cimport BoundingBox
from libc.math cimport sqrt, fabs
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
DEF CONE = 0
DEF FACE = 1


cdef class Cone(Primitive):
    """
    A cone primitive.

    The cone is defined by a radius and height. It lies along the z-axis
    and extends over the z range [0, height]. The bottom end of the cone is
    capped with a disk forming a closed surface.
    """

    def __init__(self, double radius=0.5, double height=1.0, object parent = None,
                 AffineMatrix transform not None = AffineMatrix(), Material material not None = Material(),
                 unicode name not None= ""):
        """
        Radius is radius of the cone in x-y plane.
        Height of cone is the extent along z-axis [0, height].

        :param radius: Radius of the cone in meters (default = 0.5).
        :param height: Height of the cone in meters (default = 1.0).
        :param parent: Scene-graph parent node or None (default = None).
        :param transform: An AffineMatrix defining the local co-ordinate system relative to the scene-graph parent (default = identity matrix).
        :param material: A Material object defining the cone's material (default = None).
        :param name: A string specifying a user-friendly name for the cylinder (default = "").
        """

        super().__init__(parent, transform, material, name)

        # validate radius and height values
        if radius <= 0.0:
            raise ValueError("Cone radius cannot be less than or equal to zero.")
        if height <= 0.0:
            raise ValueError("Cone height cannot be less than or equal to zero.")
        self._radius = radius
        self._height = height

        # initialise next intersection caching and control attributes
        self._further_intersection = False
        self._next_t = 0.0
        self._cached_origin = None
        self._cached_direction = None
        self._cached_ray = None
        self._cached_type = NO_TYPE

    property radius:
        def __get__(self):
            return self._radius

        def __set__(self, double value):
            if value <= 0.0:
                raise ValueError("Cone radius cannot be less than or equal to zero.")
            self._radius = value

            # the next intersection cache has been invalidated by the geometry change
            self._further_intersection = False
            # any geometry caching in the root node is now invalid, inform root
            self.notify_root()

    property height:
        def __get__(self):
            return self._height

        def __set__(self, double value):
            if value <= 0.0:
                raise ValueError("Cone height cannot be less than or equal to zero.")
            self._height = value

            # the next intersection cache has been invalidated by the geometry change
            self._further_intersection = False
            # any geometry caching in the root node is now invalid, inform root
            self.notify_root()

    def __str__(self):
        """String representation."""
        if self.name == "":
            return "<Cone at " + str(hex(id(self))) + ">"
        else:
            return self.name + " <Cone at " + str(hex(id(self))) + ">"

    @cython.cdivision(True)
    cpdef Intersection hit(self, Ray ray):

        cdef:
            double near_intersection, far_intersection, closest_intersection
            int near_type, far_type, closest_type
            double a, b, c, d, t0, t1, temp, near_point_z, far_point_z
            int f0, f1
            Point near_point, far_point

        # reset the next intersection cache
        self._further_intersection = False

        # convert ray origin and direction to local space
        origin = ray.origin.transform(self.to_local())
        direction = ray.direction.transform(self.to_local())

        radius = self._radius
        height = self._height

        # Compute quadratic cone coefficients
        # based on math from "Physically Based Rendering - 2nd Edition", Elsevier 2010
        k = radius / height
        k = k * k
        a = direction.x * direction.x + direction.y * direction.y - k * direction.z * direction.z
        b = 2 * (direction.x * origin.x + direction.y * origin.y - k * direction.z * (origin.z-height) )
        c = origin.x * origin.x + origin.y * origin.y - k * (origin.z - height) * (origin.z - height)
        d = b * b - 4 * a * c

        # ray misses cone if there are no real roots of the quadratic
        if d < 0:
            return None

        d = sqrt(d)

        # calculate intersections
        temp = 1 / (2.0 * a)
        t0 = -(d + b) * temp
        t1 = (d - b) * temp

        # ensure t0 is always smaller than t1
        if t0 > t1:
            temp = t0
            t0 = t1
            t1 = temp

        near_point_z = origin.z + t0 * direction.z
        far_point_z = origin.z + t1 * direction.z

        # Are both intersections inside the cone height
        if 0 < near_point_z < height and 0 < far_point_z < height:

            # Both intersections are with the cone body
            near_intersection = t0
            near_type = CONE

            far_intersection = t1
            far_type = CONE

        # Is only one of the intersections inside the cone body, therefore other is with flat cone base.
        elif 0 < near_point_z < height or 0 < far_point_z < height:

            if near_point_z < 0 or far_point_z > height:
                # Near intersection is cone base, far is cone surface
                near_intersection = -origin.z / direction.z
                near_type = FACE

                far_intersection = t0
                far_type = CONE

            else:
                # Otherwise near intersection is cone surface, far is cone base.
                near_intersection = t0
                near_type = CONE

                far_intersection = (self._height - origin.z) / direction.z
                far_type = FACE

        # Both intersections are outside the bounding box.
        else:
            return None

        # are there any intersections inside the ray search range?
        if near_intersection > ray.max_distance or far_intersection < 0.0:
            return None

        # identify closest intersection
        if near_intersection >= 0.0:
            closest_intersection = near_intersection
            closest_type = near_type

            # If there is a further intersection, setup values for next calculation.
            if far_intersection <= ray.max_distance:
                self._further_intersection = True
                self._next_t = far_intersection
                self._cached_origin = origin
                self._cached_direction = direction
                self._cached_ray = ray
                self._cached_type = far_type

        elif far_intersection <= ray.max_distance:
            closest_intersection = far_intersection
            closest_type = far_type

        else:
            return None

        return self._generate_intersection(ray, origin, direction, closest_intersection, closest_type)

    cpdef Intersection next_intersection(self):

        if not self._further_intersection:
            return None

        # this is the 2nd and therefore last intersection
        self._further_intersection = False

        return self._generate_intersection(self._cached_ray, self._cached_origin, self._cached_direction, self._next_t, self._cached_type)


    # This function is called twice. Used in hit() and next_intersection()
    @cython.cdivision(True)
    cdef inline Intersection _generate_intersection(self, Ray ray, Point origin, Vector direction, double ray_distance, int type):

        cdef:
            Point hit_point, inside_point, outside_point
            Vector interior_offset
            Normal normal, op
            bint exiting

        # point of surface intersection in local space
        hit_point = new_point(origin.x + ray_distance * direction.x,
                              origin.y + ray_distance * direction.y,
                              origin.z + ray_distance * direction.z)

        # if hit point equals tip, set normal to up.
        # calculate surface normal in local space
        if type == CONE and (hit_point.x == 0.0) and (hit_point.y == 0.0) and (hit_point.z == self.height):
            normal = new_normal(0, 0, 1)

        elif type == CONE:
            # TODO - explore optimisation of this section, two sqrts too many
            # Unit vector that points from origin to hit_point in x-y plane at the base of the cone.
            op = new_normal(hit_point.x, hit_point.y, 0)
            op = op.normalise()
            heighttoradius = self.height/self.radius
            normal = new_normal(op.x * heighttoradius, op.y * heighttoradius, 1/heighttoradius)
            normal = normal.normalise()

        else:
            normal = new_normal(0, 0, -1)

        # displace hit_point away from surface to generate inner and outer points
        inside_point = self._interior_point(hit_point, normal, type)

        # inside_point = new_point(hit_point.x - EPSILON * normal.x,
        #                           hit_point.y - EPSILON * normal.y,
        #                           hit_point.z - EPSILON * normal.z)

        outside_point = new_point(hit_point.x + EPSILON * normal.x,
                                  hit_point.y + EPSILON * normal.y,
                                  hit_point.z + EPSILON * normal.z)

        # is ray exiting surface
        if direction.dot(normal) >= 0.0:
            exiting = True
        else:
            exiting = False

        return new_intersection(ray, ray_distance, self, hit_point, inside_point, outside_point,
                                normal, exiting, self.to_local(), self.to_root())

    @cython.cdivision(True)
    cdef inline Point _interior_point(self, Point hit_point, Normal normal, int type):

        cdef double x, y, z, old_radius, new_radius, scale

        if self.height - hit_point.z < EPSILON:
            print("cone tip")
            print(self.height)
            print(hit_point.z)
            print(self.height - hit_point.z)
            input("...")
            # Avoid tip of cone
            x = 0.0
            y = 0.0
            z = self.height - EPSILON

        elif hit_point.z < EPSILON:
            print("cone base")

            if (hit_point.x**2 + hit_point.y**2) > self.radius**2:
                # Avoid bottom edges of cone
                new_radius = self.radius - EPSILON
                old_radius = sqrt(hit_point.x**2 + hit_point.y**2)

                scale = new_radius/old_radius

                x = scale * hit_point.x
                y = scale * hit_point.y
                z = EPSILON

            else:
                # Avoid base of cone
                x = hit_point.x
                y = hit_point.y
                z = EPSILON

        else:
            # Avoid sides of cone
            old_radius = sqrt(hit_point.x**2 + hit_point.y**2)
            new_radius = old_radius - EPSILON

            scale = new_radius/old_radius
            x = scale * hit_point.x
            y = scale * hit_point.y
            z = hit_point.z

        return new_point(x, y, z)


    cpdef bint contains(self, Point point) except -1:
        cdef:
            double cone_dist, cone_radius, orth_distance

        # convert point to local object space
        point = point.transform(self.to_local())

        # Calculate points' distance along z axis from cone tip
        cone_dist = self.height - point.z

        # reject points that are outside the cone's height (i.e. above the cones' tip or below its base)
        if not 0 <= cone_dist <= self.height:
            return False

        # Calculate the cone radius at that point along the height axis:
        cone_radius = (cone_dist / self.height) * self.radius

        # Calculate the point's orthogonal distance from the axis to compare against the cone radius:
        orth_distance = sqrt(point.x**2 + point.y**2)

        # Points distance from axis must be less than cone radius at that height
        return orth_distance < cone_radius

    cpdef BoundingBox bounding_box(self):

        cdef:
            list points
            Point point
            BoundingBox box

        box = BoundingBox()

        # calculate local bounds
        box.lower = new_point(-self._radius, -self._radius, 0.0)
        box.upper = new_point(self._radius, self._radius, self._height)

        # obtain local space vertices
        points = box.vertices()

        # convert points to world space and build an enclosing world space bounding box
        # a small degree of padding is added to avoid potential numerical accuracy issues
        box = BoundingBox()
        for point in points:

            box.extend(point.transform(self.to_root()), BOX_PADDING)

        return box

