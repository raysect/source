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

# additional ray distance to avoid re-hitting the same surface point
DEF EPSILON = 1e-9

# object type enumeration
DEF NO_TYPE = -1
DEF CONE = 0
DEF BASE = 1


cdef class Cone(Primitive):
    """
    A cone primitive.

    The cone is defined by a radius and height. It lies along the z-axis
    and extends over the z range [0, height]. The tip of the cone lies at
    z = height. The base of the cone sits on the x-y plane and is capped
    with a disk, forming a closed surface.

    :param float radius: Radius of the cone in meters in x-y plane (default = 0.5).
    :param float height: Height of the cone in meters (default = 1.0).
    :param Node parent: Scene-graph parent node or None (default = None).
    :param AffineMatrix3D transform: An AffineMatrix3D defining the local co-ordinate
      system relative to the scene-graph parent (default = identity matrix).
    :param Material material: A Material object defining the cone's material (default = None).
    :param str name: A string specifying a user-friendly name for the cone (default = "").

    .. code-block:: pycon

        >>> from raysect.core import translate
        >>> from raysect.primitive import Box
        >>> from raysect.optical import World
        >>> from raysect.optical.material import UniformSurfaceEmitter
        >>> from raysect.optical.library.spectra.colours import green
        >>>
        >>> world = World()
        >>>
        >>> cone = Cone(0.5, 2.0, parent=world, transform=translate(0, 0, 1),
                        material=UniformSurfaceEmitter(green), name="green cone")
    """

    def __init__(self, double radius=0.5, double height=1.0, object parent=None,
                 AffineMatrix3D transform=None, Material material=None,
                 str name=None):

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

    @property
    def radius(self):
        """
        The radius of the cone base in the x-y plane
        """
        return self._radius

    @radius.setter
    def radius(self, double value):
        if value <= 0.0:
            raise ValueError("Cone radius cannot be less than or equal to zero.")
        self._radius = value

        # the next intersection cache has been invalidated by the geometry change
        self._further_intersection = False

        # any geometry caching in the root node is now invalid, inform root
        self.notify_geometry_change()

    @property
    def height(self):
        """
        The extend of the cone along the z-axis
        """
        return self._height

    @height.setter
    def height(self, double value):
        if value <= 0.0:
            raise ValueError("Cone height cannot be less than or equal to zero.")
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
            double a, b, c, k, t0, t1, t0_z, t1_z, temp_d, r2
            int t0_type, t1_type, temp_i
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

        # Compute quadratic cone coefficients
        # based on math from "Physically Based Rendering - 2nd Edition", Elsevier 2010
        k = radius / height
        k = k * k
        a = direction.x * direction.x + direction.y * direction.y - k * direction.z * direction.z
        b = 2 * (direction.x * origin.x + direction.y * origin.y - k * direction.z * (origin.z - height))
        c = origin.x * origin.x + origin.y * origin.y - k * (origin.z - height) * (origin.z - height)

        # calculate intersection distances by solving the quadratic equation
        # ray misses if there are no real roots of the quadratic
        if not solve_quadratic(a, b, c, &t0, &t1):
            return None

        if t0 == t1:

            # ray intersects the tip of the cone
            t0 = -b / (2.0 * a)
            t0_type = CONE

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

            # ray hits full cone quadratic twice

            # calculate z height of intersection points
            t0_z = origin.z + t0 * direction.z
            t1_z = origin.z + t1 * direction.z

            t0_outside = t0_z < 0 or t0_z > height
            t1_outside = t1_z < 0 or t1_z > height

            if t0_outside and t1_outside:

                # ray intersects cone outside of height range
                return None

            elif not t0_outside and t1_outside:

                # t0 is in range, t1 is outside
                t0_type = CONE

                t1 = -origin.z / direction.z
                t1_type = BASE

            elif t0_outside and not t1_outside:

                # t0 is outside range, t1 is inside
                t0_type = BASE
                t0 = -origin.z / direction.z

                t1_type = CONE

            else:

                # both intersections are valid and within the cone body
                t0_type = CONE
                t1_type = CONE

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
            double a, b
            bint exiting

        # point of surface intersection in local space
        hit_point = new_point3d(
            origin.x + ray_distance * direction.x,
            origin.y + ray_distance * direction.y,
            origin.z + ray_distance * direction.z
        )

        # calculate surface normal in local space
        if type == BASE:

            # cone base
            normal = new_normal3d(0, 0, -1)

        elif type == CONE and hit_point.z >= self._height:

            # cone tip
            normal = new_normal3d(0, 0, 1)

        else:

            # cone body
            a = hit_point.y / hit_point.x
            b = self._height / sqrt(1 + a*a)
            b = -b if hit_point.x < 0 else b
            normal = new_normal3d(b, b*a, self._radius)
            normal = normal.normalise()

        # displace hit_point away from surface to generate inner and outer points
        inside_point = self._interior_point(hit_point, normal)

        outside_point = new_point3d(
            hit_point.x + EPSILON * normal.x,
            hit_point.y + EPSILON * normal.y,
            hit_point.z + EPSILON * normal.z
        )

        # is ray exiting surface
        exiting = direction.dot(normal) >= 0.0

        return new_intersection(
            ray, ray_distance, self, hit_point, inside_point, outside_point,
            normal, exiting, self.to_local(), self.to_root()
        )

    @cython.cdivision(True)
    cdef Point3D _interior_point(self, Point3D hit_point, Normal3D normal):

        cdef:
            double x, y, z, k
            double inner_height, inner_radius, scale

        x = hit_point.x - EPSILON * normal.x
        y = hit_point.y - EPSILON * normal.y
        z = hit_point.z - EPSILON * normal.z
        k = self._radius / self._height

        # If the hit point is near the tip of the cone, we need to ensure that the inside point is
        # shifted down so that the inside point is at least an epsilon away from all the cone
        # surfaces. This is done by identifying a new cone, inset from the outer cone by 1 epsilon
        # If the inside point lies is above the tip of the inner cone, it is moved to the tip of the
        # inner cone.
        inner_height = self._height - EPSILON * sqrt(1 + k*k) / k
        # assert inner_height >= EPSILON, 'Numerical failure in Cone, this occurs if the cone has an extreme aspect ratio or ray distances are too large compared with the cone size.'
        if z > inner_height:
            return new_point3d(0, 0, inner_height)

        # is interior point too close to or below the base, if so shift up following the cone surface
        if z < EPSILON:

            # calculate the base radius for a cone inset from the base and sides of the cone by epsilon
            inner_radius = k * (self._height - EPSILON) - EPSILON * sqrt(1 + k*k)
            # assert inner_radius >= 0, 'Numerical failure in Cone, this occurs if the cone has an extreme aspect ratio or ray distances are too large compared with the cone size.'

            # rescale radius of inner point to lie on edge of inner cone base and shift point up by one epsilon
            scale = inner_radius / sqrt(hit_point.x*hit_point.x + hit_point.y*hit_point.y)
            return new_point3d(scale * hit_point.x, scale * hit_point.y, EPSILON)

        return new_point3d(x, y, z)

    @cython.cdivision(True)
    cpdef bint contains(self, Point3D point) except -1:

        cdef:
            double cone_radius_sqr, point_radius_sqr

        # convert point to local object space
        point = point.transform(self.to_local())

        # reject points that are outside the cone's height (i.e. above the cones' tip or below its base)
        if point.z < 0 or point.z > self._height:
            return False

        # calculate the point's orthogonal distance from the axis to compare against the cone radius:
        point_radius_sqr = point.x*point.x + point.y*point.y

        # calculate the cone radius at that point along the height axis:
        cone_radius_sqr = (self._height - point.z) * self._radius / self._height
        cone_radius_sqr *= cone_radius_sqr

        # Points distance from axis must be less than cone radius at that height (compare squared radii to avoid costly sqrt)
        return point_radius_sqr <= cone_radius_sqr

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
        return Cone(self._radius, self._height, parent, transform, material, name)