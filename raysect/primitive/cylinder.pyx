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

from raysect.core cimport AffineMatrix3D, new_normal3d, new_point3d, new_vector3d, Material, new_intersection, BoundingBox3D
from raysect.core.math.cython cimport solve_quadratic, swap_double
from libc.math cimport sqrt, fabs
cimport cython

# cython doesn't have a built-in infinity constant, this compiles to +infinity
DEF INFINITY = 1e999

# bounding box is padded by a small amount to avoid numerical accuracy issues
DEF BOX_PADDING = 1e-9

# additional ray distance to avoid re-hitting the same surface point
DEF EPSILON = 1e-9

# object type enumeration
DEF NO_TYPE = -1
DEF CYLINDER = 0
DEF SLAB = 1

# slab face enumeration
DEF NO_FACE = -1
DEF LOWER_FACE = 0
DEF UPPER_FACE = 1


cdef class Cylinder(Primitive):
    """
    A cylinder primitive.

    The cylinder is defined by a radius and height. It lies along the z-axis
    and extends over the z range [0, height]. The ends of the cylinder are
    capped with disks forming a closed surface.

    :param float radius: Radius of the cylinder in meters (default = 0.5).
    :param float height: Height of the cylinder in meters (default = 1.0).
    :param Node parent: Scene-graph parent node or None (default = None).
    :param AffineMatrix3D transform: An AffineMatrix3D defining the local co-ordinate
      system relative to the scene-graph parent (default = identity matrix).
    :param Material material: A Material object defining the cylinder's material (default = None).
    :param str name: A string specifying a user-friendly name for the cylinder (default = "").

    .. code-block:: pycon

        >>> from raysect.core import translate
        >>> from raysect.primitive import Cylinder
        >>> from raysect.optical import World
        >>> from raysect.optical.material import UniformSurfaceEmitter
        >>> from raysect.optical.library.spectra.colours import blue
        >>>
        >>> world = World()
        >>>
        >>> cylinder = Cylinder(0.5, 2.0, parent=world, transform=translate(0, 0, 1),
                                material=UniformSurfaceEmitter(blue), name="blue cylinder")
    """

    def __init__(self, double radius=0.5, double height=1.0, object parent=None,
                 AffineMatrix3D transform=None, Material material=None, str name=None):

        super().__init__(parent, transform, material, name)

        if radius < 0.0:
            raise ValueError("Cylinder radius cannot be less than zero.")

        if height < 0.0:
            raise ValueError("Cylinder height cannot be less than zero.")

        self._radius = radius
        self._height = height

        # initialise next intersection caching and control attributes
        self._further_intersection = False
        self._next_t = 0.0
        self._cached_origin = None
        self._cached_direction = None
        self._cached_ray = None
        self._cached_face = NO_FACE
        self._cached_type = NO_TYPE

    @property
    def radius(self):
        """
        Radius of the cylinder in x-y plane.
        """
        return self._radius

    @radius.setter
    def radius(self, double value):
        if value < 0.0:
            raise ValueError("Cylinder radius cannot be less than zero.")
        self._radius = value

        # the next intersection cache has been invalidated by the geometry change
        self._further_intersection = False

        # any geometry caching in the root node is now invalid, inform root
        self.notify_geometry_change()

    @property
    def height(self):
        """
        Extent of the cylinder along the z-axis.
        """
        return self._height

    @height.setter
    def height(self, double value):
        if value < 0.0:
            raise ValueError("Cylinder height cannot be less than zero.")
        self._height = value

        # the next intersection cache has been invalidated by the geometry change
        self._further_intersection = False

        # any geometry caching in the root node is now invalid, inform root
        self.notify_geometry_change()

    @cython.cdivision(True)
    cpdef Intersection hit(self, Ray ray):

        cdef:
            double near_intersection, far_intersection, closest_intersection
            int near_face, far_face, closest_face
            int near_type, far_type, closest_type
            double a, b, c, t0, t1
            int f0, f1

        # reset the next intersection cache
        self._further_intersection = False

        # convert ray origin and direction to local space
        origin = ray.origin.transform(self.to_local())
        direction = ray.direction.transform(self.to_local())

        # check ray intersects infinite cylinder and obtain intersections
        # is ray parallel to cylinder surface?
        if direction.x == 0 and direction.y == 0:

            if self._inside_cylinder(origin):

                near_intersection = -INFINITY
                near_type = NO_TYPE
                near_face = NO_FACE

                far_intersection = INFINITY
                far_type = NO_TYPE
                far_face = NO_FACE

            else:

                # no ray cylinder intersection
                return None

        else:

            # coefficients of quadratic equation
            a = direction.x * direction.x + direction.y * direction.y
            b = 2.0 * (direction.x * origin.x + direction.y * origin.y)
            c = origin.x * origin.x + origin.y * origin.y - self._radius * self._radius

            # calculate intersection distances by solving the quadratic equation
            # ray misses if there are no real roots of the quadratic
            if not solve_quadratic(a, b, c, &t0, &t1):
                return None

            # ensure t0 is always smaller than t1
            if t0 > t1:
                swap_double(&t0, &t1)

            # set intersection parameters
            near_intersection = t0
            near_type = CYLINDER
            near_face = NO_FACE

            far_intersection = t1
            far_type = CYLINDER
            far_face = NO_FACE

        # union slab with the cylinder
        # slab contributes no intersections if the ray is parallel to the slab surfaces
        if direction.z != 0.0:

            # calculate intersections with slab planes
            temp = 1.0 / direction.z

            if direction.z > 0:

                # calculate length along ray path of intersections
                t0 = -origin.z * temp
                t1 = (self._height - origin.z) * temp

                f0 = LOWER_FACE
                f1 = UPPER_FACE

            else:

                # calculate length along ray path of intersections
                t0 = (self._height - origin.z) * temp
                t1 = -origin.z * temp

                f0 = UPPER_FACE
                f1 = LOWER_FACE

            # calculate intersection overlap
            if t0 > near_intersection:

                near_intersection = t0
                near_face = f0
                near_type = SLAB

            if t1 < far_intersection:

                far_intersection = t1
                far_face = f1
                far_type = SLAB

        # does ray intersect cylinder?
        if near_intersection > far_intersection:
            return None

        # are there any intersections inside the ray search range?
        if near_intersection > ray.max_distance or far_intersection < 0.0:
            return None

        # identify closest intersection
        if near_intersection >= 0.0:
            closest_intersection = near_intersection
            closest_face = near_face
            closest_type = near_type

            if far_intersection <= ray.max_distance:
                self._further_intersection = True
                self._next_t = far_intersection
                self._cached_origin = origin
                self._cached_direction = direction
                self._cached_ray = ray
                self._cached_face = far_face
                self._cached_type = far_type

        elif far_intersection <= ray.max_distance:
            closest_intersection = far_intersection
            closest_face = far_face
            closest_type = far_type
        else:
            return None

        return self._generate_intersection(ray, origin, direction, closest_intersection, closest_face, closest_type)

    cpdef Intersection next_intersection(self):

        if not self._further_intersection:
            return None

        # this is the 2nd and therefore last intersection
        self._further_intersection = False
        return self._generate_intersection(self._cached_ray, self._cached_origin, self._cached_direction, self._next_t, self._cached_face, self._cached_type)

    cdef Intersection _generate_intersection(self, Ray ray, Point3D origin, Vector3D direction, double ray_distance, int face, int type):

        cdef:
            Point3D hit_point, inside_point, outside_point
            Vector3D interior_offset
            Normal3D normal
            bint exiting

        # point of surface intersection in local space
        hit_point = new_point3d(origin.x + ray_distance * direction.x,
                                origin.y + ray_distance * direction.y,
                                origin.z + ray_distance * direction.z)

        # calculate surface normal in local space
        if type == CYLINDER:
            normal = new_normal3d(hit_point.x, hit_point.y, 0)
            normal = normal.normalise()
        else:
            if face == LOWER_FACE:
                normal = new_normal3d(0, 0, -1)
            else:
                normal = new_normal3d(0, 0, 1)

        # displace hit_point away from surface to generate inner and outer points
        interior_offset = self._interior_offset(hit_point, normal, type)

        inside_point = new_point3d(hit_point.x + interior_offset.x,
                                   hit_point.y + interior_offset.y,
                                   hit_point.z + interior_offset.z)

        outside_point = new_point3d(hit_point.x + EPSILON * normal.x,
                                    hit_point.y + EPSILON * normal.y,
                                    hit_point.z + EPSILON * normal.z)

        # is ray exiting surface
        exiting = direction.dot(normal) >= 0.0

        return new_intersection(ray, ray_distance, self, hit_point, inside_point, outside_point,
                                normal, exiting, self.to_local(), self.to_root())

    @cython.cdivision(True)
    cdef Vector3D _interior_offset(self, Point3D hit_point, Normal3D normal, int type):

        cdef double x, y, z, length

        # shift away from cylinder surface
        if type == CYLINDER:
            x = -EPSILON * normal.x
            y = -EPSILON * normal.y
        else:
            x = 0
            y = 0
            if hit_point.x != 0.0 and hit_point.y != 0.0:
                length = sqrt(hit_point.x * hit_point.x + hit_point.y * hit_point.y)
                if (length - self._radius) < EPSILON:
                    length = 1.0 / length
                    x = -EPSILON * length * hit_point.x
                    y = -EPSILON * length * hit_point.y

        # shift away from slab surface
        if fabs(hit_point.z) < EPSILON:
            z = EPSILON
        elif fabs(hit_point.z - self._height) < EPSILON:
            z = -EPSILON
        else:
            z = 0

        return new_vector3d(x, y, z)

    cpdef bint contains(self, Point3D point) except -1:

        # convert point to local object space
        point = point.transform(self.to_local())
        return self._inside_slab(point) and self._inside_cylinder(point)

    cdef bint _inside_cylinder(self, Point3D point):

        # is the point inside the cylinder radius
        return (point.x * point.x + point.y * point.y) <= (self._radius * self._radius)

    cdef bint _inside_slab(self, Point3D point):

        # first check point is within the cylinder upper and lower bounds
        return 0.0 <= point.z <= self._height

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
        return Cylinder(self._radius, self._height, parent, transform, material, name)