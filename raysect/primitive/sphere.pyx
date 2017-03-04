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


from raysect.core cimport Material, new_intersection, BoundingBox3D, new_point3d, new_normal3d, Normal3D, AffineMatrix3D
from libc.math cimport sqrt
cimport cython


# bounding box is padded by a small amount to avoid numerical accuracy issues
DEF BOX_PADDING = 1e-9

# additional ray distance to avoid re-hitting the same surface point
DEF EPSILON = 1e-9


cdef class Sphere(Primitive):
    """
    A sphere primitive.

    The sphere is centered at the origin of the local co-ordinate system.

    :param float radius: Radius of the sphere in meters (default = 0.5).
    :param Node parent: Scene-graph parent node or None (default = None).
    :param AffineMatrix3D transform: An AffineMatrix3D defining the local co-ordinate system relative to the scene-graph parent (default = identity matrix).
    :param Material material: A Material object defining the sphere's material (default = None).
    :param str name: A string specifying a user-friendly name for the sphere (default = "").

    """

    def __init__(self, double radius=0.5, object parent=None, AffineMatrix3D transform not None=AffineMatrix3D(), Material material not None=Material(), str name=None):

        super().__init__(parent, transform, material, name)

        if radius < 0.0:

            raise ValueError("Sphere radius cannot be less than zero.")

        self._radius = radius

        # initialise next intersection caching and control attributes
        self._further_intersection = False
        self._next_t = 0.0
        self._cached_origin = None
        self._cached_direction = None
        self._cached_ray = None

    property radius:
        """
        The radius of this sphere.

        :getter: Returns this sphere's radius.
        :setter: Sets this sphere's radius.
        :rtype: float
        """

        def __get__(self):
            return self._radius

        def __set__(self, double radius):

            if radius == self._radius:

                return

            if radius < 0.0:

                raise ValueError("Sphere radius cannot be less than zero.")

            self._radius = radius

            # the next intersection cache has been invalidated by the radius change
            self._further_intersection = False

            # any geometry caching in the root node is now invalid, inform root
            self.notify_geometry_change()

    cpdef Intersection hit(self, Ray ray):

        cdef Point3D origin
        cdef Vector3D direction
        cdef double a, b, c, d, q, t0, t1, temp, t_closest

        # reset further intersection state
        self._further_intersection = False

        # convert ray parameters to local space
        origin = ray.origin.transform(self.to_local())
        direction = ray.direction.transform(self.to_local())

        # coefficients of quadratic equation and discriminant
        a = (direction.x * direction.x
           + direction.y * direction.y
           + direction.z * direction.z)

        b = 2 * (direction.x * origin.x
               + direction.y * origin.y
               + direction.z * origin.z)

        c = (origin.x * origin.x
           + origin.y * origin.y
           + origin.z * origin.z
           - self._radius * self._radius)

        d = b*b - 4*a*c

        # ray misses sphere if there are no real roots of the quadratic
        if d < 0:

            return None

        # calculate intersection distances using method described in the book:
        # "Physically Based Rendering - 2nd Edition", Elsevier 2010
        # this method is more numerically stable than the usual root equation
        if b < 0:

            q = -0.5 * (b - sqrt(d))

        else:

            q = -0.5 * (b + sqrt(d))

        with cython.cdivision(True):

            t0 = q / a
            t1 = c / q

        # ensure t0 is always smaller than t1
        if t0 > t1:

            # swap
            temp = t0
            t0 = t1
            t1 = temp

        # test the intersection points inside the ray search range [0, max_distance]
        if (t0 > ray.max_distance) or (t1 < 0.0):

            return None

        if t0 >= 0.0:

            t_closest = t0
            if t1 <= ray.max_distance:

                self._further_intersection = True
                self._cached_ray = ray
                self._cached_origin = origin
                self._cached_direction = direction
                self._next_t = t1

        elif t1 <= ray.max_distance:

            t_closest = t1

        else:

            return None

        return self._generate_intersection(ray, origin, direction, t_closest)

    cpdef Intersection next_intersection(self):

        if not self._further_intersection:

            return None

        # this is the 2nd and therefore last intersection
        self._further_intersection = False

        return self._generate_intersection(self._cached_ray, self._cached_origin, self._cached_direction, self._next_t)

    cdef inline Intersection _generate_intersection(self, Ray ray, Point3D origin, Vector3D direction, double ray_distance):

        cdef Point3D hit_point, inside_point, outside_point
        cdef Normal3D normal
        cdef double delta_x, delta_y, delta_z
        cdef bint exiting

        # point of surface intersection in local space
        hit_point = new_point3d(origin.x + ray_distance * direction.x,
                                origin.y + ray_distance * direction.y,
                                origin.z + ray_distance * direction.z)

        # normal is normalised vector from sphere origin to hit_point
        normal = new_normal3d(hit_point.x, hit_point.y, hit_point.z)
        normal = normal.normalise()

        # calculate points inside and outside of surface for daughter rays to
        # spawn from - these points are displaced from the surface to avoid
        # re-hitting the same surface
        delta_x = EPSILON * normal.x
        delta_y = EPSILON * normal.y
        delta_z = EPSILON * normal.z

        inside_point = new_point3d(hit_point.x - delta_x,
                                   hit_point.y - delta_y,
                                   hit_point.z - delta_z)

        outside_point = new_point3d(hit_point.x + delta_x,
                                    hit_point.y + delta_y,
                                    hit_point.z + delta_z)

        # is ray exiting surface
        if direction.dot(normal) >= 0.0:

            exiting = True

        else:

            exiting = False

        return new_intersection(ray, ray_distance, self, hit_point, inside_point, outside_point,
                                normal, exiting, self.to_local(), self.to_root())

    cpdef bint contains(self, Point3D p) except -1:

        cdef Point3D local_point
        cdef double distance_sqr

        # convert world space point to local space
        local_point = p.transform(self.to_local())

        # calculate squared distance of the point from the sphere origin
        distance_sqr = (local_point.x * local_point.x
                      + local_point.y * local_point.y
                      + local_point.z * local_point.z)

        # point is outside sphere if point distance is greater than the radius
        # compare squares to avoid costly square root calculation
        if distance_sqr > self._radius * self._radius:

            return False

        return True

    cpdef BoundingBox3D bounding_box(self):

        cdef double extent
        cdef Point3D origin, lower, upper

        # obtain sphere origin in world space
        origin = new_point3d(0, 0, 0).transform(self.to_root())

        # calculate upper and lower corners of box
        extent = self._radius + BOX_PADDING
        lower = new_point3d(origin.x - extent, origin.y - extent, origin.z - extent)
        upper = new_point3d(origin.x + extent, origin.y + extent, origin.z + extent)

        return BoundingBox3D(lower, upper)
