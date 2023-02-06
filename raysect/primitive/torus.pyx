# cython: language_level=3

# Copyright (c) 2014-2021, Dr Alex Meakins, Raysect Project
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


from raysect.core cimport Material, new_intersection, BoundingBox3D, BoundingSphere3D, new_point3d, new_normal3d, Normal3D, AffineMatrix3D
from raysect.core.math.cython cimport solve_quartic, swap_double
from libc.math import atan2, cos, sin


# bounding box and sphere are padded by small amounts to avoid numerical accuracy issues
DEF BOX_PADDING = 1e-9
DEF SPHERE_PADDING = 1.000000001

# additional ray distance to avoid re-hitting the same surface point
DEF EPSILON = 1e-9


cdef class Torus(Primitive):
    """
    A torus primitive.

    The torus is defined by major and minor radius.
    The major radius is the distance from the center of the tube to the center of the torus.
    The minor radius is the radius of the tube.
    The center of the torus corresponds to the origin of the local co-ordinate system.
    The axis of revolution coincides with the z-axis

    :param float major_radius: Major radius of the torus in meters (default = 1.0).
    :param float minor_radius: Minor radius of the torus in meters (default = 0.5).
    :param Node parent: Scene-graph parent node or None (default = None).
    :param AffineMatrix3D transform: An AffineMatrix3D defining the local co-ordinate system relative to the scene-graph parent (default = identity matrix).
    :param Material material: A Material object defining the torus's material (default = None).
    :param str name: A string specifying a user-friendly name for the torus (default = "").

    :ivar float mjor_radius: The major radius of the torus in meters.
    :ivar float minor_radius: The minor radius of the torus in meters.

    .. code-block:: pycon

        >>> from raysect.core import translate
        >>> from raysect.primitive import Torus
        >>> from raysect.optical import World
        >>> from raysect.optical.material import UniformSurfaceEmitter
        >>> from raysect.optical.library.spectra.colours import orange
        >>>
        >>> world = World()
        >>>
        >>> torus = Torus(1.0, 0.5, parent=world, transform=translate(3, 0, 0),
                          material=UniformSurfaceEmitter(orange), name="orange torus")
    """

    def __init__(self, double major_radius=1.0, double minor_radius=0.5, object parent=None,
                 AffineMatrix3D transform=None, Material material=None, str name=None):

        super().__init__(parent, transform, material, name)

        if major_radius < minor_radius or minor_radius < 0.0:
            raise ValueError("Torus minor radius cannot be less than zero and greater than major radius.")

        self._major_radius = major_radius
        self._minor_radius = minor_radius

        # initialise next intersection caching and control attributes
        self._further_intersection = False
        self._next_t = 0.0
        self._cached_origin = None
        self._cached_direction = None
        self._cached_ray = None

    @property
    def major_radius(self):
        """
        The major radius of this torus.

        :rtype: float
        """
        return self._major_radius

    @major_radius.setter
    def major_radius(self, double major_radius):

        # don't do anything if the value is unchanged
        if major_radius == self._major_radius:
            return

        if major_radius < 0.0:
            raise ValueError("Torus major radius cannot be less than zero.")
        if major_radius < self._minor_radius:
            raise ValueError("Torus major radius cannot be less than minor radius.")
        self._major_radius = major_radius

        # the next intersection cache has been invalidated by the major radius change
        self._further_intersection = False

        # any geometry caching in the root node is now invalid, inform root
        self.notify_geometry_change()

    @property
    def minor_radius(self):
        """
        The minor radius of this torus.

        :rtype: float
        """
        return self._minor_radius

    @minor_radius.setter
    def minor_radius(self, double minor_radius):

        # don't do anything if the value is unchanged
        if minor_radius == self._minor_radius:
            return

        if minor_radius < 0.0:
            raise ValueError("Torus minor radius cannot be less than zero.")
        if minor_radius > self._major_radius:
            raise ValueError("Torus minor radius cannot be greater than major radius.")
        self._minor_radius = minor_radius

        # the next intersection cache has been invalidated by the minor radius change
        self._further_intersection = False

        # any geometry caching in the root node is now invalid, inform root
        self.notify_geometry_change()

    cpdef Intersection hit(self, Ray ray):

        cdef Point3D origin
        cdef Vector3D direction
        cdef double sq_origin, sq_r, sq_R, dot_origin_direction, f, b, c, d, e, t0, t1, t2, t3, t_closest

        # reset further intersection state
        self._further_intersection = False

        # convert ray parameters to local space
        origin = ray.origin.transform(self.to_local())
        direction = ray.direction.transform(self.to_local()).normalise()

        # coefficients of quartic equation
        sq_origin = origin.x * origin.x + origin.y * origin.y + origin.z * origin.z
        dot_origin_direction = direction.x * origin.x + direction.y * origin.y + direction.z * origin.z
        sq_r = self._minor_radius * self._minor_radius
        sq_R = self._major_radius * self._major_radius
        f = sq_origin - (sq_r + sq_R)

        b = 4.0 * dot_origin_direction
        c = 2.0 * f + 4.0 * dot_origin_direction * dot_origin_direction + 4.0 * sq_R * direction.y * direction.y
        d = 4.0 * f * dot_origin_direction + 8.0 * sq_R * origin.y * direction.y
        e = f * f - 4.0 * sq_R * (sq_r - origin.y * origin.y)

        # calculate intersection distances by solving the quartic equation
        # ray misses if there are no real roots of the quartic
        num = solve_quartic(1.0, b, c, d, e, &t0, &t1, &t2, &t3)

        if num == 0:
            return None
        
        elif num == 1:
            # test the intersection points inside the ray search range [0, max_distance]
            if t0 > ray.max_distance or t0 < 0.0:
                return None
            else:
                t_closest = t0
                return self._generate_intersection(ray, origin, direction, t_closest)
        
        elif num == 2:
            # ensure t0 is always smaller than t1
            if t0 > t1:
                swap_double(&t0, &t1)



        # ensure t0 is always smaller than t1
        if t0 > t1:
            swap_double(&t0, &t1)

        # test the intersection points inside the ray search range [0, max_distance]
        if t0 > ray.max_distance or t1 < 0.0:
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

    cdef Intersection _generate_intersection(self, Ray ray, Point3D origin, Vector3D direction, double ray_distance):

        cdef Point3D hit_point, tube_centre, inside_point, outside_point
        cdef Normal3D normal
        cdef double phi, delta_x, delta_y, delta_z
        cdef bint exiting

        # point of surface intersection in local space
        hit_point = new_point3d(origin.x + ray_distance * direction.x,
                                origin.y + ray_distance * direction.y,
                                origin.z + ray_distance * direction.z)

        # normal is normalised vector from torus tube centre to hit_point
        phi = atan2(hit_point.y, hit_point.x)
        tube_centre = new_point3d(self._major_radius * cos(phi), self._major_radius * sin(phi), 0.0)
        tube_to_hit = tube_centre.vector_to(hit_point)
        normal = new_normal3d(tube_to_hit.x, tube_to_hit.y, tube_to_hit.z)
        normal = normal.normalise()

        # calculate points inside and outside of surface for daughter rays to
        # spawn from - these points are displaced from the surface to avoid
        # re-hitting the same surface
        delta_x = EPSILON * normal.x
        delta_y = EPSILON * normal.y
        delta_z = EPSILON * normal.z

        inside_point = new_point3d(hit_point.x - delta_x, hit_point.y - delta_y, hit_point.z - delta_z)
        outside_point = new_point3d(hit_point.x + delta_x, hit_point.y + delta_y, hit_point.z + delta_z)

        # is ray exiting surface
        exiting = direction.dot(normal) >= 0.0

        return new_intersection(ray, ray_distance, self, hit_point, inside_point, outside_point,
                                normal, exiting, self.to_local(), self.to_root())

    cpdef bint contains(self, Point3D point) except -1:

        cdef Point3D local_point
        cdef double discriminant, distance_xy, distance_sqr, sq_R, R2_r2

        # convert world space point to local space
        local_point = point.transform(self.to_local())

        # calculate the interior discriminant
        distance_xy = local_point.x * local_point.x + local_point.y * local_point.y
        distance_sqr = distance_xy + local_point.z * local_point.z
        sq_R = self._major_radius * self._major_radius
        R2_r2 = sq_R - self._minor_radius * self._minor_radius
        discriminant = distance_sqr * distance_sqr + 2.0 * distance_sqr * R2_r2 + R2_r2 * R2_r2 - 4.0 * sq_R * distance_xy

        # point is outside torus if discriminant is greater than 0
        return discriminant <= 0.0

    cpdef BoundingBox3D bounding_box(self):

        cdef double extent
        cdef Point3D origin, lower, upper

        # obtain torus origin in world space
        origin = new_point3d(0, 0, 0).transform(self.to_root())

        # calculate upper and lower corners of box
        extent = self._major_radius + self._minor_radius + BOX_PADDING
        lower = new_point3d(origin.x - extent, origin.y - extent, origin.z - extent)
        upper = new_point3d(origin.x + extent, origin.y + extent, origin.z + extent)

        return BoundingBox3D(lower, upper)

    cpdef BoundingSphere3D bounding_sphere(self):
        cdef Point3D centre = new_point3d(0, 0, 0).transform(self.to_root())
        return BoundingSphere3D(centre, (self._major_radius + self._minor_radius) * SPHERE_PADDING)

    cpdef object instance(self, object parent=None, AffineMatrix3D transform=None, Material material=None, str name=None):
        return Torus(self._major_radius, self._minor_radius, parent, transform, material, name)