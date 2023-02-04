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

from raysect.core cimport new_point3d, Normal3D, new_normal3d, AffineMatrix3D, Material, new_intersection, BoundingBox3D
from libc.math cimport fabs
cimport cython

# cython doesn't have a built-in infinity constant, this compiles to +infinity
DEF INFINITY = 1e999

# bounding box is padded by a small amount to avoid numerical accuracy issues
DEF BOX_PADDING = 1e-9

# additional ray distance to avoid re-hitting the same surface point
DEF EPSILON = 1e-9

# slab face enumeration
DEF NO_FACE = -1
DEF LOWER_FACE = 0
DEF UPPER_FACE = 1

# axis enumeration
DEF NO_AXIS = -1
DEF X_AXIS = 0
DEF Y_AXIS = 1
DEF Z_AXIS = 2


cdef class Box(Primitive):
    """
    A box primitive.

    The box is defined by lower and upper points in the local co-ordinate
    system.

    :param Point3D lower: Lower point of the box (default = Point3D(-0.5, -0.5, -0.5)).
    :param Point3D upper: Upper point of the box (default = Point3D(0.5, 0.5, 0.5)).
    :param Node parent: Scene-graph parent node or None (default = None).
    :param AffineMatrix3D transform: An AffineMatrix3D defining the local co-ordinate system relative to the scene-graph parent (default = identity matrix).
    :param Material material: A Material object defining the box's material (default = None).
    :param str name: A string specifying a user-friendly name for the box (default = "").

    .. code-block:: pycon

        >>> from raysect.core import Point3D, translate
        >>> from raysect.primitive import Box
        >>> from raysect.optical import World
        >>> from raysect.optical.material import UniformSurfaceEmitter
        >>> from raysect.optical.library.spectra.colours import red
        >>>
        >>> world = World()
        >>>
        >>> cube = Box(Point3D(0,0,0), Point3D(1,1,1), parent=world, transform=translate(0, 1, 0),
                       material=UniformSurfaceEmitter(red), name="red cube")
    """

    def __init__(self, Point3D lower=None, Point3D upper=None, object parent=None, AffineMatrix3D transform=None, Material material=None, str name=None):

        super().__init__(parent, transform, material, name)

        if lower is not None and upper is not None:

            if lower.x > upper.x or lower.y > upper.y or lower.z > upper.z:
                raise ValueError("The lower point coordinates must be less than or equal to the upper point coordinates.")
            self._lower = lower
            self._upper = upper

        elif lower is None and upper is None:

            # default to unit box centred on the axis
            self._lower = new_point3d(-0.5, -0.5, -0.5)
            self._upper = new_point3d(0.5, 0.5, 0.5)

        else:
            raise ValueError("Lower and upper points must both be defined.")

        # initialise next intersection caching and control attributes
        self._further_intersection = False
        self._next_t = 0.0
        self._cached_origin = None
        self._cached_direction = None
        self._cached_ray = None
        self._cached_face = NO_FACE
        self._cached_axis = NO_AXIS

    @property
    def lower(self):
        """
        Lower 3D coordinate of the box in primitive's local coordinates.

        :rtype: Point3D
        """

        return self._lower

    @lower.setter
    def lower(self, Point3D value not None):
        if value.x > self._upper.x or value.y > self._upper.y or value.z > self._upper.z:
            raise ValueError("The lower point coordinates must be less than or equal to the upper point coordinates.")
        self._lower = value

        # the next intersection cache has been invalidated by the geometry change
        self._further_intersection = False

        # any geometry caching in the root node is now invalid, inform root
        self.notify_geometry_change()

    @property
    def upper(self):
        """
        Upper 3D coordinate of the box in primitive's local coordinates.

        :rtype: Point3D
        """

        return self._upper

    @upper.setter
    def upper(self, Point3D value not None):
        if self._lower.x > value.x or self._lower.y > value.y or self._lower.z > value.z:
            raise ValueError("The upper point coordinates must be greater than or equal to the lower point coordinates.")
        self._upper = value

        # the next intersection cache has been invalidated by the geometry change
        self._further_intersection = False

        # any geometry caching in the root node is now invalid, inform root
        self.notify_geometry_change()

    cpdef Intersection hit(self, Ray ray):

        cdef:
            Point3D origin
            Vector3D direction
            double near_intersection, far_intersection, closest_intersection
            int near_face, far_face, closest_face
            int near_axis, far_axis, closest_axis

        # invalidate next intersection cache
        self._further_intersection = False

        # convert ray origin and direction to local space
        origin = ray.origin.transform(self.to_local())
        direction = ray.direction.transform(self.to_local())

        # set initial ray-slab intersection search range
        near_intersection = -INFINITY
        far_intersection = INFINITY

        # initially there are no face intersections, set to invalid values to aid debugging
        near_face = NO_FACE
        near_axis = NO_AXIS
        far_face = NO_FACE
        far_axis = NO_AXIS

        # evaluate ray-slab intersection for x, y and z dimensions and update the intersection positions
        self._slab(X_AXIS, origin.x, direction.x, self._lower.x, self._upper.x, &near_intersection, &far_intersection, &near_face, &far_face, &near_axis, &far_axis)
        self._slab(Y_AXIS, origin.y, direction.y, self._lower.y, self._upper.y, &near_intersection, &far_intersection, &near_face, &far_face, &near_axis, &far_axis)
        self._slab(Z_AXIS, origin.z, direction.z, self._lower.z, self._upper.z, &near_intersection, &far_intersection, &near_face, &far_face, &near_axis, &far_axis)

        # does ray intersect box?
        if near_intersection > far_intersection:
            return None

        # are there any intersections inside the ray search range?
        if near_intersection > ray.max_distance or far_intersection < 0.0:
            return None

        # identify closest intersection
        if near_intersection >= 0.0:

            closest_intersection = near_intersection
            closest_face = near_face
            closest_axis = near_axis

            if far_intersection <= ray.max_distance:

                self._further_intersection = True
                self._next_t = far_intersection
                self._cached_origin = origin
                self._cached_direction = direction
                self._cached_ray = ray
                self._cached_face = far_face
                self._cached_axis = far_axis

        elif far_intersection <= ray.max_distance:

            closest_intersection = far_intersection
            closest_face = far_face
            closest_axis = far_axis

        else:
            return None

        return self._generate_intersection(ray, origin, direction, closest_intersection, closest_face, closest_axis)

    cpdef Intersection next_intersection(self):

        if not self._further_intersection:
            return None

        # this is the 2nd and therefore last intersection
        self._further_intersection = False

        return self._generate_intersection(self._cached_ray, self._cached_origin, self._cached_direction, self._next_t, self._cached_face, self._cached_axis)

    cdef void _slab(self, int axis, double origin, double direction, double lower, double upper, double *near_intersection, double *far_intersection, int *near_face, int *far_face, int *near_axis, int *far_axis) nogil:

        cdef:
            double reciprocal, tmin, tmax
            int fmin, fmax

        if direction != 0.0:

            # calculate intersections with slab planes
            with cython.cdivision(True):
                reciprocal = 1.0 / direction

            if direction > 0:

                # calculate length along ray path of intersections
                tmin = (lower - origin) * reciprocal
                tmax = (upper - origin) * reciprocal

                fmin = LOWER_FACE
                fmax = UPPER_FACE

            else:

                # calculate length along ray path of intersections
                tmin = (upper - origin) * reciprocal
                tmax = (lower - origin) * reciprocal

                fmin = UPPER_FACE
                fmax = LOWER_FACE

        else:

            # ray is not propagating along this axis so limits are infinite
            if origin < lower:

                tmin = -INFINITY
                tmax = -INFINITY

            elif origin > upper:

                tmin = INFINITY
                tmax = INFINITY

            else:

                tmin = -INFINITY
                tmax = INFINITY

            fmin = NO_FACE
            fmax = NO_FACE

        # calculate slab intersection overlap, store closest dimension and intersected face
        if tmin > near_intersection[0]:
            near_intersection[0] = tmin
            near_face[0] = fmin
            near_axis[0] = axis

        if tmax < far_intersection[0]:
            far_intersection[0] = tmax
            far_face[0] = fmax
            far_axis[0] = axis

    cdef Intersection _generate_intersection(self, Ray ray, Point3D origin, Vector3D direction, double ray_distance, int face, int axis):

        cdef Point3D hit_point, inside_point, outside_point
        cdef Normal3D normal
        cdef Intersection intersection
        cdef bint exiting

        # point of surface intersection in local space
        hit_point = new_point3d(origin.x + ray_distance * direction.x,
                                origin.y + ray_distance * direction.y,
                                origin.z + ray_distance * direction.z)

        # calculate surface normal in local space
        normal = new_normal3d(0, 0, 0)
        if face == LOWER_FACE:
            normal.set_index(axis, -1.0)
        else:
            normal.set_index(axis, 1.0)

        # displace hit_point away from surface to generate inner and outer points
        inside_point = new_point3d(hit_point.x + self._interior_offset(hit_point.x, self._lower.x, self._upper.x),
                                   hit_point.y + self._interior_offset(hit_point.y, self._lower.y, self._upper.y),
                                   hit_point.z + self._interior_offset(hit_point.z, self._lower.z, self._upper.z))

        outside_point = new_point3d(hit_point.x + EPSILON * normal.x,
                                    hit_point.y + EPSILON * normal.y,
                                    hit_point.z + EPSILON * normal.z)

        # is ray exiting surface
        exiting = direction.dot(normal) >= 0.0

        return new_intersection(ray, ray_distance, self, hit_point, inside_point, outside_point,
                                normal, exiting, self.to_local(), self.to_root())

    cdef double _interior_offset(self, double hit_point, double lower, double upper) nogil:
        """
        Calculates an interior offset that ensures the inside_point is away from the primitive surface.

        The calculation breaks down for boxes dimensions close to the minimum offset epsilon (~1e-9).
        Users should not be creating objects that approach raysect numerical accuracy limits.
        """

        if fabs(hit_point - lower) < EPSILON:
            return EPSILON
        elif fabs(hit_point - upper) < EPSILON:
            return -EPSILON
        return 0.0

    cpdef bint contains(self, Point3D point) except -1:

        # convert point to local object space
        point = point.transform(self.to_local())

        # point is inside box if it is inside all slabs
        if (point.x < self._lower.x) or (point.x > self._upper.x):
            return False

        if (point.y < self._lower.y) or (point.y > self._upper.y):
            return False

        if (point.z < self._lower.z) or (point.z > self._upper.z):
            return False

        return True

    cpdef BoundingBox3D bounding_box(self):

        cdef:
            list points
            Point3D point
            BoundingBox3D box

        # generate box vertices
        points = [
            self._lower,
            new_point3d(self._lower.x, self._lower.y, self._upper.z),
            new_point3d(self._lower.x, self._upper.y, self._lower.z),
            new_point3d(self._lower.x, self._upper.y, self._upper.z),
            new_point3d(self._upper.x, self._lower.y, self._lower.z),
            new_point3d(self._upper.x, self._lower.y, self._upper.z),
            new_point3d(self._upper.x, self._upper.y, self._lower.z),
            self._upper
            ]

        # convert points to world space and build an enclosing world space bounding box
        # a small degree of padding is added to avoid potential numerical accuracy issues
        box = BoundingBox3D()
        for point in points:
            box.extend(point.transform(self.to_root()), BOX_PADDING)

        return box

    cpdef object instance(self, object parent=None, AffineMatrix3D transform=None, Material material=None, str name=None):
        return Box(self._lower.copy(), self._upper.copy(), parent, transform, material, name)