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

# TODO: add docstrings

cimport cython
from raysect.core.math cimport new_point3d, new_point2d

# cython doesn't have a built-in infinity constant, this compiles to +infinity
DEF INFINITY = 1e999

# axis defines
DEF X_AXIS = 0
DEF Y_AXIS = 1
DEF Z_AXIS = 2

# defines the padding on the sphere which encloses the BoundingBox3D.
DEF SPHERE_PADDING = 1.000001


@cython.freelist(256)
cdef class BoundingBox3D:
    """
    Axis-aligned bounding box.

    Represents a bounding box around a primitive's surface. The points defining
    the lower and upper corners of the box must be specified in world space.

    Axis aligned bounding box ray intersections are extremely fast to evaluate
    compared to intersections with more general geometry. Prior to testing a
    primitives hit() method the hit() method of the bounding box is called. If
    the bounding box is not hit, then the expensive primitive hit() method is
    avoided.

    Combined with a spatial subdivision acceleration structure, the cost of ray-
    primitive evaluations can be heavily reduced (O(n) -> O(log n)).

    For optimal speed the bounding box is aligned with the world space axes. As
    rays are propagated in world space, co-ordinate transforms can be avoided.

    :param Point3D lower: (optional) starting point for lower box corner
    :param Point3D upper: (optional) starting point for upper box corner
    """

    def __init__(self, Point3D lower=None, Point3D upper=None):

        # initialise to a null box if called without both initial points
        if lower is None or upper is None:
            self.lower = new_point3d(INFINITY, INFINITY, INFINITY)
            self.upper = new_point3d(-INFINITY, -INFINITY, -INFINITY)
        else:
            if lower.x > upper.x or lower.y > upper.y or lower.z > upper.z:
                raise ValueError("The lower point coordinates must be less than or equal to the upper point coordinates.")
            self.lower = lower
            self.upper = upper

    def __repr__(self):
        return "BoundingBox3D({}, {})".format(self.lower, self.upper)

    def __getstate__(self):
        """Encodes state for pickling."""

        return self.lower, self.upper

    def __setstate__(self, state):
        """Decodes state for pickling."""

        self.lower, self.upper = state

    @property
    def lower(self):
        """
        The point defining the lower corner of the bounding box.

        :rtype: Point3D
        """
        return self.lower

    @lower.setter
    def lower(self, Point3D value not None):
        self.lower = value

    @property
    def upper(self):
        """
        The point defining the upper corner of the bounding box.

        :rtype: Point3D
        """

        return self.upper

    @upper.setter
    def upper(self, Point3D value not None):
        self.upper = value

    @property
    def centre(self):
        """
        The point defining the geometric centre of the bounding box.

        :rtype: Point3D
        """
        return self.get_centre()

    cdef Point3D get_centre(self):
        """
        Find the centre of the bounding box.
        """

        return new_point3d(
            0.5 * (self.lower.x + self.upper.x),
            0.5 * (self.lower.y + self.upper.y),
            0.5 * (self.lower.z + self.upper.z)
        )

    cpdef bint hit(self, Ray ray):
        """
        Returns true if the ray hits the bounding box.

        :param Ray ray: The ray to test for intersection.
        :rtype: boolean
        """

        cdef:
            double front_intersection, back_intersection

        return self.intersect(ray, &front_intersection, &back_intersection)

    cpdef tuple full_intersection(self, Ray ray):
        """
        Returns full intersection information for an intersection between a ray and a bounding box.

        The first value is a boolean which is true if an intersection has occured, false otherwise. Each intersection
        with a bounding box will produce two intersections, one on the front and back of the box. The remaining two
        tuple parameters are floats representing the distance along the ray path to the respective intersections.

        :param ray: The ray to test for intersection
        :return: A tuple of intersection parameters, (hit, front_intersection, back_intersection).
        :rtype: tuple
        """

        cdef:
            double front_intersection, back_intersection
            bint hit

        hit = self.intersect(ray, &front_intersection, &back_intersection)

        return hit, front_intersection, back_intersection

    cdef bint intersect(self, Ray ray, double *front_intersection, double *back_intersection):

        # set initial ray-slab intersection search range
        front_intersection[0] = -INFINITY
        back_intersection[0] = INFINITY

        # evaluate ray-slab intersection for x, y and z dimensions and update the intersection positions
        self._slab(ray.origin.x, ray.direction.x, self.lower.x, self.upper.x, front_intersection, back_intersection)
        self._slab(ray.origin.y, ray.direction.y, self.lower.y, self.upper.y, front_intersection, back_intersection)
        self._slab(ray.origin.z, ray.direction.z, self.lower.z, self.upper.z, front_intersection, back_intersection)

        # does ray intersect box?
        if front_intersection[0] > back_intersection[0]:
            return False

        # are both intersections behind ray origin?
        if (front_intersection[0] < 0.0) and (back_intersection[0] < 0.0):
            return False
        return True

    @cython.cdivision(True)
    cdef void _slab(self, double origin, double direction, double lower, double upper, double *front_intersection, double *back_intersection) nogil:

        cdef double reciprocal, tmin, tmax

        if direction != 0.0:

            # calculate intersections with slab planes
            reciprocal = 1.0 / direction

            if direction > 0:

                # calculate length along ray path of intersections
                tmin = (lower - origin) * reciprocal
                tmax = (upper - origin) * reciprocal

            else:

                # calculate length along ray path of intersections
                tmin = (upper - origin) * reciprocal
                tmax = (lower - origin) * reciprocal

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

        # calculate slab intersection overlap, store closest dimension and intersected place
        if tmin > front_intersection[0]:
            front_intersection[0] = tmin

        if tmax < back_intersection[0]:
            back_intersection[0] = tmax

    cpdef bint contains(self, Point3D point):
        """
        Returns true if the given 3D point lies inside the bounding box.

        :param Point3D point: A given test point.
        :rtype: boolean
        """

        # point is inside box if it is inside all slabs
        if (point.x < self.lower.x) or (point.x > self.upper.x):
            return False
        if (point.y < self.lower.y) or (point.y > self.upper.y):
            return False
        if (point.z < self.lower.z) or (point.z > self.upper.z):
            return False
        return True

    cpdef object union(self, BoundingBox3D box):
        """
        Union this bounding box instance with the input bounding box.

        The resulting bounding box will be larger so as to just enclose both bounding boxes. This class instance
        will be edited in place to have the new bounding box dimensions.

        :param BoundingBox3D box: A bounding box instance to union with this bounding box instance.
        """

        self.lower.x = min(self.lower.x, box.lower.x)
        self.lower.y = min(self.lower.y, box.lower.y)
        self.lower.z = min(self.lower.z, box.lower.z)

        self.upper.x = max(self.upper.x, box.upper.x)
        self.upper.y = max(self.upper.y, box.upper.y)
        self.upper.z = max(self.upper.z, box.upper.z)

    cpdef object extend(self, Point3D point, double padding=0.0):
        """
        Enlarge this bounding box to enclose the given point.

        The resulting bounding box will be larger so as to just enclose the existing bounding box and the new point.
        This class instance will be edited in place to have the new bounding box dimensions.

        :param Point3D point: the point to use for extending the bounding box.
        :param float padding: optional padding parameter, gives extra margin around the new point.
        """

        self.lower.x = min(self.lower.x, point.x - padding)
        self.lower.y = min(self.lower.y, point.y - padding)
        self.lower.z = min(self.lower.z, point.z - padding)

        self.upper.x = max(self.upper.x, point.x + padding)
        self.upper.y = max(self.upper.y, point.y + padding)
        self.upper.z = max(self.upper.z, point.z + padding)

    cpdef double surface_area(self):
        """
        Returns the surface area of the bounding box.

        :rtype: float
        """

        cdef double dx, dy, dz

        dx = self.upper.x - self.lower.x
        dy = self.upper.y - self.lower.y
        dz = self.upper.z - self.lower.z

        return 2 * (dx * dy + dx * dz + dy * dz)

    cpdef double volume(self):
        """
        Returns the volume of the bounding box.

        :rtype: float
        """

        return (self.upper.x - self.lower.x) * (self.upper.y - self.lower.y) * (self.upper.z - self.lower.z)

    cpdef list vertices(self):
        """
        Get the list of vertices for this bounding box.

        :return: A list of Point3D's representing the corners of the bounding box.
        :rtype: list
        """

        return [
            new_point3d(self.lower.x, self.lower.y, self.lower.z),
            new_point3d(self.lower.x, self.lower.y, self.upper.z),
            new_point3d(self.lower.x, self.upper.y, self.lower.z),
            new_point3d(self.lower.x, self.upper.y, self.upper.z),
            new_point3d(self.upper.x, self.lower.y, self.lower.z),
            new_point3d(self.upper.x, self.lower.y, self.upper.z),
            new_point3d(self.upper.x, self.upper.y, self.lower.z),
            new_point3d(self.upper.x, self.upper.y, self.upper.z),
        ]

    cpdef double extent(self, int axis) except -1:
        """
        Returns the spatial extend of this bounding box along the given dimension.

        :param int axis: specifies the axis to return, {0: X axis, 1: Y axis, 2: Z axis}.
        :rtype: float
        """

        if axis == X_AXIS:
            return max(0.0, self.upper.x - self.lower.x)
        elif axis == Y_AXIS:
            return max(0.0, self.upper.y - self.lower.y)
        elif axis == Z_AXIS:
            return max(0.0, self.upper.z - self.lower.z)
        else:
            raise ValueError("Axis must be in the range [0, 2].")

    cpdef int largest_axis(self):
        """
        Find the largest axis of this bounding box.

        :return: an int specifying the longest axis, {0: X axis, 1: Y axis, 2: Z axis}.
        :rtype: int
        """

        cdef:
            int largest_axis
            double largest_extent, extent

        largest_axis = X_AXIS
        largest_extent = self.extent(X_AXIS)

        extent = self.extent(Y_AXIS)
        if extent > largest_extent:
            largest_axis = Y_AXIS
            largest_extent = extent

        extent = self.extent(Z_AXIS)
        if extent > largest_extent:
            largest_axis = Z_AXIS
            largest_extent = extent

        return largest_axis

    cpdef double largest_extent(self):
        """
        Find the largest spatial extent across all axes.

        :return: distance along the largest bounding box axis.
        :rtype: float
        """

        return max(self.extent(X_AXIS), self.extent(Y_AXIS), self.extent(Z_AXIS))

    cpdef object pad(self, double padding):
        """
        Makes the bounding box larger by the specified amount of padding.

        Every bounding box axis will end up larger by a factor of 2 x padding.

        :param float padding: distance to use as padding margin
        """

        self.lower.x = self.lower.x - padding
        self.lower.y = self.lower.y - padding
        self.lower.z = self.lower.z - padding

        self.upper.x = self.upper.x + padding
        self.upper.y = self.upper.y + padding
        self.upper.z = self.upper.z + padding

    cpdef object pad_axis(self, int axis, double padding):
        """
        Makes the bounding box larger along the specified axis by amount of padding.

        The specified bounding box axis will end up larger by a factor of 2 x padding.

        :param int axis: The axis to apply padding to {0: X axis, 1: Y axis, 2: Z axis}.
        :param float padding: Distance to use as padding margin.
        """

        if axis < 0 or axis > 2:
            raise ValueError("Axis must be in the range [0, 2].")

        if axis == X_AXIS:
            self.lower.x = self.lower.x - padding
            self.upper.x = self.upper.x + padding

        elif axis == Y_AXIS:
            self.lower.y = self.lower.y - padding
            self.upper.y = self.upper.y + padding

        elif axis == Z_AXIS:
            self.lower.z = self.lower.z - padding
            self.upper.z = self.upper.z + padding

        else:
            raise ValueError("Axis must be in the range [0, 2].")

    cpdef BoundingSphere3D enclosing_sphere(self):
        """
        Returns a BoundingSphere3D guaranteed to enclose the bounding box.

        The sphere is centred at the box centre. A small degree of padding is
        added to avoid numerical accuracy issues.

        :return: A BoundingSphere3D object.
        :rtype: BoundingSphere3D
        """

        cdef Point3D centre = self.get_centre()
        cdef double radius = self.lower.distance_to(centre) * SPHERE_PADDING
        return BoundingSphere3D(centre, radius)


@cython.freelist(256)
cdef class BoundingBox2D:
    """
    Axis-aligned 2D bounding box.

    :param Point2D lower: (optional) starting point for lower box corner
    :param Point2D upper: (optional) starting point for upper box corner
    """
    def __init__(self, Point2D lower=None, Point2D upper=None):

        # initialise to a null box if called without both initial points
        if lower is None or upper is None:
            self.lower = new_point2d(INFINITY, INFINITY)
            self.upper = new_point2d(-INFINITY, -INFINITY)
        else:
            if lower.x > upper.x or lower.y > upper.y:
                raise ValueError("The lower point coordinates must be less than or equal to the upper point coordinates.")
            self.lower = lower
            self.upper = upper

    def __repr__(self):

        return "BoundingBox2D({}, {})".format(self.lower, self.upper)

    def __getstate__(self):
        """Encodes state for pickling."""

        return self.lower, self.upper

    def __setstate__(self, state):
        """Decodes state for pickling."""

        self.lower, self.upper = state

    @property
    def lower(self):
        """
        The point defining the lower corner of the bounding box.

        :rtype: Point2D
        """
        return self.lower

    @lower.setter
    def lower(self, Point2D value not None):
        self.lower = value

    @property
    def upper(self):
        """
        The point defining the upper corner of the bounding box.

        :rtype: Point2D
        """
        return self.upper

    @upper.setter
    def upper(self, Point2D value not None):
        self.upper = value

    cpdef bint contains(self, Point2D point):
        """
        Returns true if the given 2D point lies inside the bounding box.

        :param Point2D point: A given test point.
        :rtype: boolean
        """
        # point is inside box if it is inside all slabs
        if (point.x < self.lower.x) or (point.x > self.upper.x):
            return False
        if (point.y < self.lower.y) or (point.y > self.upper.y):
            return False
        return True

    cpdef object union(self, BoundingBox2D box):
        """
        Union this bounding box instance with the input bounding box.

        The resulting bounding box will be larger so as to just enclose both bounding boxes. This class instance
        will be edited in place to have the new bounding box dimensions.

        :param BoundingBox2D box: A bounding box instance to union with this bounding box instance.
        """
        self.lower.x = min(self.lower.x, box.lower.x)
        self.lower.y = min(self.lower.y, box.lower.y)

        self.upper.x = max(self.upper.x, box.upper.x)
        self.upper.y = max(self.upper.y, box.upper.y)

    cpdef object extend(self, Point2D point, double padding=0.0):
        """
        Enlarge this bounding box to enclose the given point.

        The resulting bounding box will be larger so as to just enclose the existing bounding box and the new point.
        This class instance will be edited in place to have the new bounding box dimensions.

        :param Point2D point: the point to use for extending the bounding box.
        :param float padding: optional padding parameter, gives extra margin around the new point.
        """
        self.lower.x = min(self.lower.x, point.x - padding)
        self.lower.y = min(self.lower.y, point.y - padding)

        self.upper.x = max(self.upper.x, point.x + padding)
        self.upper.y = max(self.upper.y, point.y + padding)

    cpdef double surface_area(self):
        """
        Returns the surface area of the bounding box.

        :rtype: float
        """
        return (self.upper.x - self.lower.x) * (self.upper.y - self.lower.y)

    cpdef list vertices(self):
        """
        Get the list of vertices for this bounding box.

        :return: A list of Point2D's representing the corners of the bounding box.
        :rtype: list
        """
        return [
            new_point2d(self.lower.x, self.lower.y),
            new_point2d(self.lower.x, self.upper.y),
            new_point2d(self.upper.x, self.lower.y),
            new_point2d(self.upper.x, self.upper.y),
        ]

    cpdef double extent(self, int axis) except -1:
        """
        Returns the spatial extend of this bounding box along the given dimension.

        :param int axis: specifies the axis to return, {0: X axis, 1: Y axis}.
        :rtype: float
        """
        if axis == X_AXIS:
            return max(0.0, self.upper.x - self.lower.x)
        elif axis == Y_AXIS:
            return max(0.0, self.upper.y - self.lower.y)
        else:
            raise ValueError("Axis must be in the range [0, 1].")

    cpdef int largest_axis(self):
        """
        Find the largest axis of this bounding box.

        :return: an int specifying the longest axis, {0: X axis, 1: Y axis}.
        :rtype: int
        """
        cdef:
            int largest_axis
            double largest_extent, extent

        largest_axis = X_AXIS
        largest_extent = self.extent(X_AXIS)

        extent = self.extent(Y_AXIS)
        if extent > largest_extent:
            largest_axis = Y_AXIS
            largest_extent = extent

        return largest_axis

    cpdef double largest_extent(self):
        """
        Find the largest spatial extent across all axes.

        :return: distance along the largest bounding box axis.
        :rtype: float
        """
        return max(self.extent(X_AXIS), self.extent(Y_AXIS))

    cpdef object pad(self, double padding):
        """
        Makes the bounding box larger by the specified amount of padding.

        Every bounding box axis will end up larger by a factor of 2 x padding.

        :param float padding: distance to use as padding margin
        """
        self.lower.x = self.lower.x - padding
        self.lower.y = self.lower.y - padding

        self.upper.x = self.upper.x + padding
        self.upper.y = self.upper.y + padding

    cpdef object pad_axis(self, int axis, double padding):
        """
        Makes the bounding box larger along the specified axis by amount of padding.

        The specified bounding box axis will end up larger by a factor of 2 x padding.

        :param int axis: The axis to apply padding to {0: X axis, 1: Y axis}.
        :param float padding: Distance to use as padding margin.
        """

        if axis < 0 or axis > 1:
            raise ValueError("Axis must be in the range [0, 1].")

        if axis == X_AXIS:
            self.lower.x = self.lower.x - padding
            self.upper.x = self.upper.x + padding

        elif axis == Y_AXIS:
            self.lower.y = self.lower.y - padding
            self.upper.y = self.upper.y + padding

        else:
            raise ValueError("Axis must be in the range [0, 1].")