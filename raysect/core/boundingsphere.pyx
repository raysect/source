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

cimport cython
from libc.math cimport M_PI
from raysect.core cimport Vector3D, new_vector3d
from raysect.core.math.cython cimport solve_quadratic, swap_double


@cython.freelist(256)
cdef class BoundingSphere3D:
    """
    A bounding sphere.

    Represents a bounding sphere around a primitive's surface. The sphere's
    centre point and radius must be specified in world space.

    :param Point3D centre: the centre point of the bounding sphere.
    :param float radius: the radius of the sphere that bounds the primitive.
    """
    def __init__(self, Point3D centre, double radius):

        if radius <= 0:
            raise ValueError("The radius of the bounding sphere must be greater than zero.")

        self.centre = centre
        self.radius = radius

    def __repr__(self):
        return "BoundingSphere3D({}, {})".format(self.centre, self.radius)

    def __getstate__(self):
        """Encodes state for pickling."""

        return self.centre, self.radius

    def __setstate__(self, state):
        """Decodes state for pickling."""

        self.centre, self.radius = state

    @property
    def centre(self):
        """
        The point defining the centre of the bounding sphere.

        :rtype: Point3D
        """
        return self.centre

    @centre.setter
    def centre(self, Point3D value not None):
        self.centre = value

    @property
    def radius(self):
        return self.radius

    @radius.setter
    def radius(self, double value):
        if value <= 0:
            raise ValueError("The radius of the bounding sphere must be greater than zero.")
        self.radius = value

    cpdef bint hit(self, Ray ray):
        """
        Returns true if the ray hits the bounding box.

        :param Ray ray: The ray to test for intersection.
        :rtype: boolean
        """

        cdef double front_intersection, back_intersection
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

        cdef Point3D origin
        cdef Vector3D direction
        cdef double a, b, c, t0, t1, t_closest

        # generate vector from origin to sphere centre
        # subtract from ray origin
        # Equivalent to shifting space such that the sphere now lies on the origin.
        origin = ray.origin.sub(new_vector3d(self.centre.x, self.centre.y, self.centre.z))
        direction = ray.direction

        # coefficients of quadratic equation and discriminant
        a = direction.x * direction.x + direction.y * direction.y + direction.z * direction.z
        b = 2 * (direction.x * origin.x + direction.y * origin.y + direction.z * origin.z)
        c = origin.x * origin.x + origin.y * origin.y + origin.z * origin.z - self.radius * self.radius

        # calculate intersection distances by solving the quadratic equation
        # ray misses if there are no real roots of the quadratic
        if not solve_quadratic(a, b, c, &t0, &t1):
            return False

        # ensure t0 is always smaller than t1
        if t0 > t1:
            swap_double(&t0, &t1)

        front_intersection[0] = t0
        back_intersection[0] = t1

        # is the intersection behind the ray origin, i.e. ray does not hit sphere
        return t1 >= 0.0

    cpdef bint contains(self, Point3D point):
        """
        Returns true if the given 3D point lies inside the bounding sphere.

        :param Point3D point: A given test point.
        :rtype: boolean
        """
        return self.centre.distance_to(point) <= self.radius


    @cython.cdivision(True)
    cpdef object union(self, BoundingSphere3D sphere):
        """
        Union this bounding sphere instance with the input bounding sphere.

        The resulting bounding sphere will be larger so as to just enclose both bounding spheres.
        This class instance will be edited in place to have the new bounding sphere dimensions.

        :param BoundingSphere3D sphere: A bounding sphere instance to union with this bounding sphere instance.
        """

        cdef:
            BoundingSphere3D smaller, larger
            double centre_distance, radius
            Vector3D smaller_to_larger
            Point3D centre

        # Do nothing if trying to union with ourself.
        if sphere is self:
            return

        # Identify which sphere is smaller
        if sphere.radius > self.radius:
            larger = sphere
            smaller = self
        else:
            larger = self
            smaller = sphere

        # One sphere completely encloses the other if the distance between them plus the radius of the smaller
        # sphere is less than the radius of the larger sphere.
        centre_distance = self.centre.distance_to(sphere.centre)
        if centre_distance + smaller.radius < larger.radius:
            self.centre = larger.centre
            self.radius = larger.radius
            return

        # The spheres either partially overlap or not at all.
        # Calculate new diameter for unioned bounding sphere.
        radius = 0.5 * (centre_distance + smaller.radius + larger.radius)

        # Calculate new centre
        smaller_to_larger = smaller.centre.vector_to(larger.centre).normalise()
        centre = smaller.centre.add(smaller_to_larger.mul(radius - smaller.radius))

        # update values
        self.radius = radius
        self.centre = centre

    cpdef object extend(self, Point3D point, double padding=0.0):
        """
        Enlarge this bounding box to enclose the given point.

        The resulting bounding box will be larger so as to just enclose the existing bounding box and the new point.
        This class instance will be edited in place to have the new bounding box dimensions.

        :param Point3D point: the point to use for extending the bounding box.
        :param float padding: optional padding parameter, gives extra margin around the new point.
        """

        cdef:
            double radius
            Vector3D centre_to_point
            Point3D centre

        # Does point lie inside current sphere?
        if self.contains(point):
            return

        # Calculate new diameter for bounding sphere that includes the point.
        radius = 0.5 * (self.centre.distance_to(point) + self.radius)

        # Calculate new centre
        centre_to_point = self.centre.vector_to(point).normalise()
        centre = self.centre.add(centre_to_point.mul(radius - self.radius))

        # update values
        self.radius = radius
        self.centre = centre

    cpdef double surface_area(self):
        """
        Returns the surface area of the bounding sphere.

        :rtype: float
        """
        return 4.0 * M_PI * self.radius * self.radius

    @cython.cdivision(True)
    cpdef double volume(self):
        """
        Returns the volume of the bounding sphere.

        :rtype: float
        """
        return 4.0/3.0 * M_PI * self.radius * self.radius * self.radius

    cpdef object pad(self, double padding):
        """
        Makes the bounding sphere larger by the specified amount of padding.

        :param float padding: Distance to use as padding margin.
        """

        self.radius = self.radius + padding
