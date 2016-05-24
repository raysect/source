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

# TODO: add more advanced material handling
# TODO: 2nd intersection calculation can be avoided subtract and intersection if the first primitive is missed

from raysect.core cimport _NodeBase, ChangeSignal, Material, new_ray, new_intersection, Point3D, AffineMatrix3D, BoundingBox3D

# bounding box is padded by a small amount to avoid numerical accuracy issues
DEF BOX_PADDING = 1e-9

# cython doesn't have a built in infinity definition
DEF INFINITY = 1e999

cdef class CSGPrimitive(Primitive):
    """
    Constructive Solid Geometry (CSG) Primitive base class.

    This is an abstract base class and can not be used directly.
    """

    def __init__(self, Primitive primitive_a not None = NullPrimitive(), Primitive primitive_b not None = NullPrimitive(), object parent = None, AffineMatrix3D transform not None = AffineMatrix3D(), Material material not None = Material(), str name=None):
        """
        Initialisation method.
        """

        super().__init__(parent, transform, material, name)

        # wrap primitives in bounding boxes
        # this must be done before building the scenegraph as re-parenting
        # triggers rebuild() on the new root node
        self._primitive_a = BoundPrimitive(primitive_a)
        self._primitive_b = BoundPrimitive(primitive_b)

        # build CSG scene graph
        self._csgroot = CSGRoot(self)
        primitive_a.parent = self._csgroot
        primitive_b.parent = self._csgroot

        # initialise next_intersection cache
        self._cache_ray = None
        self._cache_intersection_a = None
        self._cache_intersection_b = None
        self._cache_last_intersection = None
        self._cache_invalid = False

    property primitive_a:

        def __get__(self):
            return self._primitive_a.primitive

        def __set__(self, Primitive primitive not None):

            # remove old primitive from scenegraph
            self._primitive_a.primitive.parent = None

            # insert new primitive into scenegraph
            self._primitive_a = BoundPrimitive(primitive)
            primitive.parent = self._csgroot

            # invalidate next_intersection cache
            self._cache_invalid = True

    property primitive_b:

        def __get__(self):
            return self._primitive_b.primitive

        def __set__(self, Primitive primitive not None):

            # remove old primitive from scenegraph
            self._primitive_b.primitive.parent = None

            # insert new primitive into scenegraph
            self._primitive_b = BoundPrimitive(primitive)
            primitive.parent = self._csgroot

            # invalidate next_intersection cache
            self._cache_invalid = True

    cpdef Intersection hit(self, Ray ray):

        cdef:
            Ray local_ray
            Intersection intersection_a, intersection_b, closest_intersection

        # invalidate next_intersection cache
        self._cache_invalid = True

        # convert ray to local space
        local_ray = new_ray(ray.origin.transform(self.to_local()),
                            ray.direction.transform(self.to_local()),
                            INFINITY)

        # obtain initial intersections
        intersection_a = self._primitive_a.hit(local_ray)
        intersection_b = self._primitive_b.hit(local_ray)
        closest_intersection = self._closest_intersection(intersection_a, intersection_b)

        # identify first valid intersection
        return self._identify_intersection(ray, intersection_a, intersection_b, closest_intersection)

    cpdef Intersection next_intersection(self):

        cdef Intersection intersection_a, intersection_b, closest_intersection

        if self._cache_invalid:
            return None

        intersection_a = self._cache_intersection_a
        intersection_b = self._cache_intersection_b

        # replace intersection that was returned during the last call to hit() or next_intersection()
        if self._cache_last_intersection is intersection_a:
            intersection_a = self._primitive_a.next_intersection()
        else:
            intersection_b = self._primitive_b.next_intersection()

        closest_intersection = self._closest_intersection(intersection_a, intersection_b)

        # identify first valid intersection
        return self._identify_intersection(self._cache_ray, intersection_a, intersection_b, closest_intersection)

    cdef inline Intersection _identify_intersection(self, Ray ray, Intersection a, Intersection b, Intersection closest):

        cdef Intersection intersection

        # identify first intersection that satisfies csg operator
        while closest is not None:
            if self._valid_intersection(a, b, closest):
                if closest.ray_distance <= ray.max_distance:

                    # cache data for next_intersection
                    self._cache_ray = ray
                    self._cache_intersection_a = a
                    self._cache_intersection_b = b
                    self._cache_last_intersection = closest
                    self._cache_invalid = False

                    # allow derived classes to modify intersection if required
                    intersection = self._modify_intersection(closest, a, b)

                    # convert local intersection attributes to csg primitive coordinate space
                    intersection.ray = ray
                    intersection.hit_point = intersection.hit_point.transform(intersection.primitive_to_world)
                    intersection.inside_point = intersection.inside_point.transform(intersection.primitive_to_world)
                    intersection.outside_point = intersection.outside_point.transform(intersection.primitive_to_world)
                    intersection.normal = intersection.normal.transform(intersection.primitive_to_world)
                    intersection.world_to_primitive = self.to_local()
                    intersection.primitive_to_world = self.to_root()
                    intersection.primitive = self

                    return intersection

                else:
                    return None

            # closest intersection was rejected so need a replacement candidate intersection
            # from the primitive that was the source of the closest intersection
            if closest is a:
                a = self._primitive_a.next_intersection()
            else:
                b = self._primitive_b.next_intersection()
            closest = self._closest_intersection(a, b)

        # no valid intersections
        return None

    cdef inline Intersection _closest_intersection(self, Intersection a, Intersection b):

        if a is None:
            return b
        else:
            if b is None or a.ray_distance < b.ray_distance:
                return a
            else:
                return b

    cdef bint _valid_intersection(self, Intersection a, Intersection b, Intersection closest):
        raise NotImplementedError("Warning: CSG operator not implemented")

    cdef Intersection _modify_intersection(self, Intersection closest, Intersection a, Intersection b):
         # by default, do nothing
        return closest

    cdef void rebuild(self):
        """
        Triggers a rebuild of the CSG primitive's acceleration structures.
        """

        self._primitive_a = BoundPrimitive(self._primitive_a.primitive)
        self._primitive_b = BoundPrimitive(self._primitive_b.primitive)


cdef class NullPrimitive(Primitive):
    """
    Dummy primitive class.

    The _CSGPrimitive base class requires a primitive that returns a valid bounding box.
    This class overrides the bounding_box method to return an empty bounding box.
    This class is intended to act as a place holder until a user sets a valid primitive.
    """

    cpdef BoundingBox3D bounding_box(self):
        return BoundingBox3D()


cdef class CSGRoot(Node):
    """
    Specialised scenegraph root node for CSG primitives.

    The root node responds to geometry change notifications and propagates them
    to the CSG primitive and its enclosing scenegraph.
    """

    def __init__(self, CSGPrimitive csg_primitive):

        super().__init__()
        self.csg_primitive = csg_primitive

    def _change(self, _NodeBase node, ChangeSignal change not None):
        """
        Handles a scenegraph node change handler.

        Propagates geometry change notifications to the enclosing CSG primitive and its
        scenegraph.
        """

        # the CSG primitive acceleration structures must be rebuilt
        self.csg_primitive.rebuild()

        # propagate change notifications from csg scenegraph to enclosing scenegraph
        self.csg_primitive.root._change(node, change)


cdef class Union(CSGPrimitive):

    cdef bint _valid_intersection(self, Intersection a, Intersection b, Intersection closest):

        cdef bint inside_a, inside_b

        # determine ray enclosure state prior to intersection
        inside_a = a is not None and a.exiting
        inside_b = b is not None and b.exiting

        # union logic
        if not inside_a and not inside_b:
            # outside the whole object, intersection must be entering the object
            return True

        elif inside_a and not inside_b and closest is a:
            # outside primitive B and leaving primitive A, therefore leaving the unioned object
            return True

        elif not inside_a and inside_b and closest is b:
            # outside primitive A and leaving primitive B, therefore leaving the unioned object
            return True

        # all other intersections are occurring inside unioned object and are therefore invalid
        return False

    cpdef bint contains(self, Point3D p) except -1:

        p = p.transform(self.to_local())
        return self._primitive_a.contains(p) or self._primitive_b.contains(p)

    cpdef BoundingBox3D bounding_box(self):

        cdef:
            list points
            Point3D point
            BoundingBox3D box

        box = BoundingBox3D()

        # union local space bounding boxes
        box.union(self._primitive_a.box)
        box.union(self._primitive_b.box)

        # obtain local space vertices
        points = box.vertices()

        # convert points to world space and build an enclosing world space bounding box
        # a small degree of padding is added to avoid potential numerical accuracy issues
        box = BoundingBox3D()
        for point in points:
            box.extend(point.transform(self.to_root()), BOX_PADDING)

        return box


cdef class Intersect(CSGPrimitive):

    cdef bint _valid_intersection(self, Intersection a, Intersection b, Intersection closest):

        cdef bint inside_a, inside_b

        # determine ray enclosure state prior to intersection
        inside_a = a is not None and a.exiting
        inside_b = b is not None and b.exiting

        # intersect logic
        if inside_a and inside_b:
            # leaving both primitives
            return True

        elif inside_a and not inside_b and closest is b:
            # already inside primitive A and now entering primitive B
            return True

        elif not inside_a and inside_b and closest is a:
            # already inside primitive B and now entering primitive A
            return True

        # all other intersections are invalid
        return False

    cpdef bint contains(self, Point3D p) except -1:

        p = p.transform(self.to_local())
        return self._primitive_a.contains(p) and self._primitive_b.contains(p)

    cpdef BoundingBox3D bounding_box(self):

        cdef:
            list points
            Point3D point
            BoundingBox3D box

        box = BoundingBox3D()

        # find the intersection of the bounding boxes (this will always surround the intersected primitives)
        box.lower.x = max(self._primitive_a.box.lower.x, self._primitive_b.box.lower.x)
        box.lower.y = max(self._primitive_a.box.lower.y, self._primitive_b.box.lower.y)
        box.lower.z = max(self._primitive_a.box.lower.z, self._primitive_b.box.lower.z)

        box.upper.x = min(self._primitive_a.box.upper.x, self._primitive_b.box.upper.x)
        box.upper.y = min(self._primitive_a.box.upper.y, self._primitive_b.box.upper.y)
        box.upper.z = min(self._primitive_a.box.upper.z, self._primitive_b.box.upper.z)

        # obtain local space vertices
        points = box.vertices()

        # convert points to world space and build an enclosing world space bounding box
        # a small degree of padding is added to avoid potential numerical accuracy issues
        box = BoundingBox3D()
        for point in points:

            box.extend(point.transform(self.to_root()), BOX_PADDING)

        return box


cdef class Subtract(CSGPrimitive):

    cdef bint _valid_intersection(self, Intersection a, Intersection b, Intersection closest):

        cdef bint inside_a, inside_b

        # determine ray enclosure state prior to intersection
        inside_a = a is not None and a.exiting
        inside_b = b is not None and b.exiting

        # intersect logic
        if not inside_a and not inside_b and closest is a:
            # entering primitive A
            return True

        elif inside_a and not inside_b:
            # either exiting A or entering B
            return True

        elif inside_a and inside_b and closest is b:
            # inside both primitives, but leaving primitive B
            return True

        # all other intersections are invalid
        return False

    cdef Intersection _modify_intersection(self, Intersection closest, Intersection a, Intersection b):

        if closest is b:

            # invert exiting/entering state, normal and swap inside and outside points
            return new_intersection(
                ray=closest.ray,
                ray_distance=closest.ray_distance,
                primitive=closest.primitive,
                hit_point=closest.hit_point,
                inside_point=closest.outside_point,
                outside_point=closest.inside_point,
                normal=closest.normal.neg(),
                exiting=not closest.exiting,
                world_to_primitive=closest.world_to_primitive,
                primitive_to_world=closest.primitive_to_world
            )

        return closest

    cpdef bint contains(self, Point3D p) except -1:

        p = p.transform(self.to_local())
        return self._primitive_a.contains(p) and not self._primitive_b.contains(p)

    cpdef BoundingBox3D bounding_box(self):

        cdef:
            list points
            Point3D point
            BoundingBox3D box

        # a subtracted object (A - B) will only ever occupy the same or less space than the original primitive (A)
        # for simplicity just use the original primitive bounding box (A)
        points = self._primitive_a.box.vertices()

        # convert points to world space and build an enclosing world space bounding box
        # a small degree of padding is added to avoid potential numerical accuracy issues
        box = BoundingBox3D()
        for point in points:
            box.extend(point.transform(self.to_root()), BOX_PADDING)

        return box
