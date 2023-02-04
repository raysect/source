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

from raysect.core.scenegraph.signal import GEOMETRY, MATERIAL


cdef class Primitive(Node):
    """
    A scene-graph object representing a ray-intersectable surface/volume.

    A primitive class defines an open surface or closed surface (volume) that can be intersected by a ray. For example,
    this could be a geometric primitive such as a sphere, or more complicated surface such as a polyhedral mesh. The
    primitive class is the only class in the scene-graph with which a ray can interact.

    This is a base class, its functionality must be implemented fully by the deriving class.

    :param Node parent: Assigns the Node's parent to the specified scene-graph object.
    :param AffineMatrix3D transform: Sets the affine transform associated with the Node.
    :param Material material: An object representing the material properties of the primitive.
    :param str name: A string defining the node name.
    """

    def __init__(self, object parent=None, AffineMatrix3D transform=None, Material material=None, str name=None):

        super().__init__(parent, transform, name)

        if material is None:
            material = Material()

        self.material = material

    @property
    def material(self):
        """
        The material class for this primitive.

        :rtype: Material
        """
        return self._material

    @material.setter
    def material(self, Material value not None):

        # remove any reverse reference from existing material
        if self._material is not None:
            self._material.primitives.remove(self)

        # assign new material and provide it with a reverse reference
        self._material = value
        self._material.primitives.append(self)

        # inform the scene-graph root that a material change has occurred
        self.notify_material_change()

    cdef Material get_material(self):
        return self._material

    cpdef Intersection hit(self, Ray ray):
        """
        Virtual method - to be implemented by derived classes.

        Calculates the closest intersection of the Ray with the Primitive
        surface, if such an intersection exists.

        If a hit occurs an Intersection object must be returned, otherwise None
        is returned. The intersection object holds the details of the
        intersection including the point of intersection, surface normal and
        the objects involved in the intersection.

        :param Ray ray: The ray to test for intersection.
        :return: An Intersection object or None if no intersection occurs.
        :rtype: Intersection
        """

        raise NotImplementedError("Primitive surface has not been defined. Virtual method hit() has not been implemented.")

    cpdef Intersection next_intersection(self):
        """
        Virtual method - to be implemented by derived classes.

        Returns the next intersection of the ray with the primitive along the
        ray path.

        This method may only be called following a call to hit(). If the ray
        has further intersections with the primitive, these may be obtained by
        repeatedly calling the next_intersection() method. Each call to
        next_intersection() will return the next ray-primitive intersection
        along the ray's path. If no further intersections are found or
        intersections lie outside the ray parameters then next_intersection()
        will return None.

        If any geometric elements of the primitive, ray and/or scene-graph are
        altered between a call to hit() and calls to next_intersection() the
        data returned by next_intersection() may be invalid. Primitives may
        cache data to accelerate next_intersection() calls which will be
        invalidated by geometric alterations to the scene. If the scene is
        altered the data returned by next_intersection() is undefined.

        :rtype: Intersection
        """

        raise NotImplementedError("Primitive surface has not been defined. Virtual method next_intersection() has not been implemented.")

    cpdef bint contains(self, Point3D p) except -1:
        """
        Virtual method - to be implemented by derived classes.

        Must returns True if the Point3D lies within the boundary of the surface
        defined by the Primitive. False is returned otherwise.

        :param Point3D p: The Point3D to test.
        :return: True if the Point3D is enclosed by the primitive surface, False otherwise.
        :rtype: bool
        """

        raise NotImplementedError("Primitive surface has not been defined. Virtual method inside() has not been implemented.")

    cpdef BoundingBox3D bounding_box(self):
        """
        Virtual method - to be implemented by derived classes.

        When the primitive is connected to a scene-graph containing a World
        object at its root, this method should return a bounding box that
        fully encloses the primitive's surface (plus a small margin to
        avoid numerical accuracy problems). The bounding box must be defined in
        the world's coordinate space.

        If this method is called when the primitive is not connected to a
        scene-graph with a World object at its root, it must throw a TypeError
        exception.

        :return: A world space BoundingBox3D object.
        :rtype: BoundingBox3D
        """

        raise NotImplementedError("Primitive surface has not been defined. Virtual method bounding_box() has not been implemented.")

    cpdef BoundingSphere3D bounding_sphere(self):
        """
        When the primitive is connected to a scene-graph containing a World
        object at its root, this method should return a bounding sphere that
        fully encloses the primitive's surface (plus a small margin to
        avoid numerical accuracy problems). The bounding sphere must be
        defined in the world's coordinate space.

        If this method is called when the primitive is not connected to a
        scene-graph with a World object at its root, it must throw a TypeError
        exception.
        
        The default implementation is to wrap the the primitive's bounding box
        with a sphere. If the bounding sphere can be more optimally calculated
        for the primitive, it should override this method.

        :return: A world space BoundingSphere3D object.
        :rtype: BoundingSphere3D
        """

        return self.bounding_box().enclosing_sphere()

    cpdef object instance(self, object parent=None, AffineMatrix3D transform=None, Material material=None, str name=None):
        """
        Returns a new instance of the primitive with the same geometry.
        
        :param Node parent: Assigns the Node's parent to the specified scene-graph object.
        :param AffineMatrix3D transform: Sets the affine transform associated with the Node.
        :param Material material: An object representing the material properties of the primitive.
        :param str name: A string defining the node name.
        :return: 
        """

        raise NotImplementedError("Virtual method instance() has not been implemented.")

    cpdef object notify_geometry_change(self):
        """
        Notifies the scene-graph root of a change to the primitive's geometry.

        This method must be called by primitives when their geometry changes.
        The notification informs the root node that any caching structures used
        to accelerate ray-tracing calculations are now potentially invalid and
        must be recalculated, taking the new geometry into account.
        """

        self.root._change(self, GEOMETRY)

    cpdef object notify_material_change(self):
        """
        Notifies the scene-graph root of a change to the primitive's material.

        This method must be called by primitives when their material changes.
        The notification informs the root node that any caching structures used
        to accelerate ray-tracing calculations are now potentially invalid and
        must be recalculated, taking the new material into account.
        """

        self.root._change(self, MATERIAL)

