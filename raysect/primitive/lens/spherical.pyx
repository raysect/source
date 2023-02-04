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

from raysect.primitive import Intersect, Subtract, Union, Sphere, Cylinder

from raysect.core cimport Primitive
from raysect.core cimport AffineMatrix3D, translate, Material
from raysect.primitive.utility cimport EncapsulatedPrimitive
from libc.math cimport sqrt

"""
Basic spherical lens primitives.
"""

DEF PADDING = 0.000001


cdef class BiConvex(EncapsulatedPrimitive):
    """
    A bi-convex spherical lens primitive.

    A lens consisting of two convex spherical surfaces aligned on a common
    axis. The two surfaces sit at either end of a cylindrical barrel that is
    aligned to lie along the z-axis.

    The two lens surfaces are referred to as front and back respectively. The
    back surface is the negative surface most on the z-axis, while the front
    surface is the positive most surface on the z-axis. The centre of the back
    surface lies on z=0 and with the lens extending along the +ve z direction.

    :param diameter: The diameter of the lens body.
    :param center_thickness: The thickness of the lens measured along the lens axis.
    :param front_curvature: The radius of curvature of the front surface.
    :param back_curvature: The radius of curvature of the back surface.
    :param parent: Assigns the Node's parent to the specified scene-graph object.
    :param transform: Sets the affine transform associated with the Node.
    :param material: An object representing the material properties of the primitive.
    :param name: A string defining the node name.
    """

    cdef:
        readonly double diameter
        readonly double center_thickness
        readonly double edge_thickness
        readonly double front_thickness
        readonly double back_thickness
        readonly double front_curvature
        readonly double back_curvature

    def __init__(self, double diameter, double center_thickness, double front_curvature, double back_curvature, object parent=None, AffineMatrix3D transform=None, Material material=None, str name=None):

        self.diameter = diameter
        self.center_thickness = center_thickness
        self.front_curvature = front_curvature
        self.back_curvature = back_curvature

        # validate
        if diameter <= 0:
            raise ValueError("The lens diameter must be greater than zero.")

        if center_thickness <= 0:
            raise ValueError("The lens thickness must be greater than zero.")

        radius = 0.5 * diameter

        if front_curvature < radius:
            raise ValueError("The radius of curvature of the front face cannot be less than the barrel radius.")

        if back_curvature < radius:
            raise ValueError("The radius of curvature of the back face cannot be less than the barrel radius.")

        self._calc_geometry()

        if self.edge_thickness < 0:
            raise ValueError("The curvatures and/or thickness are too small to produce a lens of the specified diameter.")

        # construct lens
        if self._is_short():
            lens = self._build_short_lens()
        else:
            lens = self._build_long_lens()

        # attach to local root (performed in EncapsulatedPrimitive init)
        super().__init__(lens, parent, transform, material, name)

    cdef void _calc_geometry(self):

        cdef double radius, radius_sqr

        # barrel radius
        radius = 0.5 * self.diameter
        radius_sqr = radius * radius

        # thickness of spherical surfaces
        self.front_thickness = self.front_curvature - sqrt(self.front_curvature * self.front_curvature - radius_sqr)
        self.back_thickness = self.back_curvature - sqrt(self.back_curvature * self.back_curvature - radius_sqr)

        # edge thickness is the length of the barrel without the curved surfaces
        self.edge_thickness = self.center_thickness - (self.front_thickness + self.back_thickness)

    cdef bint _is_short(self):
        """
        Do the facing spheres overlap sufficiently to build a lens using just their intersection?        
        """

        cdef double available_thickness = min(
            2 * (self.front_curvature - self.front_thickness),
            2 * (self.back_curvature - self.back_thickness)
        )
        return self.edge_thickness <= available_thickness

    cdef Primitive _build_short_lens(self):
        """
        Short lens requires 3 primitives.
        """

        # padding to add to the barrel cylinder to avoid potential numerical accuracy issues
        padding = self.center_thickness * PADDING

        # construct lens using CSG
        front = Sphere(self.front_curvature, transform=translate(0, 0, self.center_thickness - self.front_curvature))
        back = Sphere(self.back_curvature, transform=translate(0, 0, self.back_curvature))
        barrel = Cylinder(0.5 * self.diameter, self.center_thickness + 2 * padding, transform=translate(0, 0, -padding))
        return Intersect(barrel, Intersect(front, back))

    cdef Primitive _build_long_lens(self):
        """
        Long lens requires 5 primitives.
        """

        # padding to avoid potential numerical accuracy issues
        padding = self.center_thickness * PADDING
        radius = 0.5 * self.diameter

        # front face
        front_sphere = Sphere(self.front_curvature, transform=translate(0, 0, self.center_thickness - self.front_curvature))
        front_barrel = Cylinder(radius, self.front_thickness + 2 * padding, transform=translate(0, 0, self.back_thickness + self.edge_thickness - padding))
        front_element = Intersect(front_sphere, front_barrel)

        # back face
        back_sphere = Sphere(self.back_curvature, transform=translate(0, 0, self.back_curvature))
        back_barrel = Cylinder(radius, self.back_thickness + 2 * padding, transform=translate(0, 0, -padding))
        back_element = Intersect(back_sphere, back_barrel)

        # bridging barrel
        barrel = Cylinder(radius, self.edge_thickness, transform=translate(0, 0, self.back_thickness))

        # construct lens
        return Union(barrel, Union(front_element, back_element))

    cpdef object instance(self, object parent=None, AffineMatrix3D transform=None, Material material=None, str name=None):
        return BiConvex(self.diameter, self.center_thickness, self.front_curvature, self.back_curvature, parent, transform, material, name)


cdef class BiConcave(EncapsulatedPrimitive):
    """
    A bi-concave spherical lens primitive.

    A lens consisting of two concave spherical surfaces aligned on a common
    axis. The two surfaces sit at either end of a cylindrical barrel that is
    aligned to lie along the z-axis.

    The two lens surfaces are referred to as front and back respectively. The
    back surface is the negative surface most on the z-axis, while the front
    surface is the positive most surface on the z-axis. The centre of the back
    surface lies on z=0 and with the lens extending along the +ve z direction.

    :param diameter: The diameter of the lens body.
    :param center_thickness: The thickness of the lens measured along the lens axis.
    :param front_curvature: The radius of curvature of the front surface.
    :param back_curvature: The radius of curvature of the back surface.
    :param parent: Assigns the Node's parent to the specified scene-graph object.
    :param transform: Sets the affine transform associated with the Node.
    :param material: An object representing the material properties of the primitive.
    :param name: A string defining the node name.
    """

    cdef:
        readonly double diameter
        readonly double center_thickness
        readonly double edge_thickness
        readonly double front_thickness
        readonly double back_thickness
        readonly double front_curvature
        readonly double back_curvature

    def __init__(self, double diameter, double center_thickness, double front_curvature, double back_curvature, object parent=None, AffineMatrix3D transform=None, Material material=None, str name=None):

        self.diameter = diameter
        self.center_thickness = center_thickness
        self.front_curvature = front_curvature
        self.back_curvature = back_curvature

        # validate
        if diameter <= 0:
            raise ValueError("The lens diameter must be greater than zero.")

        if center_thickness <= 0:
            raise ValueError("The lens thickness must be greater than zero.")

        radius = 0.5 * diameter

        if front_curvature < radius:
            raise ValueError("The radius of curvature of the front face cannot be less than the barrel radius.")

        if back_curvature < radius:
            raise ValueError("The radius of curvature of the back face cannot be less than the barrel radius.")

        self._calc_geometry()

        # construct lens using CSG
        front = Sphere(front_curvature, transform=translate(0, 0, center_thickness + front_curvature))
        back = Sphere(back_curvature, transform=translate(0, 0, -back_curvature))
        barrel = Cylinder(radius, self.edge_thickness, transform=translate(0, 0, -self.back_thickness))
        lens = Subtract(Subtract(barrel, front), back)

        # attach to local root (performed in EncapsulatedPrimitive init)
        super().__init__(lens, parent, transform, material, name)

    cdef void _calc_geometry(self):

        cdef double radius, radius_sqr

        # barrel radius
        radius = 0.5 * self.diameter
        radius_sqr = radius * radius

        # thickness of spherical surfaces
        self.front_thickness = self.front_curvature - sqrt(self.front_curvature * self.front_curvature - radius_sqr)
        self.back_thickness = self.back_curvature - sqrt(self.back_curvature * self.back_curvature - radius_sqr)

        # edge thickness is the length of the barrel without the curved surfaces
        self.edge_thickness = self.center_thickness + self.front_thickness + self.back_thickness

    cpdef object instance(self, object parent=None, AffineMatrix3D transform=None, Material material=None, str name=None):
        return BiConcave(self.diameter, self.center_thickness, self.front_curvature, self.back_curvature, parent, transform, material, name)


cdef class PlanoConvex(EncapsulatedPrimitive):
    """
    A plano-convex spherical lens primitive.

    A lens consisting of a convex spherical surface and a plane (flat) surface,
    aligned on a common axis. The two surfaces sit at either end of a
    cylindrical barrel that is aligned to lie along the z-axis.

    The two lens surfaces are referred to as front and back respectively. The
    back surface is the plane surface, it is the negative surface most on the
    z-axis. The front surface is the spherical surface, it is the positive most
    surface on the z-axis. The back (plane) surface lies on z=0 with the lens
    extending along the +ve z direction.

    :param diameter: The diameter of the lens body.
    :param center_thickness: The thickness of the lens measured along the lens axis.
    :param curvature: The radius of curvature of the spherical front surface.
    :param parent: Assigns the Node's parent to the specified scene-graph object.
    :param transform: Sets the affine transform associated with the Node.
    :param material: An object representing the material properties of the primitive.
    :param name: A string defining the node name.
    :return:
    """

    cdef:
        readonly double diameter
        readonly double center_thickness
        readonly double edge_thickness
        readonly double curve_thickness
        readonly double curvature

    def __init__(self, double diameter, double center_thickness, double curvature, object parent=None, AffineMatrix3D transform=None, Material material=None, str name=None):

        self.diameter = diameter
        self.center_thickness = center_thickness
        self.curvature = curvature

        # validate
        if diameter <= 0:
            raise ValueError("The lens diameter must be greater than zero.")

        if center_thickness <= 0:
            raise ValueError("The lens thickness must be greater than zero.")

        radius = 0.5 * diameter

        if curvature < radius:
            raise ValueError("The radius of curvature of the face cannot be less than the barrel radius.")

        self._calc_geometry()

        if self.edge_thickness < 0:
            raise ValueError("The curvature and/or thickness is too small to produce a lens of the specified diameter.")

        # construct lens
        if self._is_short():
            lens = self._build_short_lens()
        else:
            lens = self._build_long_lens()

        # attach to local root (performed in EncapsulatedPrimitive init)
        super().__init__(lens, parent, transform, material, name)
    
    cdef void _calc_geometry(self):

        cdef double radius, radius_sqr

        # barrel radius
        radius = 0.5 * self.diameter
        radius_sqr = radius * radius

        # thickness of spherical surfaces
        self.curve_thickness = self.curvature - sqrt(self.curvature * self.curvature - radius_sqr)

        # edge thickness is the length of the barrel without the curved surfaces
        self.edge_thickness = self.center_thickness - self.curve_thickness

    cdef bint _is_short(self):
        """
        Does the front sphere have sufficient radius to build the lens with just an intersection?        
        """

        cdef double available_thickness = 2 * (self.curvature - self.curve_thickness)
        return self.edge_thickness <= available_thickness

    cdef Primitive _build_short_lens(self):
        """
        Short lens requires 2 primitives.
        """

        # padding to add to the barrel cylinder to avoid potential numerical accuracy issues
        padding = self.center_thickness * PADDING

        # construct lens using CSG
        front = Sphere(self.curvature, transform=translate(0, 0, self.center_thickness - self.curvature))
        barrel = Cylinder(0.5 * self.diameter, self.center_thickness + padding)
        return Intersect(barrel, front)

    cdef Primitive _build_long_lens(self):
        """
        Long lens requires 3 primitives.
        """

        # padding to avoid potential numerical accuracy issues
        padding = self.center_thickness * PADDING
        radius = 0.5 * self.diameter

        # curved face
        curved_sphere = Sphere(self.curvature, transform=translate(0, 0, self.center_thickness - self.curvature))
        curved_barrel = Cylinder(radius, self.curve_thickness + 2 * padding, transform=translate(0, 0, self.edge_thickness - padding))
        curved_element = Intersect(curved_sphere, curved_barrel)

        # barrel
        barrel = Cylinder(radius, self.edge_thickness)

        # construct lens
        return Union(barrel, curved_element)

    cpdef object instance(self, object parent=None, AffineMatrix3D transform=None, Material material=None, str name=None):
        return PlanoConvex(self.diameter, self.center_thickness, self.curvature, parent, transform, material, name)


cdef class PlanoConcave(EncapsulatedPrimitive):
    """
    A plano-concave spherical lens primitive.

    A lens consisting of a concave spherical surface and a plane (flat)
    surface, aligned on a common axis. The two surfaces sit at either end of a
    cylindrical barrel that is aligned to lie along the z-axis.

    The two lens surfaces are referred to as front and back respectively. The
    back surface is the plane surface, it is the negative surface most on the
    z-axis. The front surface is the spherical surface, it is the positive most
    surface on the z-axis. The back (plane) surface lies on z=0 with the lens
    extending along the +ve z direction.

    :param diameter: The diameter of the lens body.
    :param center_thickness: The thickness of the lens measured along the lens axis.
    :param curvature: The radius of curvature of the spherical front surface.
    :param parent: Assigns the Node's parent to the specified scene-graph object.
    :param transform: Sets the affine transform associated with the Node.
    :param material: An object representing the material properties of the primitive.
    :param name: A string defining the node name.
    :return:
    """

    cdef:
        readonly double diameter
        readonly double center_thickness
        readonly double edge_thickness
        readonly double curve_thickness
        readonly double curvature

    def __init__(self, double diameter, double center_thickness, double curvature, object parent=None, AffineMatrix3D transform=None, Material material=None, str name=None):

        self.diameter = diameter
        self.center_thickness = center_thickness
        self.curvature = curvature

        # validate
        if diameter <= 0:
            raise ValueError("The lens diameter must be greater than zero.")

        if center_thickness <= 0:
            raise ValueError("The lens thickness must be greater than zero.")

        radius = 0.5 * diameter

        if curvature < radius:
            raise ValueError("The radius of curvature of the face cannot be less than the barrel radius.")

        self._calc_geometry()

        # construct lens using CSG
        curve = Sphere(curvature, transform=translate(0, 0, center_thickness + curvature))
        barrel = Cylinder(radius, self.edge_thickness)
        lens = Subtract(barrel, curve)

        # attach to local root (performed in EncapsulatedPrimitive init)
        super().__init__(lens, parent, transform, material, name)

    cdef void _calc_geometry(self):

        cdef double radius, radius_sqr

        # barrel radius
        radius = 0.5 * self.diameter
        radius_sqr = radius * radius

        # thickness of spherical surfaces
        self.curve_thickness = self.curvature - sqrt(self.curvature * self.curvature - radius_sqr)

        # edge thickness is the length of the barrel without the curved surfaces
        self.edge_thickness = self.center_thickness + self.curve_thickness

    cpdef object instance(self, object parent=None, AffineMatrix3D transform=None, Material material=None, str name=None):
        return PlanoConcave(self.diameter, self.center_thickness, self.curvature, parent, transform, material, name)


cdef class Meniscus(EncapsulatedPrimitive):
    """
    A meniscus spherical lens primitive.

    A lens consisting of a concave and a convex spherical surface aligned on a
    common axis. The two surfaces sit at either end of a cylindrical barrel
    that is aligned to lie along the z-axis.

    The two lens surfaces are referred to as front and back respectively. The
    back surface is concave, it is the negative surface most on the z-axis. The
    front surface is convex, it is the positive most surface on the z-axis. The
    centre of the back surface lies on z=0 and with the lens extending along
    the +ve z direction.

    :param diameter: The diameter of the lens body.
    :param center_thickness: The thickness of the lens measured along the lens axis.
    :param front_curvature: The radius of curvature of the front (convex) surface.
    :param back_curvature: The radius of curvature of the back (concave) surface.
    :param parent: Assigns the Node's parent to the specified scene-graph object.
    :param transform: Sets the affine transform associated with the Node.
    :param material: An object representing the material properties of the primitive.
    :param name: A string defining the node name.
    """
    cdef:
        readonly double diameter
        readonly double center_thickness
        readonly double edge_thickness
        readonly double front_thickness
        readonly double back_thickness
        readonly double front_curvature
        readonly double back_curvature

    def __init__(self, double diameter, double center_thickness, double front_curvature, double back_curvature, object parent=None, AffineMatrix3D transform=None, Material material=None, str name=None):

        self.diameter = diameter
        self.center_thickness = center_thickness
        self.front_curvature = front_curvature
        self.back_curvature = back_curvature

        # validate
        if diameter <= 0:
            raise ValueError("The lens diameter must be greater than zero.")

        if center_thickness <= 0:
            raise ValueError("The lens thickness must be greater than zero.")

        radius = 0.5 * diameter

        if front_curvature < radius:
            raise ValueError("The radius of curvature of the front face cannot be less than the barrel radius.")

        if back_curvature < radius:
            raise ValueError("The radius of curvature of the back face cannot be less than the barrel radius.")

        self._calc_geometry()

        if self.edge_thickness < 0:
            raise ValueError("The curvatures and/or thickness are not compatible with the specified diameter.")

        # construct lens
        if self._is_short():
            lens = self._build_short_lens()
        else:
            lens = self._build_long_lens()

        # attach to local root (performed in EncapsulatedPrimitive init)
        super().__init__(lens, parent, transform, material, name)

    cdef void _calc_geometry(self):

        cdef double radius, radius_sqr

        # barrel radius
        radius = 0.5 * self.diameter
        radius_sqr = radius * radius

        # thickness of spherical surfaces
        self.front_thickness = self.front_curvature - sqrt(self.front_curvature * self.front_curvature - radius_sqr)
        self.back_thickness = self.back_curvature - sqrt(self.back_curvature * self.back_curvature - radius_sqr)

        # edge thickness is the length of the barrel without the front surface
        self.edge_thickness = self.center_thickness - self.front_thickness + self.back_thickness

    cdef bint _is_short(self):
        """
        Does the front sphere have sufficient radius to build the lens with just an intersection?        
        """

        cdef double available_thickness = 2 * self.front_curvature - self.front_thickness
        return (self.center_thickness + self.back_thickness) <= available_thickness

    cdef Primitive _build_short_lens(self):
        """
        Short lens requires 3 primitives.
        """

        # padding to add to the barrel cylinder to avoid potential numerical accuracy issues
        padding = (self.back_thickness + self.center_thickness) * PADDING

        # construct lens using CSG
        front = Sphere(self.front_curvature, transform=translate(0, 0, self.center_thickness - self.front_curvature))
        back = Sphere(self.back_curvature, transform=translate(0, 0, -self.back_curvature))
        barrel = Cylinder(0.5 * self.diameter, self.back_thickness + self.center_thickness + padding, transform=translate(0, 0, -self.back_thickness))
        return Subtract(Intersect(barrel, front), back)

    cdef Primitive _build_long_lens(self):
        """
        Long lens requires 4 primitives.
        """

        # padding to avoid potential numerical accuracy issues
        padding = (self.back_thickness + self.center_thickness) * PADDING
        radius = 0.5 * self.diameter

        # front face
        front_sphere = Sphere(self.front_curvature, transform=translate(0, 0, self.center_thickness - self.front_curvature))
        front_barrel = Cylinder(radius, self.front_thickness + 2 * padding, transform=translate(0, 0, self.center_thickness - self.front_thickness - padding))
        front_element = Intersect(front_sphere, front_barrel)

        # back face
        back_element = Sphere(self.back_curvature, transform=translate(0, 0, -self.back_curvature))

        # barrel
        barrel = Cylinder(radius, self.edge_thickness, transform=translate(0, 0, -self.back_thickness))

        # construct lens
        return Subtract(Union(barrel, front_element), back_element)

    cpdef object instance(self, object parent=None, AffineMatrix3D transform=None, Material material=None, str name=None):
        return Meniscus(self.diameter, self.center_thickness, self.front_curvature, self.back_curvature, parent, transform, material, name)































