# cython: language_level=3

# Copyright (c) 2014-2016, Dr Alex Meakins, Raysect Project
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

from raysect.primitive import Intersect, Subtract, Sphere, Cylinder

from raysect.core cimport AffineMatrix3D, translate, Material
from raysect.primitive.utility cimport EncapsulatedPrimitive
from libc.math cimport sqrt

"""
Basic spherical lens primitives.
"""

DEF PAD_FACTOR = 1.000001

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
        self._calc_geometry()
        radius = 0.5 * diameter

        # validate
        if diameter <= 0:
            raise ValueError("The lens diameter must be greater than zero.")

        if center_thickness <= 0:
            raise ValueError("The lens thickness must be greater than zero.")

        if front_curvature <= 0:
            raise ValueError("The front radius of curvature must be greater than zero.")

        if back_curvature <= 0:
            raise ValueError("The back radius of curvature must be greater than zero.")

        if front_curvature < radius:
            raise ValueError("The radius of curvature of the front face cannot be less than the barrel radius.")

        if back_curvature < radius:
            raise ValueError("The radius of curvature of the back face cannot be less than the barrel radius.")

        if self.edge_thickness < 0:
            raise ValueError("The curvatures and/or thickness are too small to produce a lens of the specified diameter.")

        # padding to add to the barrel cylinder to avoid potential numerical accuracy issues
        padding = center_thickness * PAD_FACTOR

        # construct lens using CSG
        front = Sphere(front_curvature, transform=translate(0, 0, center_thickness - front_curvature))
        back = Sphere(back_curvature, transform=translate(0, 0, back_curvature))
        barrel = Cylinder(0.5 * diameter, center_thickness + padding, transform=translate(0, 0, -0.5 * padding))
        lens = Intersect(barrel, Intersect(front, back))

        # attach to local root (performed in EncapsulatedPrimitive init)
        super().__init__(lens, parent, transform, material, name)

    cdef inline void _calc_geometry(self):

        cdef double radius, radius_sqr

        # barrel radius
        radius = 0.5 * self.diameter
        radius_sqr = radius * radius

        # thickness of spherical surfaces
        self.front_thickness = self.front_curvature - sqrt(self.front_curvature * self.front_curvature - radius_sqr)
        self.back_thickness = self.back_curvature - sqrt(self.back_curvature * self.back_curvature - radius_sqr)

        # edge thickness is the length of the barrel without the curved surfaces
        self.edge_thickness = self.center_thickness - (self.front_thickness + self.back_thickness)


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
        self._calc_geometry()
        radius = 0.5 * diameter

        # validate
        if diameter <= 0:
            raise ValueError("The lens diameter must be greater than zero.")

        if center_thickness <= 0:
            raise ValueError("The lens thickness must be greater than zero.")

        if front_curvature <= 0:
            raise ValueError("The front radius of curvature must be greater than zero.")

        if back_curvature <= 0:
            raise ValueError("The back radius of curvature must be greater than zero.")

        if front_curvature < radius:
            raise ValueError("The radius of curvature of the front face cannot be less than the barrel radius.")

        if back_curvature < radius:
            raise ValueError("The radius of curvature of the back face cannot be less than the barrel radius.")

        # padding to add to the barrel cylinder to avoid potential numerical accuracy issues
        padding = self.edge_thickness * PAD_FACTOR

        # construct lens using CSG
        front = Sphere(front_curvature, transform=translate(0, 0, center_thickness + front_curvature))
        back = Sphere(back_curvature, transform=translate(0, 0, -back_curvature))
        barrel = Cylinder(0.5 * diameter, self.edge_thickness + padding, transform=translate(0, 0, -0.5 * padding + self.back_thickness))
        lens = Subtract(Subtract(barrel, front), back)

        # attach to local root (performed in EncapsulatedPrimitive init)
        super().__init__(lens, parent, transform, material, name)

    cdef inline void _calc_geometry(self):

        cdef double radius, radius_sqr

        # barrel radius
        radius = 0.5 * self.diameter
        radius_sqr = radius * radius

        # thickness of spherical surfaces
        self.front_thickness = self.front_curvature - sqrt(self.front_curvature * self.front_curvature - radius_sqr)
        self.back_thickness = self.back_curvature - sqrt(self.back_curvature * self.back_curvature - radius_sqr)

        # edge thickness is the length of the barrel without the curved surfaces
        self.edge_thickness = self.center_thickness + self.front_thickness + self.back_thickness


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
        self._calc_geometry()
        radius = 0.5 * diameter

        # validate
        if diameter <= 0:
            raise ValueError("The lens diameter must be greater than zero.")

        if center_thickness <= 0:
            raise ValueError("The lens thickness must be greater than zero.")

        if curvature <= 0:
            raise ValueError("The radius of curvature must be greater than zero.")

        if curvature < radius:
            raise ValueError("The radius of curvature of the face cannot be less than the barrel radius.")

        if self.edge_thickness < 0:
            raise ValueError("The curvature and/or thickness is too small to produce a lens of the specified diameter.")

        # padding to add to the barrel cylinder to avoid potential numerical accuracy issues
        padding = self.edge_thickness * PAD_FACTOR

        # construct lens using CSG
        curve = Sphere(curvature, transform=translate(0, 0, center_thickness - curvature))
        barrel = Cylinder(0.5 * diameter, self.edge_thickness + padding)
        lens = Intersect(barrel, curve)

        # attach to local root (performed in EncapsulatedPrimitive init)
        super().__init__(lens, parent, transform, material, name)

    cdef inline void _calc_geometry(self):

        cdef double radius, radius_sqr

        # barrel radius
        radius = 0.5 * self.diameter
        radius_sqr = radius * radius

        # thickness of spherical surfaces
        self.curve_thickness = self.curvature - sqrt(self.curvature * self.curvature - radius_sqr)

        # edge thickness is the length of the barrel without the curved surfaces
        self.edge_thickness = self.center_thickness - self.curve_thickness


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
        self._calc_geometry()
        radius = 0.5 * diameter

        # validate
        if diameter <= 0:
            raise ValueError("The lens diameter must be greater than zero.")

        if center_thickness <= 0:
            raise ValueError("The lens thickness must be greater than zero.")

        if curvature <= 0:
            raise ValueError("The radius of curvature must be greater than zero.")

        if curvature < radius:
            raise ValueError("The radius of curvature of the face cannot be less than the barrel radius.")

        # padding to add to the barrel cylinder to avoid potential numerical accuracy issues
        padding = self.edge_thickness * PAD_FACTOR

        # construct lens using CSG
        curve = Sphere(curvature, transform=translate(0, 0, center_thickness + curvature))
        barrel = Cylinder(0.5 * diameter, self.edge_thickness + padding)
        lens = Subtract(barrel, curve)

        # attach to local root (performed in EncapsulatedPrimitive init)
        super().__init__(lens, parent, transform, material, name)

    cdef inline void _calc_geometry(self):

        cdef double radius, radius_sqr

        # barrel radius
        radius = 0.5 * self.diameter
        radius_sqr = radius * radius

        # thickness of spherical surfaces
        self.curve_thickness = self.curvature - sqrt(self.curvature * self.curvature - radius_sqr)

        # edge thickness is the length of the barrel without the curved surfaces
        self.edge_thickness = self.center_thickness + self.curve_thickness


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
        readonly double full_thickness
        readonly double front_thickness
        readonly double back_thickness
        readonly double front_curvature
        readonly double back_curvature

    def __init__(self, double diameter, double center_thickness, double front_curvature, double back_curvature, object parent=None, AffineMatrix3D transform=None, Material material=None, str name=None):

        self.diameter = diameter
        self.center_thickness = center_thickness
        self.front_curvature = front_curvature
        self.back_curvature = back_curvature
        self._calc_geometry()
        radius = 0.5 * diameter

        # validate
        if diameter <= 0:
            raise ValueError("The lens diameter must be greater than zero.")

        if center_thickness <= 0:
            raise ValueError("The lens thickness must be greater than zero.")

        if front_curvature <= 0:
            raise ValueError("The front radius of curvature must be greater than zero.")

        if back_curvature <= 0:
            raise ValueError("The back radius of curvature must be greater than zero.")

        if front_curvature < radius:
            raise ValueError("The radius of curvature of the front face cannot be less than the barrel radius.")

        if back_curvature < radius:
            raise ValueError("The radius of curvature of the back face cannot be less than the barrel radius.")

        if self.edge_thickness < 0:
            raise ValueError("The curvatures and/or thickness are not compatible with the specified diameter.")

        # padding to add to the barrel cylinder to avoid potential numerical accuracy issues
        full_thickness = self.edge_thickness + self.front_thickness
        padding = full_thickness * PAD_FACTOR

        # construct lens using CSG
        front = Sphere(front_curvature, transform=translate(0, 0, center_thickness - front_curvature))
        back = Sphere(back_curvature, transform=translate(0, 0, -back_curvature))
        barrel = Cylinder(0.5 * diameter, full_thickness + padding, transform=translate(0, 0, -0.5 * padding - self.back_thickness))
        lens = Intersect(barrel, Subtract(front, back))

        # attach to local root (performed in EncapsulatedPrimitive init)
        super().__init__(lens, parent, transform, material, name)

    cdef inline void _calc_geometry(self):

        cdef double radius, radius_sqr

        # barrel radius
        radius = 0.5 * self.diameter
        radius_sqr = radius * radius

        # thickness of spherical surfaces
        self.front_thickness = self.front_curvature - sqrt(self.front_curvature * self.front_curvature - radius_sqr)
        self.back_thickness = self.back_curvature - sqrt(self.back_curvature * self.back_curvature - radius_sqr)

        # edge thickness is the length of the barrel without the curved surfaces
        self.edge_thickness = self.center_thickness -self.front_thickness + self.back_thickness
































