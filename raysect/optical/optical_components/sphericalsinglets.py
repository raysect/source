
from ...core import AffineMatrix, translate, Point
from ...primitive import Sphere, Box, Cylinder
from ...primitive.csg import Union, Intersect, Subtract
from ..material.glass_libraries import Schott
from math import sqrt, fabs

# load default_glass
schott = Schott()
NBK7 = schott("N-BK7")

# infinity constant
INFINITY = 1e999

# class SphericalSingletLens(r1, r2, th, dia, parent=None, transform=AffineMatrix(), material=BK7()):
#
#     def __init__(self, r1, r2, th, dia, parent, transform, material):
#         # TODO: ask alex what is the purpose of the padding variable
#         padding = 1e-5
#         s1 = Sphere(fabs(r1), transform=translate(0, 0, -r1 + 0.5 * w))
#         s2 = Sphere(r2, transform=translate(0, 0, r2 - 0.5 * w))
#         cyl = Cylinder(0.5 * d, w + padding, transform=translate(0, 0, -0.5 * (w + padding)))
#         ret = Intersect(cyl, Intersect(s1, s2), parent, transform, material)
#
#
#     @property
#     def r1(self):
#         return self._r1
#
#     @r1.setter
#     def r1(self, value):
#         self._r1 = value
#
#     @property
#     def r2(self):
#         return self._r2
#
#     @r2.setter
#     def r2(self, value):
#         self._r2 = value
#
#     @property
#     def diameter(self):
#         return self._diameter
#
#     @diameter.setter
#     def diameter(self, value):
#         self._diameter = value
#
#     @property
#     def thickness(self):
#         return self._thickness
#
#     @thickness.setter
#     def thickness(self, value):
#         self._thickness = value
#
#     @property
#     def material(self):
#         return self._material
#
#     @material.setter
#     def material(self, value):
#         self._material = value
#
#     @property
#     def focallength(self):
#         raise NotImplementedError("This feature has not yet been implemented.")
#
#     @focallength.setter
#     def focallength(self, value):
#         raise NotImplementedError("This feature has not yet been implemented.")


def planar_convex_lens(focallength, diameter, thickness=None, material=NBK7, planarorigin=True, designwvl=587.6,
                       parent=None, transform=AffineMatrix()):

    if (focallength <= 0) or (diameter <= 0):
        raise ValueError("The focal-length, aperture and thickness of a planar-convex lens must all be greater than zero.")

    # TODO: fix this to get refractive index n from material properly.
    n = 1.25

    # set r2 based on the focal-length and the refractive index of material
    r2 = -focallength * (n - 1)

    # check for appropriate thickness value
    if diameter > fabs(r2):
        raise ValueError("Lens diameter should not be greater than Convex surface radius.")
    else:
        dia = diameter

    # calculate lens thickness if not given as input
    hyp = sqrt(r2**2 - (0.5 * dia)**2)
    if not thickness:
        th = fabs(r2) - hyp

    # otherwise, check the thickness given is valid. Must be consistent with the aperture value.
    else:
        if thickness <= 0:
            raise ValueError("Lens thickness cannot be less than zero.")
        elif thickness < (r2 - hyp):
            raise ValueError("Lens thickness specified is not consistent with the focal length and lens diameter given.")
        # TODO - need a better criteria for detecting non-sensible values of thickness
        elif thickness > 5 * (r2 - hyp):
            raise ValueError("Thin lens approximation is broken! Lens is much thicker than expected.")
        th = thickness

    if planarorigin:
        zoffset = 0.0
    else:
        zoffset = - thickness

    s2 = Sphere(fabs(r2), transform=translate(0, 0, r2 + th - zoffset))
    box = Box(Point(-dia, -dia, 0.0 - zoffset), Point(dia, dia, th - zoffset))

    return Intersect(box, s2, parent, transform, material)


def biconvex(r1, r2, w, d, parent=None, transform=AffineMatrix(), material=NBK7):
    padding = 1e-5
    s1 = Sphere(r1, transform=translate(0, 0, -r1 + 0.5 * w))
    s2 = Sphere(r2, transform=translate(0, 0, r2 - 0.5 * w))
    cyl = Cylinder(0.5 * d, w + padding, transform=translate(0, 0, -0.5 * (w + padding)))
    return Intersect(cyl, Intersect(s1, s2), parent, transform, material)


def biconcave(r1, r2, w, d, parent=None, transform=AffineMatrix(), material=NBK7):
    padding = 0.05
    s1 = Sphere(r1, transform=translate(0, 0, r1 + 0.5 * w))
    s2 = Sphere(r2, transform=translate(0, 0, -r2 - 0.5 * w))
    cyl = Cylinder(0.5 * d, w + padding, transform=translate(0, 0, -0.5 * (w + padding)))
    return Subtract(cyl, Union(s1, s2), parent, transform, material)

# lens1 = biconvex(0.0791, 0.6967, 0.015, 0.076, system_origin, transform=translate(0, 0, 0.008 + 0.025 + 0.5 * 0.015 + 2e-6), material=LaK9())
#
# Box(Point([-1.0, -1.0, 0]), Point([1.0, 1.0, 0.1]), system_origin, transform=translate(0, 0, 2.0), material=Checkerboard(0.05, 0.00, 1.1))