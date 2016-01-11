
from core import AffineMatrix, translate, Point, rotate
from primitive import Sphere, Box, Cylinder
from ...primitive.csg import Union, Intersect, Subtract
from optical.material.glass_libraries import Schott
from math import sqrt, fabs

# load default_glass
schott = Schott()
NBK7 = schott("N-BK7")

# infinity constant
INFINITY = 1e999


def planar_convex_lens(focallength, diameter, thickness=None, material=NBK7, designwvl=587.6,
                       parent=None, transform=AffineMatrix()):

    if (focallength <= 0) or (diameter <= 0):
        raise ValueError("The focal-length, aperture and thickness of a planar-convex lens must all be greater than zero.")

    # TODO: fix this to get refractive index n from material properly.
    n = 1.25

    # set r2 based on the focal-length and the refractive index of material
    r2 = -focallength * (n - 1)

    # check for appropriate thickness value
    if diameter > fabs(r2)*2:
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

    s2 = Sphere(fabs(r2), transform=translate(0, 0, r2 + th))
    box = Box(Point(-dia, -dia, 0.0), Point(dia, dia, th))

    return Intersect(box, s2, parent, transform, material)


def planar_concave_lens(focallength, diameter, thickness=None, material=NBK7, designwvl=587.6,
                        parent=None, transform=AffineMatrix()):

    if (focallength <= 0) or (diameter <= 0):
        raise ValueError("The focal-length, aperture and thickness of a planar-convex lens must all be greater than zero.")

    # TODO: fix this to get refractive index n from material properly.
    n = 1.25

    # set r2 based on the focal-length and the refractive index of material
    r2 = focallength * (n - 1)

    # check for appropriate thickness value
    if diameter > fabs(2 * r2):
        raise ValueError("Lens diameter should not be greater than Convex surface radius.")
    else:
        dia = diameter

    # calculate lens thickness if not given as input
    hyp = sqrt(r2**2 - (0.5 * dia)**2)
    vacuum_th = fabs(r2) - hyp
    if not thickness:
        th = vacuum_th

    # otherwise, check the thickness given is valid. Must be consistent with the aperture value.
    else:
        if thickness <= 0:
            raise ValueError("Lens thickness cannot be less than zero.")
        th = thickness

    s2 = Sphere(fabs(r2), transform=translate(0, 0, r2 + th))
    cyl = Cylinder(0.5 * dia, (th + vacuum_th))
    return Subtract(cyl, s2, parent, transform, material)


def symmetric_biconvex_lens(focallength, diameter, thickness=None, material=NBK7, designwvl=587.6,
                            parent=None, transform=AffineMatrix()):

    if (focallength <= 0) or (diameter <= 0):
        raise ValueError("The focal-length, aperture and thickness of a planar-convex lens must all be greater than zero.")

    # TODO: fix this to get refractive index n from material properly. Needs to be at the design wavelength
    n = 1.25

    # set radii of both lens based on the focal-length and the refractive index of material
    r = 2 * focallength * (n - 1)

    # check for appropriate thickness value
    if diameter > fabs(2 * r):
        raise ValueError("Lens diameter should not be greater than Convex surface radius.")
    else:
        dia = diameter

    # calculate lens thickness if not given as input
    hyp = sqrt(r**2 - (0.5 * dia)**2)
    curve_th = r - hyp
    if not thickness:
        th = 2 * curve_th

    # otherwise, check the thickness given is valid. Must be consistent with the aperture value.
    else:
        if thickness <= 0:
            raise ValueError("Lens thickness cannot be less than zero.")
        elif thickness < (2 * curve_th):
            raise ValueError("Lens thickness specified is too small for the lens diameter given.")
        # TODO - need a better criteria for detecting non-sensible values of thickness
        elif thickness > 5 * (2 * curve_th):
            raise ValueError("Thin lens approximation is broken! Lens is much thicker than expected.")
        th = thickness

    s1 = Sphere(r, transform=translate(0, 0, r))
    s2 = Sphere(r, transform=translate(0, 0, -r + 2 * th))
    cyl = Cylinder(0.5 * dia, th)
    return Intersect(cyl, Intersect(s1, s2), parent, transform, material)


def symmetric_biconcave_lens(focallength, diameter, thickness=None, material=NBK7, designwvl=587.6,
                            parent=None, transform=AffineMatrix()):

    if (focallength <= 0) or (diameter <= 0):
        raise ValueError("The focal-length, aperture and thickness of a planar-convex lens must all be greater than zero.")

    # TODO: fix this to get refractive index n from material properly. Needs to be at the design wavelength
    n = 1.25

    # set radii of both lens based on the focal-length and the refractive index of material
    r = 2 * focallength * (n - 1)

    # check for appropriate thickness value
    if diameter > fabs(2 * r):
        raise ValueError("Lens diameter should not be greater than Convex surface radius.")
    else:
        dia = diameter

    # calculate lens thickness if not given as input
    hyp = sqrt(r**2 - (0.5 * dia)**2)
    curve_th = r - hyp
    if not thickness:
        th = curve_th

    # otherwise, check the thickness given is valid. Must be consistent with the aperture value.
    else:
        if thickness <= 0:
            raise ValueError("Lens thickness cannot be less than zero.")
        elif thickness < curve_th:
            raise ValueError("Lens thickness specified is too small for the lens diameter given.")
        # TODO - need a better criteria for detecting non-sensible values of thickness
        elif thickness > 5 * curve_th:
            raise ValueError("Thin lens approximation is broken! Lens is much thicker than expected.")
        th = thickness

    s1 = Sphere(r, transform=translate(0, 0, -r))
    s2 = Sphere(r, transform=translate(0, 0, r + th))
    cyl = Cylinder(0.5 * dia, (th + 2 * hyp), transform=translate(0, 0, -hyp))
    return Subtract(cyl, Union(s1, s2), parent, transform, material)


def positive_meniscus_lens(focallength, diameter, r1, r2=None, thickness=None, material=NBK7, designwvl=587.6,
                            parent=None, transform=AffineMatrix()):
    # Also known as a meniscus convex lens

    r1 = abs(r1)  # sign convention is arbitrary for meniscus lens.
    # TODO: fix this to get refractive index n from material properly. Needs to be at the design wavelength
    n = 1.25

    # test or calculate the required lens parameters.
    if (focallength <= 0) or (diameter <= 0):
        raise ValueError("The focal-length, aperture and thickness of a meniscus lens must all be greater than zero.")
    if r2:
        if fabs(r1) >= fabs(r2):
            raise ValueError("abs(r1) must be less than abs(r2) for a positive meniscus (convex) lens.")
        elif r2 != 1/(1/r1 - 1/(f(n-1))):
            raise ValueError("The r1 and r2 values given are not consistant with the thin lens approximation for the parameters given.")
    else:
        r2 = 1/(1/r1 - 1/(f(n-1)))

    if thickness:
        if thickness >= diameter/3:
            raise ValueError("Thickness larger than expected. Thin lens approximation may be invalid.")
    else:
        pass


def generic_spherical_singlet(r1, r2, dia, thickness, material=NBK7, parent=None, transform=AffineMatrix()):

    # generic planar-convex lens
    if r1 == INFINITY and r2 < 0:
        s2 = Sphere(fabs(r2), transform=translate(0, 0, r2 + thickness))
        box = Box(Point(-dia, -dia, 0.0), Point(dia, dia, thickness))
        return Intersect(box, s2, parent, transform, material)

    # generic planar-concave lens
    elif r1 == INFINITY and r2 > 0:
        hyp = sqrt(r2**2 - (0.5 * dia)**2)
        vacuum_th = fabs(r2) - hyp
        s2 = Sphere(fabs(r2), transform=translate(0, 0, r2 + thickness))
        cyl = Cylinder(0.5 * dia, (thickness + vacuum_th))
        return Subtract(cyl, s2, parent, transform, material)

    else:
        raise ValueError("Lens specification is invalid.")



