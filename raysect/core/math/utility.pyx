
from libc.math cimport sqrt, sin, cos, atan2, M_PI
from raysect.core.math.point cimport Point3D, new_point3d


cdef:
    double RAD2DEG = 360 / (2 * M_PI)
    double DEG2RAD = (2 * M_PI) / 360


cpdef tuple cartesian_to_cylindrical(Point3D point):
    """
    Convert the given 3D point in cartesian space to cylindrical coordinates. 
    
    :param Point3D point: The 3D point to be transformed into cylindrical coordinates.
    :rtype: tuple
    :return: A tuple of r, z, phi coordinates.
    
    .. code-block:: pycon
    
        >>> from raysect.core.math import cartesian_to_cylindrical, Point3D
        
        >>> point = Point3D(1, 1, 1)
        >>> cartesian_to_cylindrical(point)
        (1.4142135623730951, 1.0, 45.0)
    """

    cdef double r, phi

    r = sqrt(point.x*point.x + point.y*point.y)
    phi = atan2(point.y, point.x) * RAD2DEG

    return r, point.z, phi


cpdef Point3D cylindrical_to_cartesian(double r, double z, double phi):
    """
    Convert a 3D point in cylindrical coordinates to a point in cartesian coordinates.
    
    :param float r: The radial coordinate.
    :param float z: The z-axis height coordinate.
    :param float phi: The azimuthal coordinate in degrees.
    :rtype: Point3D
    :return: A Point3D in cartesian space.
    
    .. code-block:: pycon
    
        >>> from raysect.core.math import cylindrical_to_cartesian

        >>> cylindrical_to_cartesian(1, 0, 45)
        Point3D(0.7071067811865476, 0.7071067811865475, 0.0)
    """


    cdef double x, y

    if r < 0:
        raise ValueError("R coordinate cannot be less than 0.")

    x = r * cos(phi * DEG2RAD)
    y = r * sin(phi * DEG2RAD)

    return new_point3d(x, y, z)
