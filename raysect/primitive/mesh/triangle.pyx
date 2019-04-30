

cimport cython
from libc.math cimport abs
from raysect.core.math cimport Vector3D, new_vector3d, new_point3d


cdef class Triangle:

    def __init__(self, double v1x, double v1y, double v1z,
                 double v2x, double v2y, double v2z,
                 double v3x, double v3y, double v3z):

        cdef:
            Vector3D v1v2, v1v3, cross

        self.v1 = new_point3d(v1x, v1y, v1z)
        self.v2 = new_point3d(v2x, v2y, v2z)
        self.v3 = new_point3d(v3x, v3y, v3z)

        self.vc = new_point3d((v1x + v2x + v3x) / 3, (v1y + v2y + v3y) / 3, (v1z + v2z + v3z) / 3)

        v1v2 = self.v1.vector_to(self.v2)
        v1v3 = self.v1.vector_to(self.v3)
        cross = v1v2.cross(v1v3)

        if cross.x == 0 and cross.y == 0 and cross.z == 0:
            raise ValueError('The points specified do not define a valid 3D triangle.')

        self.area = cross.get_length()
        self.normal = cross.normalise()


@cython.cdivision(True)
cpdef tuple triangle3d_intersects_triangle3d(Triangle triangle_1, Triangle triangle_2, double tolerance=1e-6):
    """
    Cython utility for finding the intersection of two 3d triangles.
    
    If not intersection is found the tuple (False, None) will be returned. If an intersection
    is found, the tuple returned is (True, Intersect).
    
    http://web.stanford.edu/class/cs277/resources/papers/Moller1997b.pdf

    :param Triangle triangle_1: the first triangle to be tested.
    :param Triangle triangle_2: the second triangle to be tested.
    :param double tolerance: the tolerance level of the intersection calculation.
    :return: (False, None) if not intersection is found.
      If an intersection is found, the tuple returned is (True, Intersect).
    :rtype: tuple
    """

    cdef:

        Point3D u1, u2, u3, v1, v2, v3
        Point3D uc, vc
        Vector3D n_pi1, n_pi2, line_vec
        double d_pi1, d_pi2
        double uv_distance, d_u_from_v, d_v_from_u
        double lo_x, lo_y, lo_z
        double l
        double du1, du2, du3, dv1, dv2, dv3
        double pu1, pu2, pu3, pv1, pv2, pv3
        double min_pu, max_pu, min_mv, max_mv
        double t1, t2, t3, t4
        Point3D t2_point, t3_point
        bint valid_intersection_found = False
        bint t1_found = False, t2_found = False
        bint t3_found = False, t4_found = False

    u1 = triangle_1.v1
    u2 = triangle_1.v2
    u3 = triangle_1.v3
    uc = triangle_1.vc

    v1 = triangle_2.v1
    v2 = triangle_2.v2
    v3 = triangle_2.v3
    vc = triangle_2.vc

    # TODO - consider moving to Hessian form of plane equation

    # Setup Plane 1 equation
    n_pi1 = u1.vector_to(u2).cross(u1.vector_to(u3))  # normal vector of plane 1
    d_pi1 = n_pi1.dot(new_vector3d(u1.x, u1.y, u1.z))  # point satisfying the plane equation for plane 1

    # Setup Plane 2 equation
    n_pi2 = v1.vector_to(v2).cross(v1.vector_to(v3))  # normal vector of plane 2
    d_pi2 = n_pi2.dot(new_vector3d(v1.x, v1.y, v1.z))  # point satisfying the plane equation for plane 2

    # Find line on intersection of planes P1 and P2
    line_vec = n_pi1.cross(n_pi2)

    # Planes might be parallel
    if line_vec.get_length() == 0:
        if n_pi1.x == n_pi2.x and n_pi1.y == n_pi2.y and n_pi1.z == n_pi2.z and d_pi1 == d_pi2:
            raise NotImplementedError("Planes are parallel and overlapping, this case is not yet implemented.")
        else:
            return False,
    line_vec = line_vec.normalise()

    if n_pi1.y != 0.0:

        denominator = n_pi2.x - (n_pi1.x * n_pi2.y / n_pi1.y)
        if denominator != 0:
            lo_x = (d_pi2 - (d_pi1 * n_pi2.y / n_pi1.y)) / denominator
            lo_y = (d_pi1 - n_pi1.x * lo_x) / n_pi1.y
            lo_z = 0
            valid_intersection_found = True

    if not valid_intersection_found and n_pi2.y != 0.0:

        denominator = n_pi1.x - (n_pi2.x * n_pi1.y / n_pi2.y)
        if denominator != 0:
            lo_x = (d_pi1 - (d_pi2 * n_pi1.y / n_pi2.y)) / denominator
            lo_y = (d_pi2 - n_pi2.x * lo_x) / n_pi2.y
            lo_z = 0
            valid_intersection_found = True

    if not valid_intersection_found and n_pi1.z != 0.0:

        denominator = n_pi2.x - (n_pi1.x * n_pi2.z / n_pi1.z)
        if denominator != 0:
            lo_x = (d_pi2 - (d_pi1 * n_pi2.z / n_pi1.z)) / denominator
            lo_y = 0
            lo_z = (d_pi1 - n_pi1.x * lo_x) / n_pi1.z
            valid_intersection_found = True

    if not valid_intersection_found and n_pi2.z != 0.0:

        denominator = n_pi1.x - (n_pi2.x * n_pi1.z / n_pi2.z)
        if denominator != 0:
            lo_x = (d_pi1 - (d_pi2 * n_pi1.z / n_pi2.z)) / denominator
            lo_y = 0
            lo_z = (d_pi2 - n_pi2.x * lo_x) / n_pi2.z
            valid_intersection_found = True

    if not valid_intersection_found and n_pi1.y != 0.0:

        denominator = n_pi2.z - (n_pi1.z * n_pi2.y / n_pi1.y)
        if denominator != 0:
            lo_x = 0
            lo_z = (d_pi2 - (d_pi1 * n_pi2.y / n_pi1.y)) / denominator
            lo_y = (d_pi1 - n_pi1.z * lo_z) / n_pi1.y
            valid_intersection_found = True

    if not valid_intersection_found and n_pi2.y != 0.0:

        denominator = n_pi1.z - (n_pi2.z * n_pi1.y / n_pi2.y)
        if denominator != 0:
            lo_x = 0
            lo_z = (d_pi1 - (d_pi2 * n_pi1.y / n_pi2.y)) / denominator
            lo_y = (d_pi2 - n_pi2.z * lo_z) / n_pi2.y
            valid_intersection_found = True

    if not valid_intersection_found:
        print('')
        print('Debugging information')
        print('')
        print('u1 = Point3D({}, {}, {})'.format(triangle_1.v1.x, triangle_1.v1.y, triangle_1.v1.z))
        print('u2 = Point3D({}, {}, {})'.format(triangle_1.v2.x, triangle_1.v2.y, triangle_1.v2.z))
        print('u3 = Point3D({}, {}, {})'.format(triangle_1.v3.x, triangle_1.v3.y, triangle_1.v3.z))
        print('')
        print('v1 = Point3D({}, {}, {})'.format(triangle_2.v1.x, triangle_2.v1.y, triangle_2.v1.z))
        print('v2 = Point3D({}, {}, {})'.format(triangle_2.v2.x, triangle_2.v2.y, triangle_2.v2.z))
        print('v3 = Point3D({}, {}, {})'.format(triangle_2.v3.x, triangle_2.v3.y, triangle_2.v3.z))
        raise ValueError("Unsolvable triangle intersection problem.")

    line_origin = new_point3d(lo_x, lo_y, lo_z)

    l = n_pi2.get_length()
    du1 = (n_pi2.dot(new_vector3d(u1.x, u1.y, u1.z)) - d_pi2) / l
    du2 = (n_pi2.dot(new_vector3d(u2.x, u2.y, u2.z)) - d_pi2) / l
    du3 = (n_pi2.dot(new_vector3d(u3.x, u3.y, u3.z)) - d_pi2) / l

    # case for no intersection
    if (du1 > 0 and du2 > 0 and du3 > 0) or (du1 < 0 and du2 < 0 and du3 < 0):
        return False,

    l = n_pi1.get_length()
    dv1 = (n_pi1.dot(new_vector3d(v1.x, v1.y, v1.z)) - d_pi1) / l
    dv2 = (n_pi1.dot(new_vector3d(v2.x, v2.y, v2.z)) - d_pi1) / l
    dv3 = (n_pi1.dot(new_vector3d(v3.x, v3.y, v3.z)) - d_pi1) / l

    if (dv1 > 0 and dv2 > 0 and dv3 > 0) or (dv1 < 0 and dv2 < 0 and dv3 < 0):
        return False,

    # case for co-planar triangles
    elif (du1 == 0 and du2 == 0 and du3 == 0) or (dv1 == 0 and dv2 == 0 and dv3 == 0):
        raise NotImplementedError("Planes are parallel and overlapping, this case is not yet implemented.")

    # case for overlapping 3D triangles
    else:

        pu1 = line_vec.dot(line_origin.vector_to(u1))
        pu2 = line_vec.dot(line_origin.vector_to(u2))
        pu3 = line_vec.dot(line_origin.vector_to(u3))

        min_pu = min(pu1, pu2, pu3)
        max_pu = max(pu1, pu2, pu3)

        pv1 = line_vec.dot(line_origin.vector_to(v1))
        pv2 = line_vec.dot(line_origin.vector_to(v2))
        pv3 = line_vec.dot(line_origin.vector_to(v3))

        min_pv = min(pv1, pv2, pv3)
        max_pv = max(pv1, pv2, pv3)

        if not du1 - du2 == 0:
            t1 = pu1 + (pu2 - pu1) * (du1 / (du1 - du2))
            if min_pu <= t1 <= max_pu:
                t1_found = True

        if not du1 - du3 == 0:
            if not t1_found:
                t1 = pu1 + (pu3 - pu1) * (du1 / (du1 - du3))
                if min_pu <= t1 <= max_pu:
                    t1_found = True
            else:
                t2 = pu1 + (pu3 - pu1) * (du1 / (du1 - du3))
                if min_pu <= t2 <= max_pu:
                    t2_found = True

        if not t2_found and not du2 - du3 == 0:
            t2 = pu2 + (pu3 - pu2) * (du2 / (du2 - du3))
            if min_pu <= t2 <= max_pu:
                t2_found = True

        # ignore case of single contact point, need two contact points for valid intersection
        if not (t1_found and t2_found):
            return False,

        if t1 > t2:
            t1, t2 = t2, t1

        if not dv1 - dv2 == 0:
            t3 = pv1 + (pv2 - pv1) * (dv1 / (dv1 - dv2))
            if min_pv <= t3 <= max_pv:
                t3_found = True

        if not dv1 - dv3 == 0:
            if not t3_found:
                t3 = pv1 + (pv3 - pv1) * (dv1 / (dv1 - dv3))
                if min_pv <= t3 <= max_pv:
                    t3_found = True
            else:
                t4 = pv1 + (pv3 - pv1) * (dv1 / (dv1 - dv3))
                if min_pv <= t4 <= max_pv:
                    t4_found = True

        if not t4_found and not dv2 - dv3 == 0:
            t4 = pv2 + (pv3 - pv2) * (dv2 / (dv2 - dv3))
            if min_pv <= t4 <= max_pv:
                t4_found = True

        # ignore case of single contact point, need two contact points for valid intersection
        if not (t3_found and t4_found):
            return False,

        if t3 > t4:
            t3, t4 = t4, t3

        # ensure triangles are ordered lowest to highers in terms of parameter t (i.e. left to right)
        if t3 < t1:
            t1, t3 = t3, t1
            t2, t4 = t4, t2

        # test for no intersection
        if (t1 < t3 and t1 < t4 and t2 < t3 and t2 < t4) or (t1 > t3 and t1 > t4 and t2 > t3 and t2 > t4):
            return False,

        # case where one triangle is inside another
        elif t1 < t3 < t2 and t1 < t4 < t2:

            t3_point = new_point3d(line_origin.x + t3 * line_vec.x, line_origin.y + t3 * line_vec.y, line_origin.z + t3 * line_vec.z)
            t4_point = new_point3d(line_origin.x + t4 * line_vec.x, line_origin.y + t4 * line_vec.y, line_origin.z + t4 * line_vec.z)

            if t3_point.distance_to(t4_point) < tolerance:
                return False,

            return True, t3_point, t4_point

        # case where both triangles slightly overlap each other
        else:

            t2_point = new_point3d(line_origin.x + t2 * line_vec.x, line_origin.y + t2 * line_vec.y, line_origin.z + t2 * line_vec.z)
            t3_point = new_point3d(line_origin.x + t3 * line_vec.x, line_origin.y + t3 * line_vec.y, line_origin.z + t3 * line_vec.z)

            if t2_point.distance_to(t3_point) < tolerance:
                return False,

            return True, t2_point, t3_point
