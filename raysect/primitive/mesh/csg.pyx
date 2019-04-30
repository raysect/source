
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from scipy.spatial import Delaunay, KDTree

from raysect.core cimport Point3D, Vector3D, translate, rotate_basis, new_point3d
from raysect.primitive.mesh.triangle cimport Triangle, triangle3d_intersects_triangle3d
from raysect.primitive.mesh.weld_vertices import weld_vertices


cdef class CSG_Operator:

    cdef bint _m1(self, double signed_distance):
        raise NotImplementedError

    cdef bint _m2(self, double signed_distance):
        raise NotImplementedError


cdef class Union(CSG_Operator):

    cdef bint _m1(self, double signed_distance):
        if signed_distance >= 0:
            return True
        return False

    cdef bint _m2(self, double signed_distance):
        if signed_distance >= 0:
            return True
        return False


cdef class Intersect(CSG_Operator):

    cdef bint _m1(self, double signed_distance):
        if signed_distance <= 0:
            return True
        return False

    cdef bint _m2(self, double signed_distance):
        if signed_distance <= 0:
            return True
        return False


cdef class Subtract(CSG_Operator):

    cdef bint _m1(self, double signed_distance):
        if signed_distance >= 0:
            return True
        return False

    cdef bint _m2(self, double signed_distance):
        if signed_distance <= 0:
            return True
        return False


cpdef Mesh perform_mesh_csg(Mesh mesh_1, Mesh mesh_2, CSG_Operator operator, double tolerance=1e-6):
    """
    Performs a CSG operation on two input Meshes, returning the resulting operation as a new Mesh. 
    
    :param Mesh mesh_1: The first input Mesh.
    :param Mesh mesh_2: The second input Mesh.
    :param CSG_Operator operator: The CSG operation to perform. 
    :param float tolerance: the specifed tolerance level below which intersections will be ignored. 
    :return: A new Mesh instance representing the resulting operation.
    """

    cdef:
        tuple result

    # extract vertex and triangle data for mesh 1
    m1_vertices = mesh_1.data.vertices.copy()
    m1_triangles = mesh_1.data.triangles.copy()
    n_m1_vertices = m1_vertices.shape[0]
    n_m1_triangles = m1_triangles.shape[0]

    # extract vertex and triangle data for mesh 2
    m2_vertices = mesh_2.data.vertices.copy()
    m2_triangles = mesh_2.data.triangles.copy()
    n_m2_vertices = m2_vertices.shape[0]
    n_m2_triangles = m2_triangles.shape[0]

    # calculate triangle centres, longest side length and KDtree for mesh 2
    m2_tri_centres = np.zeros(m2_triangles.shape)
    longest_m2_side = 0
    for m2_tri_id in range(n_m2_triangles):
        t2 = build_triangle(m2_vertices, m2_triangles, m2_tri_id)
        m2_tri_centres[m2_tri_id, :] = t2.vc.x, t2.vc.y, t2.vc.z
        longest_m2_side = max(longest_m2_side, t2.v1.distance_to(t2.v2), t2.v2.distance_to(t2.v3), t2.v3.distance_to(t2.v1))
    m2_kdtree = KDTree(m2_tri_centres)

    ####################################################################################################################
    # FIND ALL INTERSECTIONS

    m1_intersects = np.full((n_m1_triangles), False)
    m1_split_points = {}

    m2_intersects = np.full((n_m2_triangles), False)
    m2_split_points = {}

    for m1_tri_id in range(n_m1_triangles):

        # find all mesh 2 triangles within radius r of mesh 1
        t1 = build_triangle(m1_vertices, m1_triangles, m1_tri_id)
        m2_candidates = m2_kdtree.query_ball_point([t1.vc.x, t1.vc.y, t1.vc.z], longest_m2_side)

        for m2_tri_id in m2_candidates:

            t2 = build_triangle(m2_vertices, m2_triangles, m2_tri_id)

            try:
                result = triangle3d_intersects_triangle3d(t1, t2, tolerance=tolerance)
            except NotImplementedError:
                continue

            if result[0]:

                intsct_pt_1 = result[1]
                intsct_pt_2 = result[2]

                m1_intersects[m1_tri_id] = True
                try:
                    m1_split_points[m1_tri_id]
                except KeyError:
                    m1_split_points[m1_tri_id] = [t1.v1, t1.v2, t1.v3]

                # add intersection points if separation from other split points is greater than tolerance
                distances = [intsct_pt_1.distance_to(pt) for pt in m1_split_points[m1_tri_id]]
                if all([d > tolerance for d in distances]):
                    m1_split_points[m1_tri_id].append(intsct_pt_1)
                distances = [intsct_pt_2.distance_to(pt) for pt in m1_split_points[m1_tri_id]]
                if all([d > tolerance for d in distances]):
                    m1_split_points[m1_tri_id].append(intsct_pt_2)

                m2_intersects[m2_tri_id] = True
                try:
                    m2_split_points[m2_tri_id]
                except KeyError:
                    m2_split_points[m2_tri_id] = [t2.v1, t2.v2, t2.v3]

                # add intersection points if separation from other split points is greater than tolerance
                distances = [intsct_pt_1.distance_to(pt) for pt in m2_split_points[m2_tri_id]]
                if all([d > tolerance for d in distances]):
                    m2_split_points[m2_tri_id].append(intsct_pt_1)
                distances = [intsct_pt_2.distance_to(pt) for pt in m2_split_points[m2_tri_id]]
                if all([d > tolerance for d in distances]):
                        m2_split_points[m2_tri_id].append(intsct_pt_2)

    # convert to numpy arrays
    for key in m1_split_points.keys():
        m1_split_points[key] = np.array(m1_split_points[key])
    for key in m2_split_points.keys():
        m2_split_points[key] = np.array(m2_split_points[key])


    ####################################################################################################################
    # MAKE NEW MESHES

    # extract vertex and triangle data for mesh 1
    m1_new_vertices = m1_vertices.tolist()
    m1_new_triangles = []
    for m1_tri_id in range(n_m1_triangles):
        if not m1_intersects[m1_tri_id]:
            m1_new_triangles.append(m1_triangles[m1_tri_id, :].tolist())

    m2_new_vertices = m2_vertices.tolist()
    m2_new_triangles = []
    for m2_tri_id in range(n_m2_triangles):
        if not m2_intersects[m2_tri_id]:
            m2_new_triangles.append(m2_triangles[m2_tri_id, :].tolist())


    ####################################################################################################################
    # SPLIT INTERSECTING TRIANGLES AND ADD THEM TO THE NEW MESH

    for m1_tri_id in m1_split_points:

        # unpack triangle 1
        t1 = build_triangle(m1_vertices, m1_triangles, m1_tri_id)

        # transform to x-y plane
        m1_transform = translate(t1.v1.x, t1.v1.y, t1.v1.z) * rotate_basis(t1.v1.vector_to(t1.v2), t1.normal)
        m1_inv_transform = m1_transform.inverse()

        num_current_m1_vertices = len(m1_new_vertices)
        vertices_2d = []
        for point in m1_split_points[m1_tri_id]:
            tv = point.transform(m1_inv_transform)
            vertices_2d.append([tv.x, tv.z])
            m1_new_vertices.append([point.x, point.y, point.z])

        # perform Delaunay triangulation
        vertices_2d = np.array(vertices_2d)
        triangle_candidates = Delaunay(vertices_2d).simplices

        new_triangles = []
        for new_tri_id in range(triangle_candidates.shape[0]):
            try:
                tnew = build_triangle(m1_split_points[m1_tri_id], triangle_candidates, new_tri_id)
            except ValueError:
                continue
            if not t1.normal.dot(tnew.normal) > 0:
                triangle_candidates[new_tri_id, 1], triangle_candidates[new_tri_id, 2] = triangle_candidates[new_tri_id, 2], triangle_candidates[new_tri_id, 1]
            if tnew.area > tolerance**2:  # ignore ridiculously small triangles
                new_triangles.append(triangle_candidates[new_tri_id])
        m1_new_triangles.extend(np.array(new_triangles) + num_current_m1_vertices)


    for m2_tri_id in m2_split_points:

        # unpack triangle 2
        t2 = build_triangle(m2_vertices, m2_triangles, m2_tri_id)

        # transform to x-y plane
        m2_transform = translate(t2.v1.x, t2.v1.y, t2.v1.z) * rotate_basis(t2.v1.vector_to(t2.v2), t2.normal)
        m2_inv_transform = m2_transform.inverse()

        num_current_m2_vertices = len(m2_new_vertices)
        vertices_2d = []
        for point in m2_split_points[m2_tri_id]:
            tv = point.transform(m2_inv_transform)
            vertices_2d.append([tv.x, tv.z])
            m2_new_vertices.append([point.x, point.y, point.z])

        # perform Delaunay triangulation
        vertices_2d = np.array(vertices_2d)
        triangle_candidates = Delaunay(vertices_2d).simplices

        new_triangles = []
        for new_tri_id in range(triangle_candidates.shape[0]):
            try:
                tnew = build_triangle(m2_split_points[m2_tri_id], triangle_candidates, new_tri_id)
            except ValueError:
                continue
            if not t2.normal.dot(tnew.normal) > 0:
                triangle_candidates[new_tri_id, 1], triangle_candidates[new_tri_id, 2] = triangle_candidates[new_tri_id, 2], triangle_candidates[new_tri_id, 1]
            if tnew.area > tolerance**2:  # ignore ridiculously small triangles
                new_triangles.append(triangle_candidates[new_tri_id])
        m2_new_triangles.extend(np.array(new_triangles) + num_current_m2_vertices)

    ####################################################################################################################
    # Recompute mesh KDTrees

    # extract vertex and triangle data for mesh 1
    m1_vertices = np.array(m1_new_vertices)
    m1_triangles = np.array(m1_new_triangles)
    n_m1_vertices = m1_vertices.shape[0]
    n_m1_triangles = m1_triangles.shape[0]

    # calculate triangle centres and KDtree for mesh 1
    m1_tri_centres = np.zeros(m1_triangles.shape)
    for m1_tri_id in range(n_m1_triangles):
        t1 = build_triangle(m1_vertices, m1_triangles, m1_tri_id)
        m1_tri_centres[m1_tri_id, :] = t1.vc.x, t1.vc.y, t1.vc.z
    m1_kdtree = KDTree(m1_tri_centres)

    # extract vertex and triangle data for mesh 2
    m2_vertices = np.array(m2_new_vertices)
    m2_triangles = np.array(m2_new_triangles)
    n_m2_vertices = m2_vertices.shape[0]
    n_m2_triangles = m2_triangles.shape[0]

    # calculate triangle centres and KDtree for mesh 2
    m2_tri_centres = np.zeros(m2_triangles.shape)
    for m2_tri_id in range(n_m2_triangles):
        t2 = build_triangle(m2_vertices, m2_triangles, m2_tri_id)
        m2_tri_centres[m2_tri_id, :] = t2.vc.x, t2.vc.y, t2.vc.z
    m2_kdtree = KDTree(m2_tri_centres)

    ####################################################################################################################
    # Perform Signed distance operations

    combined_vertices = m1_vertices.tolist()
    combined_vertices.extend(m2_vertices.tolist())

    combined_triangles = []

    for m1_tri_id in range(n_m1_triangles):

        # unpack vertices for triangle in mesh 1
        t1 = build_triangle(m1_vertices, m1_triangles, m1_tri_id)

        # query closest triangle in mesh 2
        closest_m2_tri = m2_kdtree.query([t1.vc.x, t1.vc.y, t1.vc.z])[1]

        # unpack vertices for closest triangle in mesh 2
        t2 = build_triangle(m2_vertices, m2_triangles, closest_m2_tri)

        d_u_from_v = -t2.normal.dot(t1.vc.vector_to(t2.vc))

        if operator._m1(d_u_from_v):
            combined_triangles.append(m1_triangles[m1_tri_id, :].tolist())

    for m2_tri_id in range(n_m2_triangles):

        # unpack vertices for triangle in mesh 2
        t2 = build_triangle(m2_vertices, m2_triangles, m2_tri_id)

        # query closest triangle in mesh 1
        closest_m1_tri = m1_kdtree.query([t2.vc.x, t2.vc.y, t2.vc.z])[1]

        # unpack vertices for closest triangle in mesh 1
        t1 = build_triangle(m1_vertices, m1_triangles, closest_m1_tri)

        d_v_from_u = -t1.normal.dot(t2.vc.vector_to(t1.vc))

        if operator._m2(d_v_from_u):
            combined_triangles.append((m2_triangles[m2_tri_id, :] + n_m1_vertices).tolist())

    return weld_vertices(Mesh(combined_vertices, combined_triangles))


cdef Triangle build_triangle(np.ndarray vertex_array, np.ndarray triangle_array, int triangle_id):

    cdef:
        int v1, v2, v3
        double v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z

    v1, v2, v3 = triangle_array[triangle_id]

    v1x, v1y, v1z = vertex_array[v1]
    v2x, v2y, v2z = vertex_array[v2]
    v3x, v3y, v3z = vertex_array[v3]

    return Triangle(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z)



