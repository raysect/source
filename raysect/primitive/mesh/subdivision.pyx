
from raysect.core cimport Point3D, new_point3d


cpdef Mesh subdivide(Mesh mesh, int subdivision_count=3):
    """
    Returns a new Mesh that has been globally subdivided by the specified amount.

    .. Warning::
       Because this operation is recursive it can quickly lead to very large meshes.
       It should be used with care.

    :param int subdivision_count: The number of recursive subdivisions to perform.
    :return: A new Mesh object that has been subdivided to the specified resolution.
    """

    cdef:
        list vertices, triangles
        int i, j, num_vertices, num_triangles
        int v0_id, v1_id, v2_id
        Point3D v0, v1, v2, v3, v4, v5

    vertices = mesh.data.vertices.tolist()
    triangles = mesh.data.triangles.tolist()

    num_vertices = len(vertices)
    num_triangles = len(triangles)
    for i in range(subdivision_count):
        for j in range(num_triangles):
            triangle = triangles[j]
            # extract current triangle vertices
            v0_id = triangle[0]
            v1_id = triangle[1]
            v2_id = triangle[2]
            v0 = new_point3d(vertices[v0_id][0], vertices[v0_id][1], vertices[v0_id][2])
            v1 = new_point3d(vertices[v1_id][0], vertices[v1_id][1], vertices[v1_id][2])
            v2 = new_point3d(vertices[v2_id][0], vertices[v2_id][1], vertices[v2_id][2])

            # subdivide with three new vertices
            v3 = v0 + v0.vector_to(v1) * 0.5
            v3_id = num_vertices
            v4 = v1 + v1.vector_to(v2) * 0.5
            v4_id = num_vertices + 1
            v5 = v2 + v2.vector_to(v0) * 0.5
            v5_id = num_vertices + 2
            vertices.append([v3.x, v3.y, v3.z])
            vertices.append([v4.x, v4.y, v4.z])
            vertices.append([v5.x, v5.y, v5.z])

            # ... and three new faces
            triangles[j] = [v0_id, v3_id, v5_id]  # replace the first face
            triangles.append([v3_id, v1_id, v4_id])
            triangles.append([v4_id, v2_id, v5_id])
            triangles.append([v3_id, v4_id, v5_id])

            num_vertices += 3
            num_triangles += 3

    # TODO pass kdtree parameters such as kdtree_max_depth, etc.
    return Mesh(vertices, triangles, smoothing=mesh.data.smoothing, closed=mesh.data.closed,
                parent=mesh.parent, transform=mesh.transform, material=mesh.material, name=mesh.name)




