import pickle
import unittest

from raysect.core.math import AffineMatrix3D, Point3D
from raysect.primitive.mesh.tetra_mesh import TetraMeshData

# Configure Test Framework


class TestTetraMeshData(unittest.TestCase):
    def setUp(self):
        # Define sample points and a single tetrahedron.
        # Points of a unit tetrahedron: volume should be 1/6.
        self.points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        self.tetrahedra = [
            (0, 1, 2, 3),
        ]
        self.mesh = TetraMeshData(self.points, self.tetrahedra)

    def test_initialization(self):
        # Test that the mesh is created with the expected number of points and tetrahedra.
        self.assertEqual(len(self.mesh.vertices), 4)
        self.assertEqual(len(self.mesh.tetrahedra), 1)

    def test_invalid_tetrahedron_indices(self):
        # Test that constructing a mesh with tetrahedron indices out of bounds raises an error.
        invalid_tetrahedra = [(0, 1, 2, 5)]  # Index 5 is out-of-range.
        with self.assertRaises(IndexError):
            TetraMeshData(self.points, invalid_tetrahedra)

    def test_vertex_method(self):
        # Test that the vertex method returns the correct point.
        vertex0 = self.mesh.vertex(0)
        self.assertAlmostEqual(vertex0.x, 0.0, places=5)
        self.assertAlmostEqual(vertex0.y, 0.0, places=5)
        self.assertAlmostEqual(vertex0.z, 0.0, places=5)

    def test_invalid_vertex_index(self):
        # Test that accessing a vertex with an out-of-range index raises an IndexError.
        with self.assertRaises(IndexError):
            self.mesh.vertex(10)

    def test_barycenter(self):
        # Test that barycenter of the tetrahedron is correctly computed.
        # Barycenter is the average of the four vertices.
        expected_barycenter = (
            (0.0 + 1.0 + 0.0 + 0.0) / 4,
            (0.0 + 0.0 + 1.0 + 0.0) / 4,
            (0.0 + 0.0 + 0.0 + 1.0) / 4,
        )
        barycenter = self.mesh.barycenter(0)
        self.assertAlmostEqual(barycenter.x, expected_barycenter[0], places=5)
        self.assertAlmostEqual(barycenter.y, expected_barycenter[1], places=5)
        self.assertAlmostEqual(barycenter.z, expected_barycenter[2], places=5)

    def test_compute_volume(self):
        # Assume TetraMeshData has a volume method that returns the volume of specified tetrahedron
        # and a volume_total method that returns the total volume of all tetrahedra.
        # The volume of the tetrahedron with the provided vertices is 1/6.
        expected_volume = 1 / 6
        volume = self.mesh.volume(0)
        self.assertAlmostEqual(volume, expected_volume, places=5)

        total_volume = self.mesh.volume_total()
        self.assertAlmostEqual(total_volume, expected_volume, places=5)

    def test_is_contained(self):
        # Test that is_contained returns True for a point inside the tetrahedron
        # and False for a point outside.
        inside_point = Point3D(0.1, 0.1, 0.1)
        outside_point = Point3D(1.0, 1.0, 1.0)
        self.assertTrue(self.mesh.is_contained(inside_point))
        self.assertFalse(self.mesh.is_contained(outside_point))

    def test_bounding_box(self):
        # Test the bounding box computation using an identity transformation.
        # Construct an identity affine matrix.
        identity = AffineMatrix3D([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        bbox = self.mesh.bounding_box(identity)
        # The tetrahedron vertices span from (0,0,0) to (1,1,1)
        # and the bounding box is padded by BOX_PADDING (1e-6) on each side.
        padding = 1e-6
        self.assertAlmostEqual(bbox.lower.x, 0.0 - padding, places=5)
        self.assertAlmostEqual(bbox.lower.y, 0.0 - padding, places=5)
        self.assertAlmostEqual(bbox.lower.z, 0.0 - padding, places=5)
        self.assertAlmostEqual(bbox.upper.x, 1.0 + padding, places=5)
        self.assertAlmostEqual(bbox.upper.y, 1.0 + padding, places=5)
        self.assertAlmostEqual(bbox.upper.z, 1.0 + padding, places=5)

    def test_pickle_state(self):
        # Test that the mesh can be pickled and unpickled without loss of state.
        state = pickle.dumps(self.mesh)
        new_mesh = pickle.loads(state)
        self.assertEqual(len(new_mesh.vertices), len(self.mesh.vertices))
        self.assertEqual(len(new_mesh.tetrahedra), len(self.mesh.tetrahedra))
        self.assertAlmostEqual(new_mesh.volume(0), self.mesh.volume(0), places=5)
        # Verify barycenter consistency.
        bary_orig = self.mesh.barycenter(0)
        bary_new = new_mesh.barycenter(0)
        self.assertAlmostEqual(bary_new.x, bary_orig.x, places=5)
        self.assertAlmostEqual(bary_new.y, bary_orig.y, places=5)
        self.assertAlmostEqual(bary_new.z, bary_orig.z, places=5)


if __name__ == "__main__":
    unittest.main()
