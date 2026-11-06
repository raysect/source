import math
import unittest
import numpy as np
from raysect.core import Vector3D
from raysect.optical.material import CustomBRDF


def direction(theta, phi=0):
    return Vector3D(math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta))


class TestCustomBRDF(unittest.TestCase):
    def test_invalid_table_and_direction(self):
        for values in (np.ones((1, 2, 2)), np.full((2, 2, 2), -1), np.full((2, 2, 2), np.nan)):
            with self.assertRaises(ValueError):
                CustomBRDF(values)
        self.assertEqual(CustomBRDF(np.ones((2, 2, 2))).bsdf(Vector3D(0, 0, 0), direction(0), 500), 0)

    def test_constant_and_boundaries(self):
        material = CustomBRDF(np.full((4, 8, 5), 0.4 / math.pi))
        for ti in (0, 0.7, math.pi / 2):
            for to in (0, 0.6, math.pi / 2):
                for phi in (-1e-10, 0, 1e-10, math.pi):
                    self.assertAlmostEqual(material.bsdf(direction(ti), direction(to, phi), 500), 0.4 / math.pi)
        self.assertEqual(material.bsdf(direction(0), Vector3D(0, 0, -1), 500), 0)

    def test_interpolation_and_periodic_azimuth(self):
        values = np.zeros((3, 4, 3))
        values[:, 0, :] = 2
        values[:, 3, :] = 4
        material = CustomBRDF(values)
        self.assertAlmostEqual(material.bsdf(direction(0.3), direction(0.4, -math.pi / 4), 500), 3)
        self.assertAlmostEqual(material.bsdf(direction(0.3), direction(0.4, 2 * math.pi - 1e-9), 500),
                               material.bsdf(direction(0.3), direction(0.4, 1e-9), 500), places=7)

    def test_sampling_matches_cosine_pdf_and_lambert_estimate(self):
        material = CustomBRDF(np.full((3, 4, 3), 0.4 / math.pi))
        incoming = direction(0.5)
        zs = []
        for _ in range(5000):
            outgoing = material.sample(incoming, False)
            self.assertAlmostEqual(material.pdf(incoming, outgoing, False), outgoing.z / math.pi)
            zs.append(outgoing.z)
            estimate = material.bsdf(incoming, outgoing, 500) * outgoing.z / material.pdf(incoming, outgoing, False)
            self.assertAlmostEqual(estimate, 0.4)
        self.assertAlmostEqual(np.mean(zs), 2 / 3, delta=0.02)

class TestRayTracing(unittest.TestCase):
    def test_constant_brdf_in_uniform_enclosure(self):
        from raysect.optical import World, Ray, Point3D, ConstantSF
        from raysect.optical.material import UniformSurfaceEmitter
        from raysect.primitive import Box
        world = World()
        Box(Point3D(-10, -10, -10), Point3D(10, 10, 10), world,
            material=UniformSurfaceEmitter(ConstantSF(1)))
        Box(Point3D(-2, -2, -1), Point3D(2, 2, 0), world,
            material=CustomBRDF(np.full((4, 8, 4), 0.4 / math.pi)))
        ray = Ray(origin=Point3D(0, 0, 1), direction=Vector3D(0, 0, -1),
                  min_wavelength=999, max_wavelength=1001, bins=1,
                  importance_sampling=False)
        for _ in range(20):
            self.assertAlmostEqual(ray.trace(world).samples[0], 0.4, places=6)

class TestSpectralCustomBRDF(unittest.TestCase):
    def test_wavelength_interpolation_and_clamping(self):
        values = np.stack([np.full((3, 4, 3), x) for x in (0.1, 0.3, 0.9)])
        material = CustomBRDF(values, [400, 500, 800])
        values[:] = 0  # constructor retains its own copy
        v = direction(0.4)
        for wavelength, expected in ((300, 0.1), (400, 0.1), (450, 0.2),
                                     (500, 0.3), (650, 0.6), (800, 0.9), (900, 0.9)):
            self.assertAlmostEqual(material.bsdf(v, v, wavelength), expected)
        single = CustomBRDF(np.full((1, 2, 2, 2), 0.25), [1000])
        self.assertAlmostEqual(single.bsdf(v, v, 500), 0.25)
        self.assertAlmostEqual(single.bsdf(v, v, 1500), 0.25)

    def test_multilinear_interpolation_with_varying_axes(self):
        # The expected value is evaluated independently from an affine field.
        ti = np.linspace(0, math.pi / 2, 3)[:, None, None]
        phi = np.arange(4)[None, :, None] * math.pi / 2
        to = np.linspace(0, math.pi / 2, 3)[None, None, :]
        base = 0.2 + 0.1 * ti + 0.05 * phi + 0.03 * to
        material = CustomBRDF(np.stack([base, base + 0.4]), [400, 800])
        for incoming, outgoing, azimuth, wavelength in ((0.2, 0.6, 0.4, 500),
                                                       (0.9, 1.0, 2.3, 700)):
            expected = 0.2 + 0.1 * incoming + 0.05 * azimuth + 0.03 * outgoing + (wavelength - 400) / 1000
            actual = material.bsdf(direction(incoming), direction(outgoing, azimuth), wavelength)
            self.assertAlmostEqual(actual, expected)

    def test_invalid_wavelengths(self):
        values = np.ones((2, 2, 2, 2))
        for wavelengths in (None, [400], [400, 400], [500, 400], [0, 500], [400, float('nan')]):
            with self.assertRaises(ValueError):
                CustomBRDF(values, wavelengths)

    def test_traced_spectrum_uses_each_bin_wavelength(self):
        from raysect.optical import World, Ray, Point3D, ConstantSF
        from raysect.optical.material import UniformSurfaceEmitter
        from raysect.primitive import Box
        values = np.stack([np.full((3, 4, 3), x / math.pi) for x in (0.2, 0.8)])
        world = World()
        Box(Point3D(-10, -10, -10), Point3D(10, 10, 10), world,
            material=UniformSurfaceEmitter(ConstantSF(1)))
        Box(Point3D(-2, -2, -1), Point3D(2, 2, 0), world,
            material=CustomBRDF(values, [400, 800]))
        ray = Ray(origin=Point3D(0, 0, 1), direction=Vector3D(0, 0, -1),
                  min_wavelength=400, max_wavelength=800, bins=4, importance_sampling=False)
        np.testing.assert_allclose(ray.trace(world).samples, [0.275, 0.425, 0.575, 0.725], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
