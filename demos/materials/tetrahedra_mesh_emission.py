import os
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from raysect.core.math.function.float.function3d.interpolate import Discrete3DMesh
from raysect.optical import Point3D, Vector3D, World, rotate_basis, translate
from raysect.optical.library import RoughTitanium
from raysect.optical.material import InhomogeneousVolumeEmitter
from raysect.optical.observer import PinholeCamera, RGBAdaptiveSampler2D, RGBPipeline2D
from raysect.primitive import Box, TetraMeshData

RESOURCE_PATH = Path(__file__).parent.parent / "resources"


class UnityEmitter(InhomogeneousVolumeEmitter):

    def __init__(self, radiation_function):
        super().__init__()
        self.radiation_function = radiation_function

    def emission_function(self, point, direction, spectrum, world, ray, primitive, to_local, to_world):
        spectrum = ray.new_spectrum()
        spectrum.samples[:] = self.radiation_function(point.x, point.y, point.z)
        return spectrum


# Rabbit tetrahedra mesh emitter
data = np.load(RESOURCE_PATH / "stanford_bunny_tetra_mesh.npz")
vertices = data["vertices"]
tetrahedra = data["tetrahedra"]
print("Creating tetra mesh...")
mesh = TetraMeshData(vertices, tetrahedra)
tetra = Discrete3DMesh.from_mesh(mesh, np.ones(mesh.tetrahedra.shape[0]), limit=False, default_value=0.0)
print("Done.")

# scene
world = World()
bbox = mesh.bounding_box(world.to(world))
emitter = Box(bbox.lower, bbox.upper, material=UnityEmitter(tetra), parent=world)
floor = Box(Point3D(-100, -0.1, -100), Point3D(100, -0.01, 100), world, material=RoughTitanium(0.1))

# camera
camera_pos = Point3D(-0.015, 0.204, 0.25)
rgb_pipeline = RGBPipeline2D(display_update_time=5)
sampler = RGBAdaptiveSampler2D(rgb_pipeline, min_samples=100, fraction=0.2)
camera = PinholeCamera(
    (512, 512),
    parent=world,
    pipelines=[rgb_pipeline],
    frame_sampler=sampler,
    transform=(
        translate(camera_pos.x, camera_pos.y, camera_pos.z)
        * rotate_basis(
            camera_pos.vector_to(Point3D(0, 0.06, 0)),
            Vector3D(0, 1, 0)
        )
    )
)
camera.fov = 50
camera.spectral_bins = 1
camera.spectral_rays = 1
camera.pixel_samples = 200

# integration resolution
emitter.material.integrator.step = 0.0001

# start ray tracing
os.nice(15)
plt.ion()
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
for p in range(1, 100):
    print("Rendering pass {}...".format(p))
    camera.observe()
    rgb_pipeline.save("demo_volume_{}_pass_{}.png".format(timestamp, p))
    print()

# display final result
plt.ioff()
rgb_pipeline.display()
plt.show()
