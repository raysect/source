from matplotlib import pyplot as plt

from raysect.optical import ConstantSF, Point3D, World, d65_white, rotate, translate
from raysect.optical.library.metal import Copper
from raysect.optical.material import Lambert, UniformSurfaceEmitter
from raysect.optical.observer import PinholeCamera, RGBAdaptiveSampler2D, RGBPipeline2D
from raysect.primitive import Box, Cylinder, Torus

world = World()

# Torus
torus = Torus(
    1.0,
    0.5,
    world,
    transform=translate(0, 0.0, 0.6),
    material=Copper(),
)

# floor
Box(
    Point3D(-100, -100, -10),
    Point3D(100, 100, 0),
    parent=world,
    material=Lambert(ConstantSF(1.0)),
)

# emitter
Cylinder(
    3.0,
    100.0,
    parent=world,
    transform=translate(0, 0, 8) * rotate(90, 0, 0) * translate(0, 0, -50),
    material=UniformSurfaceEmitter(d65_white, 1.0),
)

# camera
rgb = RGBPipeline2D(display_unsaturated_fraction=0.995)
sampler = RGBAdaptiveSampler2D(rgb, min_samples=500, fraction=0.1, cutoff=0.01)
camera = PinholeCamera(
    (512, 512),
    parent=world,
    transform=rotate(0, 45, 0) * translate(0, 0, 5) * rotate(0, -180, 0),
    pipelines=[rgb],
    frame_sampler=sampler,
)
camera.spectral_bins = 21
camera.spectral_rays = 1
camera.pixel_samples = 250
camera.ray_max_depth = 10000
camera.ray_extinction_min_depth = 3
camera.ray_extinction_prob = 0.01


# start ray tracing
plt.ion()
for p in range(0, 1000):
    print(f"Rendering pass {p}...")
    camera.observe()
    print()

plt.ioff()
rgb.display()
plt.show()
