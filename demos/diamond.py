
import os
import time
from matplotlib.pyplot import *

from raysect.optical import World, translate, rotate, Point3D, d65_white, ConstantSF
from raysect.primitive import Sphere, Box, import_obj
from raysect.optical.observer import PinholeCamera, RGBPipeline2D, RGBAdaptiveSampler2D
from raysect.optical.library import RoughTitanium
from raysect.optical.material import UniformSurfaceEmitter, UniformVolumeEmitter, Dielectric, Sellmeier


# DIAMOND MATERIAL
diamond_material = Dielectric(Sellmeier(0.3306, 4.3356, 0.0, 0.1750**2, 0.1060**2, 0.0), ConstantSF(1))
diamond_material.importance = 2

world = World()

base_path = os.path.split(os.path.realpath(__file__))[0]

# the diamond
diamond = import_obj(os.path.join(base_path, "resources/diamond.obj"), scaling=0.01, smoothing=False, parent=world,
                     transform=translate(0.0, 0.713001, 0.0)*rotate(-10, 0, 0), material=diamond_material)

# floor
Box(Point3D(-100, -0.1, -100), Point3D(100, 0, 100), world, material=RoughTitanium(0.1))

# front light
Sphere(5, world, transform=translate(1, 8, -10), material=UniformVolumeEmitter(d65_white, 1.0))

# fill light
Sphere(10, world, transform=translate(7, 20, 20), material=UniformSurfaceEmitter(d65_white, 0.15))

# camera
rgb_pipeline = RGBPipeline2D(display_update_time=15, display_unsaturated_fraction=0.998)
sampler = RGBAdaptiveSampler2D(rgb_pipeline, min_samples=400, fraction=0.1)
camera = PinholeCamera((1024, 1024), parent=world, transform=translate(0, 4, -3.5) * rotate(0, -46, 0), pipelines=[rgb_pipeline], frame_sampler=sampler)
camera.spectral_bins = 18
camera.spectral_rays = 9
camera.pixel_samples = 200


# start ray tracing
ion()
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
for p in range(1, 1000):
    print("Rendering pass {}...".format(p))
    camera.observe()
    rgb_pipeline.save("demo_diamond_{}_pass_{}.png".format(timestamp, p))
    print()

# display final result
ioff()
rgb_pipeline.display()
show()
