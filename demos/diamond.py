import time
from raysect.optical import World, translate, rotate, Point3D, d65_white, ConstantSF
from raysect.primitive import Sphere, Box, Cylinder, import_obj
from raysect.optical.observer import PinholeCamera, RGBPipeline2D, RGBAdaptiveSampler2D
from raysect.optical.library import RoughIron
from raysect.optical.material import Lambert, UniformVolumeEmitter, Dielectric, Sellmeier
from matplotlib.pyplot import *

# DIAMOND MATERIAL
diamond_material = Dielectric(Sellmeier(0.3306, 4.3356, 0.0, 0.1750**2, 0.1060**2, 0.0), ConstantSF(1))
diamond_material.importance = 1

world = World()

diamond = import_obj("./resources/diamond.obj", scaling=0.01, smoothing=False, parent=world,
                     transform=translate(0.0, 0.713001, 0.0)*rotate(-25, 0, 0), material=diamond_material)

Box(Point3D(-100, -0.1, -100), Point3D(100, 0, 100), world, material=RoughIron(0.04))

# Box(Point3D(-100, -0.1, -100), Point3D(100, 0, 100), world, material=Lambert(ConstantSF(0.5)))

Sphere(0.5, world, transform=translate(-1, 4, -4), material=UniformVolumeEmitter(d65_white, 1.0))

rgb_pipeline = RGBPipeline2D(display_update_time=15, display_unsaturated_fraction=0.985)
sampler = RGBAdaptiveSampler2D(rgb_pipeline, min_samples=500)

camera = PinholeCamera((512, 512), parent=world, transform=translate(0, 4, -3.5) * rotate(0, -46, 0), pipelines=[rgb_pipeline], frame_sampler=sampler)
camera.spectral_bins = 16
camera.spectral_rays = 8
camera.pixel_samples = 250

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
