from raysect.optical import World, translate, rotate, Point3D, d65_white, ConstantSF
from raysect.primitive import Sphere, Box, Cylinder, import_obj
from raysect.optical.observer import PinholeCamera, RGBPipeline2D, RGBAdaptiveSampler2D
from raysect.optical.library import RoughAluminium
from raysect.optical.material import Lambert, UniformVolumeEmitter, Dielectric, Sellmeier
from matplotlib.pyplot import *

# DIAMOND MATERIAL
diamond_material = Dielectric(Sellmeier(0.3306, 4.3356, 0.0, 0.1750**2, 0.1060**2, 0.0), ConstantSF(0.25))
diamond_material.importance = 2

world = World()

diamond = import_obj("./resources/diamond.obj", scaling=0.01, smoothing=False, parent=world,
                     transform=translate(0.4, 0.713001, 0.0)*rotate(-10, 0, 0), material=diamond_material)

Box(Point3D(-100, -0.1, -100), Point3D(100, 0, 100), world, material=RoughAluminium(0.25))

# Box(Point3D(-100, -0.1, -100), Point3D(100, 0, 100), world, material=Lambert(ConstantSF(0.5)))

Cylinder(0.5, 1, world, transform=translate(5, 6, 4) * rotate(90, 0, 0), material=UniformVolumeEmitter(d65_white, 1.0))

rgb_pipeline = RGBPipeline2D(display_update_time=15, display_unsaturated_fraction=0.995)
sampler = RGBAdaptiveSampler2D(rgb_pipeline, min_samples=300)

camera = PinholeCamera((800, 800), parent=world, transform=translate(0, 4, -3.5) * rotate(0, -48, 0), pipelines=[rgb_pipeline], frame_sampler=sampler)
camera.spectral_bins = 16
camera.spectral_rays = 16
camera.pixel_samples = 150

# start ray tracing
ion()
for p in range(1, 1000):
    print("Rendering pass {}...".format(p))
    camera.observe()
    rgb_pipeline.save("demo_diamond_pass_{}.png".format(p))
    print()

# display final result
ioff()
rgb_pipeline.display()
show()
