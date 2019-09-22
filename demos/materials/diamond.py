
import os
import time
from matplotlib.pyplot import *

from raysect.optical import World, translate, rotate, Point3D, d65_white, ConstantSF
from raysect.primitive import Sphere, Box, import_obj
from raysect.optical.observer import PinholeCamera, RGBPipeline2D, RGBAdaptiveSampler2D
from raysect.optical.library import RoughTitanium
from raysect.optical.material import UniformSurfaceEmitter, UniformVolumeEmitter, Dielectric, Sellmeier


# DIAMOND MATERIAL
diamond_material = Dielectric(Sellmeier(0.3306, 4.3356, 0.0, 0.1750**2, 0.1060**2, 0.0), ConstantSF(0.998))
diamond_material.importance = 2

world = World()

base_path = os.path.split(os.path.realpath(__file__))[0]

# the diamond
diamond = import_obj(os.path.join(base_path, "../resources/diamond.obj"), scaling=0.01, smoothing=False, parent=world,
                     transform=translate(0.0, 0.713001, 0.0)*rotate(-10, 0, 0), material=diamond_material)

# floor
Box(Point3D(-100, -0.1, -100), Point3D(100, 0, 100), world, material=RoughTitanium(0.1))

# front light
Sphere(5, world, transform=translate(1, 8, -10), material=UniformVolumeEmitter(d65_white, 1.0))

# fill light
Sphere(10, world, transform=translate(7, 20, 20), material=UniformSurfaceEmitter(d65_white, 0.15))

# camera
rgb = RGBPipeline2D(display_update_time=15, display_unsaturated_fraction=0.995)
sampler = RGBAdaptiveSampler2D(rgb, min_samples=1000, fraction=0.1, cutoff=0.01)
camera = PinholeCamera((1024, 1024), parent=world, transform=translate(0, 4, -3.5) * rotate(0, -46, 0), pipelines=[rgb], frame_sampler=sampler)
camera.spectral_bins = 21
camera.spectral_rays = 21
camera.pixel_samples = 250
camera.ray_max_depth = 10000
camera.ray_extinction_min_depth = 3
camera.ray_extinction_prob = 0.002

# start ray tracing
ion()
name = 'diamond'
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
render_pass = 1
while not camera.render_complete:

    print("Rendering pass {}...".format(render_pass))
    camera.observe()
    rgb.save("{}_{}_pass_{}.png".format(name, timestamp, render_pass))
    print()

    render_pass += 1

ioff()
rgb.display()

