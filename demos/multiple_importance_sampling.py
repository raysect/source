
from matplotlib.pyplot import *

from raysect.primitive import Sphere, Box
from raysect.optical import World, translate, rotate, Point3D, d65_white
from raysect.optical.observer import PinholeCamera, RGBPipeline2D
from raysect.optical.material import UniformSurfaceEmitter
from raysect.optical.library import *


"""
Multiple Importance Sampling Demo
=================================

Raysect recreates of the results of Veach's thesis on multiple importance sampling.

Veach, E., 1997. Robust Monte Carlo methods for lighting simulation
(Doctoral dissertation, Ph. D. thesis, Stanford University).
"""

pixels = (1280, 720)
samples = 100

# set-up scenegraph
world = World()

# background
wall = Box(Point3D(-10, -8, 0), Point3D(10, 8, 0.1), world, transform=translate(0, -1, 10), material=UniformSurfaceEmitter(d65_white, 0.00005))
wall.material.importance = 0

# emitting spheres
Sphere(radius=0.5, parent=world, transform=translate(-2, 3, 2), material=UniformSurfaceEmitter(light_blue))
Sphere(radius=0.2, parent=world, transform=translate(-0.667, 3, 2), material=UniformSurfaceEmitter(green, scale=0.5**2/0.2**2))
Sphere(radius=0.05, parent=world, transform=translate(0.667, 3, 2), material=UniformSurfaceEmitter(orange, scale=0.5**2/0.05**2))
Sphere(radius=0.008, parent=world, transform=translate(2, 3, 2), material=UniformSurfaceEmitter(red, scale=0.5**2/0.008**2))

# reflecting plates
Box(lower=Point3D(-3, -0.1, -0.5), upper=Point3D(3, 0.1, 0.5), parent=world,
    transform=translate(0, 1.5, 2)*rotate(0, 45.5, 0), material=RoughAluminium(0.0003))

Box(lower=Point3D(-3, -0.1, -0.5), upper=Point3D(3, 0.1, 0.5), parent=world,
    transform=translate(0, 0.7, 1)*rotate(0, 32, 0), material=RoughAluminium(0.005))

Box(lower=Point3D(-3, -0.1, -0.5), upper=Point3D(3, 0.1, 0.5), parent=world,
    transform=translate(0, 0.05, 0)*rotate(0, 24.5, 0), material=RoughAluminium(0.03))

Box(lower=Point3D(-3, -0.1, -0.5), upper=Point3D(3, 0.1, 0.5), parent=world,
    transform=translate(0, -0.5, -1)*rotate(0, 19, 0), material=RoughAluminium(0.1))

ion()

# Light sampling
light_sampling = RGBPipeline2D(name="Light Sampling")
light_sampling.display_sensitivity = 200
light_sampling.accumulate = False
camera = PinholeCamera(pixels, fov=45, parent=world, transform=translate(0, 1, -10) * rotate(0, 0, 0), pipelines=[light_sampling])
camera.pixel_samples = samples
camera.ray_importance_sampling = True
camera.ray_important_path_weight = 1.0
camera.observe()

# BRDF sampling
brdf_sampling = RGBPipeline2D(name="BRDF Sampling")
brdf_sampling.display_sensitivity = 200
brdf_sampling.accumulate = False
camera = PinholeCamera(pixels, fov=45, parent=world, transform=translate(0, 1, -10) * rotate(0, 0, 0), pipelines=[brdf_sampling])
camera.pixel_samples = samples
camera.ray_importance_sampling = False
camera.observe()

# MIS sampling
mis_sampling = RGBPipeline2D(name="MIS Sampling")
mis_sampling.display_sensitivity = 200
mis_sampling.accumulate = False
camera = PinholeCamera(pixels, fov=45, parent=world, transform=translate(0, 1, -10) * rotate(0, 0, 0), pipelines=[mis_sampling])
camera.pixel_samples = samples
camera.ray_importance_sampling = True
camera.ray_important_path_weight = 0.5
camera.observe()

# save results
light_sampling.save('mis_light_sampling.png')
brdf_sampling.save('mis_brdf_sampling.png')
mis_sampling.save('mis_combined_sampling.png')

# final display
light_sampling.display()
brdf_sampling.display()
mis_sampling.display()

ioff()
show()
