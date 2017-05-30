
from raysect.optical import World, translate, rotate, Point3D
from raysect.optical.observer import PinholeCamera, RGBPipeline2D
from raysect.optical.material import UniformSurfaceEmitter, Lambert
from raysect.optical.library import *
from raysect.primitive import Sphere, Box, Cylinder
from matplotlib.pyplot import *

"""
Multiple Importance Sampling Demo
=================================

Raysect recreates of the results of Veach's thesis on multiple importance sampling.

Veach, E., 1997. Robust Monte Carlo methods for lighting simulation
(Doctoral dissertation, Ph. D. thesis, Stanford University).
"""


pixels = (128*4, 72*4)
# pixels = (1280, 720)
samples = 5

# set-up scenegraph
world = World()

# floor, wall and fill light
# Box(Point3D(-100, -0.1, -100), Point3D(100, 0, 100), world, transform=translate(0, -1, 0), material=Lambert(ConstantSF(1.0)))
# Box(Point3D(-100, -0.1, -100), Point3D(100, 0, 100), world, transform=translate(0, 0, 15) * rotate(0, 90, 0), material=Lambert(ConstantSF(1.0)))
# Cylinder(3.0, 8.0, world, transform=translate(4, 10, -8) * rotate(90, 0, 0), material=UniformSurfaceEmitter(d65_white, 0.005))

# emitting spheres
Sphere(radius=0.5, parent=world, transform=translate(-2, 3, 2), material=UniformSurfaceEmitter(light_blue))
Sphere(radius=0.2, parent=world, transform=translate(0, 3, 2), material=UniformSurfaceEmitter(green))
Sphere(radius=0.05, parent=world, transform=translate(2, 3, 2), material=UniformSurfaceEmitter(orange))

# reflecting plates
Box(lower=Point3D(-3, -0.1, -0.5), upper=Point3D(3, 0.1, 0.5), parent=world,
    transform=translate(0, 1.5, 2)*rotate(0, 45.5, 0), material=RoughAluminium(0.001))

Box(lower=Point3D(-3, -0.1, -0.5), upper=Point3D(3, 0.1, 0.5), parent=world,
    transform=translate(0, 0.7, 1)*rotate(0, 32, 0), material=RoughAluminium(0.01))

Box(lower=Point3D(-3, -0.1, -0.5), upper=Point3D(3, 0.1, 0.5), parent=world,
    transform=translate(0, 0.05, 0)*rotate(0, 24.5, 0), material=RoughAluminium(0.04))

Box(lower=Point3D(-3, -0.1, -0.5), upper=Point3D(3, 0.1, 0.5), parent=world,
    transform=translate(0, -0.5, -1)*rotate(0, 19, 0), material=RoughAluminium(0.12))

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

# final display
light_sampling.display()
brdf_sampling.display()
ioff()
mis_sampling.display()
