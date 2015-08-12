from raysect.core.acceleration import Unaccelerated
from raysect.optical import World, translate, rotate, Point, Vector, Ray, d65_white, ConstantSF, InterpolatedSF
from raysect.optical.observer.camera import PinholeCamera
from raysect.optical.material.emitter import UniformVolumeEmitter, UniformSurfaceEmitter, Checkerboard
from raysect.optical.material.dielectric import Dielectric, Sellmeier
from raysect.optical.material.lambert import Lambert
from raysect.optical.material.glass_libraries import schott
from raysect.primitive import Sphere, Box, Cylinder, Union, Intersect, Subtract
from matplotlib.pyplot import *
from numpy import array

world = World()

cyl_x = Cylinder(1, 4.2, transform=rotate(90, 0, 0)*translate(0, 0, -2.1))
cyl_y = Cylinder(1, 4.2, transform=rotate(0, 90, 0)*translate(0, 0, -2.1))
cyl_z = Cylinder(1, 4.2, transform=rotate(0, 0, 0)*translate(0, 0, -2.1))
cube = Box(Point(-1.5, -1.5, -1.5), Point(1.5, 1.5, 1.5))
sphere = Sphere(2.0)

# Intersect(sphere, Subtract(cube, Union(Union(cyl_x, cyl_y), cyl_z)), world, translate(0, 0, 1)*rotate(30, -20, 0), Lambert())
#
# Box(Point(-50, -50, 50), Point(50, 50, 50.1), world, material=Checkerboard(4, d65_white, d65_white, 0.4, 0.8))
# Box(Point(-100, -100, -100), Point(100, 100, 100), world, material=UniformSurfaceEmitter(d65_white, 0.1))

Sphere(1.5, world, translate(0, 0.0001, 0)*rotate(20, 0, 0), schott("N-BK7"))
#Sphere(1.5, world, translate(0, 0.0001, 0)*rotate(20, 0, 0), Lambert())

# Intersect(sphere, Subtract(cube, Union(Union(cyl_x, cyl_y), cyl_z)), world, translate(0, 0.0001, 2)*rotate(20, 0, 0), schott("N-BK7"))
# Intersect(sphere, Subtract(cube, Union(Union(cyl_x, cyl_y), cyl_z)), world, translate(0, 0.0001, 2)*rotate(20, 0, 0), Lambert())

Box(Point(-50, -1.51, -50), Point(50, -1.5, 50), world, material=Lambert())


# lights
Box(Point(-10, -10, 10), Point(10, 10, 10.1), world, material=Checkerboard(4, d65_white, d65_white, 0.1, 2.0), transform=rotate(45,0,0))
#Sphere(0.5, world, material=UniformSurfaceEmitter(d65_white, 10), transform=translate(2,1.5,2))
#Box(Point(-100, -100, -100), Point(100, 100, 100), world, material=UniformSurfaceEmitter(d65_white, 0.8))



ion()
camera = PinholeCamera(fov=45, parent=world, transform=translate(0, 10, -10) * rotate(0, -45, 0))
camera.ray_min_depth = 3
camera.ray_max_depth = 100
camera.ray_extinction_prob = 0.1
camera.pixel_samples = 2500
camera.rays = 1
camera.spectral_samples = 20
camera.pixels = (512, 512)
camera.display_progress = True
camera.display_update_time = 10
camera.sub_sample = True
camera.observe()

ioff()
camera.save("render.png")
camera.display()
show()

