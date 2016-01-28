from raysect.optical import World, translate, rotate, Point3D, Vector3D, Ray, d65_white, ConstantSF, InterpolatedSF
from raysect.optical.observer import PinholeCamera
from raysect.optical.material.emitter import UniformSurfaceEmitter, Checkerboard
from raysect.optical.library import schott
from raysect.primitive import Sphere, Box, Cylinder, Union, Intersect, Subtract
from matplotlib.pyplot import *

# kludge to fix matplotlib 1.4 ion() idiocy
import sys
sys.ps1 = 'SOMETHING'
ion()

world = World()

Box(Point3D(-10, -10, 4.0), Point3D(10, 10, 4.1), world, material=Checkerboard(1, d65_white, d65_white, 0.2, 0.8))
#Box(Point3D(-100, -100, -100), Point3D(100, 100, 100), world, material=UniformSurfaceEmitter(d65_white, 0.1))

cyl_x = Cylinder(1, 4.2, transform=rotate(90, 0, 0)*translate(0, 0, -2.1))
cyl_y = Cylinder(1, 4.2, transform=rotate(0, 90, 0)*translate(0, 0, -2.1))
cyl_z = Cylinder(1, 4.2, transform=rotate(0, 0, 0)*translate(0, 0, -2.1))
cube = Box(Point3D(-1.5, -1.5, -1.5), Point3D(1.5, 1.5, 1.5))
sphere = Sphere(2.0)
#target = Intersect(sphere, Subtract(cube, Union(Union(cyl_x, cyl_y), cyl_z)), world, translate(0,0,0)*rotate(0, 0, 0), schott("N-BK7"))
target = Intersect(sphere, cube, world, translate(0,0,0)*rotate(0, 0, 0), schott("N-BK7"))

camera = PinholeCamera(fov=45, parent=world, transform=translate(0, 0, -6) * rotate(0, 0, 0))
camera.ray_max_depth = 15
camera.rays = 9
camera.spectral_samples = 3
camera.pixels = (256, 256)
camera.display_progress = False

num_frames = 25*20
full_rotation = 360
for frame in range(num_frames):

    print("Rendering frame {}:".format(frame))

    rotation = full_rotation / num_frames * frame
    target.transform = rotate(rotation, 25, 5)

    camera.observe()
    camera.save("renders/anim/frame{:04}.png".format(frame))
    camera.display()
    show()
