
import matplotlib.pyplot as plt

from raysect.optical import World, translate, rotate, Point3D, d65_white
from raysect.optical.observer import PinholeCamera
from raysect.optical.material.emitter import Checkerboard
from raysect.optical.library import schott
from raysect.primitive import Sphere, Box, Cylinder, Intersect

plt.ion()

world = World()


# Background checkerboard lightsource
Box(Point3D(-10, -10, 4.0), Point3D(10, 10, 4.1), world,
    material=Checkerboard(1, d65_white, d65_white, 0.2, 0.8))


# Build a CSG primitive from a number of basic underlying primitives
cyl_x = Cylinder(1, 4.2, transform=rotate(90, 0, 0)*translate(0, 0, -2.1))
cyl_y = Cylinder(1, 4.2, transform=rotate(0, 90, 0)*translate(0, 0, -2.1))
cyl_z = Cylinder(1, 4.2, transform=rotate(0, 0, 0)*translate(0, 0, -2.1))
cube = Box(Point3D(-1.5, -1.5, -1.5), Point3D(1.5, 1.5, 1.5))
sphere = Sphere(2.0)
target = Intersect(sphere, cube, parent=world, transform=translate(0, 0, 0)*rotate(0, 0, 0),
                   material=schott("N-BK7"))


# create and setup the camera
camera = PinholeCamera((256, 256), fov=45, parent=world, transform=translate(0, 0, -6) * rotate(0, 0, 0))
camera.spectral_rays = 9
camera.spectral_bins = 30
rgb = camera.pipelines[0]

# for each frame rotate the CSG primitive and re-render
num_frames = 25*20
full_rotation = 360
for frame in range(num_frames):

    print("Rendering frame {}:".format(frame))

    rotation = full_rotation / num_frames * frame
    target.transform = rotate(rotation, 25, 5)

    camera.observe()
    rgb.save("frame{:04}.png".format(frame))

