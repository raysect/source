
from matplotlib.pyplot import *

from raysect.primitive import Sphere, Box, Cylinder
from raysect.optical import World, translate, rotate, Point3D, d65_white, ConstantSF
from raysect.optical.observer import PinholeCamera
from raysect.optical.library import Gold, Silver, Copper, Titanium, Aluminium
from raysect.optical.material import Lambert, UniformSurfaceEmitter


world = World()

Sphere(0.5, world, transform=translate(1.2, 0.5001, 0.6), material=Gold())
Sphere(0.5, world, transform=translate(0.6, 0.5001, -0.6), material=Silver())
Sphere(0.5, world, transform=translate(0, 0.5001, 0.6), material=Copper())
Sphere(0.5, world, transform=translate(-0.6, 0.5001, -0.6), material=Titanium())
Sphere(0.5, world, transform=translate(-1.2, 0.5001, 0.6), material=Aluminium())

Box(Point3D(-100, -0.1, -100), Point3D(100, 0, 100), world, material=Lambert(ConstantSF(1.0)))
Cylinder(3.0, 8.0, world, transform=translate(4, 8, 0) * rotate(90, 0, 0), material=UniformSurfaceEmitter(d65_white, 1.0))

camera = PinholeCamera((512, 512), parent=world, transform=translate(0, 4, -3.5) * rotate(0, -48, 0))
camera.spectral_bins = 15
camera.pixel_samples = 100

# start ray tracing
ion()
for p in range(1, 1000):
    print("Rendering pass {}...".format(p))
    camera.observe()
    camera.pipelines[0].save("demo_metal_{}_samples.png".format(p))
    print()

# display final result
ioff()
camera.display()
show()
