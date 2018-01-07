
from matplotlib.pyplot import *

from raysect.primitive import Sphere, Box, Cylinder
from raysect.optical import World, translate, rotate, Point3D, d65_white, ConstantSF
from raysect.optical.observer import PinholeCamera, RGBPipeline2D, RGBAdaptiveSampler2D
from raysect.optical.material import Lambert, UniformSurfaceEmitter
from raysect.optical.library.spectra.colours import *

world = World()

angle_increments = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
roughness = 0.25
colours = [yellow, orange, red_orange, red, purple, blue, light_blue, cyan, green]

# coloured metal spheres
angle = 6
radius = 0.12

distance = 3.2
for i in range(9):

    increment = angle_increments[i]
    Sphere(radius, world,
           transform=rotate(increment * angle, 0, 0) * translate(0, radius + 0.00001, distance),
           material=Lambert(colours[i]))

# diffuse ground plane
Box(Point3D(-100, -0.1, -100), Point3D(100, 0, 100), world, material=Lambert(ConstantSF(1/1000)))

# four strip lights
Cylinder(0.5, 1.0, world, transform=translate(0.5, 5, 8) * rotate(90, 0, 0),
         material=UniformSurfaceEmitter(d65_white, 1.0))
Cylinder(0.5, 1.0, world, transform=translate(0.5, 5, 6) * rotate(90, 0, 0),
         material=UniformSurfaceEmitter(d65_white, 1.0))
Cylinder(0.5, 1.0, world, transform=translate(0.5, 5, 4) * rotate(90, 0, 0),
         material=UniformSurfaceEmitter(d65_white, 1.0))
Cylinder(0.5, 1.0, world, transform=translate(0.5, 5, 2) * rotate(90, 0, 0),
         material=UniformSurfaceEmitter(d65_white, 1.0))

rgb = RGBPipeline2D(name="sRGB")
sampler = RGBAdaptiveSampler2D(rgb, ratio=10, fraction=0.2, min_samples=500, cutoff=0.05)

# observer
camera = PinholeCamera((512, 256), fov=42, parent=world, transform=translate(0, 3.3, 0) * rotate(0, -47, 0), pipelines=[rgb], frame_sampler=sampler)
camera.spectral_bins = 25
camera.pixel_samples = 250

# start ray tracing
ion()
p = 1
while not camera.render_complete:
    print("Rendering pass {}...".format(p))
    camera.observe()
    print()
    p += 1

ioff()
rgb.display()
