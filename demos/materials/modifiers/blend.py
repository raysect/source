from matplotlib.pyplot import *

from raysect.optical import World, translate, rotate, Point3D, d65_white
from raysect.primitive import Sphere, Box, Cylinder
from raysect.optical.observer import PinholeCamera, RGBPipeline2D, RGBAdaptiveSampler2D
from raysect.optical.material import Lambert, UniformSurfaceEmitter, Blend
from raysect.optical.library import *

"""
Material Blending
=================

Demonstration of using the Blend modifier to combine different materials. 
"""

world = World()

angle_increments = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
ratio_scan = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]


# glass spheres
angle = 6
distance = 3
radius = 0.15

for i in range(9):
    ratio = ratio_scan[i]
    increment = angle_increments[i]

    # use raw material if ratio = 0
    if ratio == 0:
        material = schott("N-BK7")
    elif ratio == 1.0:
        material = RoughGold(0.02)
    else:
        material = Blend(schott("N-BK7"), RoughGold(0.02), ratio, surface_only=True)

    Sphere(radius, world,
           transform=rotate(increment * angle, 0, 0) * translate(0, radius + 0.00001, distance),
           material=material)

# metal spheres
angle = 6
distance = 3.6
radius = 0.15

for i in range(9):
    ratio = ratio_scan[i]
    increment = angle_increments[i]

    # use raw material if ratio = 0
    if ratio == 0:
        material = RoughIron(0.05)
    elif ratio == 1.0:
        material = RoughCopper(0.05)
    else:
        material = Blend(RoughIron(0.05), RoughCopper(0.05), ratio, surface_only=True)

    Sphere(radius, world,
           transform=rotate(increment * angle, 0, 0) * translate(0, radius + 0.00001, distance),
           material=material)


# diffuse ground plane
Box(Point3D(-100, -0.1, -100), Point3D(100, 0, 100), world, material=Lambert())


# four strip lights
Cylinder(0.5, 1.0, world, transform=translate(0.5, 5, 8) * rotate(90, 0, 0),
         material=UniformSurfaceEmitter(d65_white, 1.0))
Cylinder(0.5, 1.0, world, transform=translate(0.5, 5, 6) * rotate(90, 0, 0),
         material=UniformSurfaceEmitter(d65_white, 1.0))
Cylinder(0.5, 1.0, world, transform=translate(0.5, 5, 4) * rotate(90, 0, 0),
         material=UniformSurfaceEmitter(d65_white, 1.0))
Cylinder(0.5, 1.0, world, transform=translate(0.5, 5, 2) * rotate(90, 0, 0),
         material=UniformSurfaceEmitter(d65_white, 1.0))

# observer
rgb = RGBPipeline2D(display_unsaturated_fraction=0.96)
sampler = RGBAdaptiveSampler2D(rgb, ratio=10, fraction=0.2, min_samples=500, cutoff=0.01)
camera = PinholeCamera((1024, 512), pipelines=[rgb], frame_sampler=sampler, transform=translate(0, 3.3, 0) * rotate(0, -47, 0), fov=42, parent=world)
camera.spectral_rays = 1
camera.spectral_bins = 15
camera.pixel_samples = 250

# start ray tracing
ion()
name = 'modifier_blend'
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
