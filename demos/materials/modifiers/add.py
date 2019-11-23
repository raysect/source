from matplotlib.pyplot import *

from raysect.optical import World, translate, rotate, Point3D, d65_white
from raysect.primitive import Sphere, Box, Cylinder
from raysect.optical.observer import PinholeCamera, RGBPipeline2D, RGBAdaptiveSampler2D
from raysect.optical.material import Lambert, UniformSurfaceEmitter, UniformVolumeEmitter, Add
from raysect.optical.library import *

"""
Material Addition
=================

Demonstration of using the Add modifier to combine different materials. 
"""

world = World()

angle_increments = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
temperature_scan = np.linspace(0, 2000, 9) + 273.15

# metal spheres
angle = 6
distance = 3.6
radius = 0.15

for i in range(9):
    temperature = temperature_scan[i]
    increment = angle_increments[i]

    material = Add(RoughIron(0.05), UniformSurfaceEmitter(BlackBody(temperature)), surface_only=True)

    Sphere(radius, world,
           transform=rotate(increment * angle, 0, 0) * translate(0, radius + 0.00001, distance),
           material=material)

# glass spheres
angle = 6
distance = 3
radius = 0.15

for i in range(9):
    temperature = temperature_scan[i]
    increment = angle_increments[i]

    # WARNING: This is an unphysical demo, the attenuation of the glass is not applied to the black body emission.
    # WARNING: The full volume emission is simply added to the light transmitted through the glass.
    # WARNING: In practice, only all emitting volumes or all absorbing volumes should be physically combined.
    # WARNING: This demo highlights the risks of using modifiers without considering the raytracing process.
    material = Add(schott("N-BK7"), UniformVolumeEmitter(BlackBody(temperature), scale=10.0), volume_only=True)

    Sphere(radius, world,
           transform=rotate(increment * angle, 0, 0) * translate(0, radius + 0.00001, distance),
           material=material)

# diffuse ground plane
Box(Point3D(-100, -0.1, -100), Point3D(100, 0, 100), world, material=Lambert())


# four strip lights
Cylinder(0.5, 1.0, world, transform=translate(0.5, 5, 8) * rotate(90, 0, 0),
         material=UniformSurfaceEmitter(d65_white, 0.5))
Cylinder(0.5, 1.0, world, transform=translate(0.5, 5, 6) * rotate(90, 0, 0),
         material=UniformSurfaceEmitter(d65_white, 0.5))
Cylinder(0.5, 1.0, world, transform=translate(0.5, 5, 4) * rotate(90, 0, 0),
         material=UniformSurfaceEmitter(d65_white, 0.5))
Cylinder(0.5, 1.0, world, transform=translate(0.5, 5, 2) * rotate(90, 0, 0),
         material=UniformSurfaceEmitter(d65_white, 0.5))

# observer
rgb = RGBPipeline2D(display_unsaturated_fraction=0.85)
sampler = RGBAdaptiveSampler2D(rgb, ratio=10, fraction=0.2, min_samples=500, cutoff=0.01)
camera = PinholeCamera((1024, 512), pipelines=[rgb], frame_sampler=sampler, transform=translate(0, 3.3, 0) * rotate(0, -47, 0), fov=42, parent=world)
camera.spectral_rays = 1
camera.spectral_bins = 15
camera.pixel_samples = 250

# start ray tracing
ion()
name = 'modifier_add'
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

