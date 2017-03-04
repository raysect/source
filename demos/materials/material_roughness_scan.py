
from raysect.optical import World, translate, rotate, Point3D, d65_white
from raysect.primitive import Sphere, Box, Cylinder
from raysect.optical.observer import PinholeCamera, AutoExposure
from raysect.optical.material import Lambert, UniformSurfaceEmitter, Roughen
from raysect.optical.library import *
from matplotlib.pyplot import *

"""
Material Roughness Scan
=======================

Demonstration of the roughen material modifier. The back row is aluminium and
the front row is glass. The surface roughness is increased from left to right
from 0 to 50%.
"""

world = World()

angle_increments = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
roughness_scan = [0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5]


# glass spheres
angle = 6
distance = 3
radius = 0.15

for i in range(9):
    roughness = roughness_scan[i]
    increment = angle_increments[i]

    # use raw material if roughness = 0
    if roughness == 0:
        material = schott("N-BK7")
    else:
        material = Roughen(schott("N-BK7"), roughness)

    Sphere(radius, world,
           transform=rotate(increment * angle, 0, 0) * translate(0, radius + 0.00001, distance),
           material=material)

# metal spheres
angle = 6
distance = 3.6
radius = 0.15

for i in range(9):
    roughness = roughness_scan[i]
    increment = angle_increments[i]

    # use raw material if roughness = 0
    if roughness == 0:
        material = Aluminium()
    else:
        material = RoughAluminium(roughness)

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
camera = PinholeCamera(parent=world, transform=translate(0, 3.3, 0) * rotate(0, -47, 0), fov=42)
camera.ray_min_depth = 2
camera.ray_max_depth = 5
camera.ray_extinction_prob = 0.01
camera.spectral_rays = 1
camera.spectral_samples = 15
camera.pixels = (1024, 512)
camera.pixel_samples = 250
camera.display_progress = True
camera.display_update_time = 10
camera.accumulate = True
camera.exposure_handler = AutoExposure()
# camera.ray_importance_sampling = False


# start ray tracing
ion()
for p in range(1, 1000):
    print("Rendering pass {} ({} samples/pixel)..."
          "".format(p, camera.accumulated_samples + camera.pixel_samples * camera.spectral_rays))
    camera.observe()
    camera.save("demo_roughen_{}_samples.png".format(camera.accumulated_samples))
    print()

# display final result
ioff()
camera.display()
show()
