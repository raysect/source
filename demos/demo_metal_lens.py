from raysect.optical import World, translate, rotate, Point3D, d65_white, ConstantSF
from raysect.primitive import Sphere, Box, Cylinder, Subtract
from raysect.primitive.lens.spherical import BiConvex
from raysect.optical.observer import CCD
from raysect.optical.material import Gold, Silver, Copper, Titanium, Aluminium, Beryllium, Lambert, UniformSurfaceEmitter, AbsorbingSurface
from raysect.optical.library import schott
from raysect.core import print_scenegraph
from matplotlib.pyplot import *

world = World()

Sphere(0.5, world, transform=translate(1.2, 0.5001, 0.6), material=Gold())
Sphere(0.5, world, transform=translate(0.6, 0.5001, -0.6), material=Silver())
Sphere(0.5, world, transform=translate(0, 0.5001, 0.6), material=Copper())
Sphere(0.5, world, transform=translate(-0.6, 0.5001, -0.6), material=Titanium())
Sphere(0.5, world, transform=translate(-1.2, 0.5001, 0.6), material=Aluminium())
Sphere(0.5, world, transform=translate(0, 0.5001, -1.8), material=Beryllium())

Box(Point3D(-100, -0.1, -100), Point3D(100, 0, 100), world, material=Lambert(ConstantSF(1.0)))
Cylinder(3.0, 8.0, world, transform=translate(4, 8, 0) * rotate(90, 0, 0), material=UniformSurfaceEmitter(d65_white, 1.0))

camera = CCD(parent=world, transform=translate(0, 4, -3.5) * rotate(0, -48, 180))
camera.sensitivity = 8.0
camera.ray_min_depth = 3
camera.ray_max_depth = 500
camera.ray_extinction_prob = 0.01
camera.spectral_rays = 1
camera.spectral_samples = 20
camera.pixels = (360, 240) #(720, 480)
camera.pixel_samples = 200
camera.display_progress = True
camera.display_update_time = 10
camera.accumulate = True
# camera.process_count = 1

# b = BiConvex(0.0508, 0.0036, 1.0295, 1.0295, parent=camera, transform=translate(0, 0, 0.1), material=schott("N-BK7"))
# b = BiConvex(0.0508, 0.0062, 0.205, 0.205, parent=camera, transform=translate(0, 0, 0.05), material=schott("N-BK7"))
b = BiConvex(0.0508, 0.0144, 0.0593, 0.0593, parent=camera, transform=translate(0, 0, 0.0536), material=schott("N-BK7"))

c = Subtract(
        Subtract(
            Cylinder(0.0260, 0.07, transform=translate(0, 0, 0)),
            Cylinder(0.0255, 0.06, transform=translate(0, 0, 0.005))
        ),
    Cylinder(0.015, 0.007, transform=translate(0, 0, 0.064)),
    parent=camera,
    transform=translate(0, 0, -0.01),
    material=AbsorbingSurface()
)


print_scenegraph(b)

# start ray tracing
ion()
for p in range(1, 100):
    print("Rendering pass {} ({} samples/pixel)...".format(p, camera.accumulated_samples + camera.pixel_samples * camera.spectral_rays))
    camera.observe()
    camera.save("demo_metal_lens_test_{}_samples.png".format(camera.accumulated_samples))
    print()

# display final result
ioff()
camera.display()
show()
