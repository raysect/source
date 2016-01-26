from raysect.optical import World, translate, rotate, Point3D, d65_white
from raysect.primitive import Sphere, Box, Cylinder
from raysect.optical.observer import PinholeCamera
from raysect.optical.material import Roughen, Gold, Silver, Copper, Titanium, Aluminium, Lambert, UniformSurfaceEmitter
from raysect.optical.library import schott
from matplotlib.pyplot import *

rough_max = 0.5

world = World()

# glass spheres
angle = 6
distance = 3
radius = 0.15

Sphere(radius, world, transform=rotate(-4 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=schott("N-BK7"))
Sphere(radius, world, transform=rotate(-3 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=Roughen(schott("N-BK7"), 0.125 * rough_max))
Sphere(radius, world, transform=rotate(-2 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=Roughen(schott("N-BK7"), 0.25 * rough_max))
Sphere(radius, world, transform=rotate(-1 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=Roughen(schott("N-BK7"), 0.375 * rough_max))
Sphere(radius, world, transform=rotate(0 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=Roughen(schott("N-BK7"), 0.5 * rough_max))
Sphere(radius, world, transform=rotate(1 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=Roughen(schott("N-BK7"), 0.625 * rough_max))
Sphere(radius, world, transform=rotate(2 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=Roughen(schott("N-BK7"), 0.750 * rough_max))
Sphere(radius, world, transform=rotate(3 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=Roughen(schott("N-BK7"), 0.875 * rough_max))
Sphere(radius, world, transform=rotate(4 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=Roughen(schott("N-BK7"), 1.0 * rough_max))

# metal spheres
angle = 6
distance = 3.6
radius = 0.15

Sphere(radius, world, transform=rotate(-4 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=Aluminium())
Sphere(radius, world, transform=rotate(-3 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=Roughen(Aluminium(), 0.125 * rough_max))
Sphere(radius, world, transform=rotate(-2 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=Roughen(Aluminium(), 0.25 * rough_max))
Sphere(radius, world, transform=rotate(-1 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=Roughen(Aluminium(), 0.375 * rough_max))
Sphere(radius, world, transform=rotate(0 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=Roughen(Aluminium(), 0.5 * rough_max))
Sphere(radius, world, transform=rotate(1 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=Roughen(Aluminium(), 0.625 * rough_max))
Sphere(radius, world, transform=rotate(2 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=Roughen(Aluminium(), 0.75 * rough_max))
Sphere(radius, world, transform=rotate(3 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=Roughen(Aluminium(), 0.875 * rough_max))
Sphere(radius, world, transform=rotate(4 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=Roughen(Aluminium(), 1.0 * rough_max))

# rough titanium ground plane
Box(Point3D(-100, -0.1, -100), Point3D(100, 0, 100), world, material=Roughen(Titanium(), 0.1))

# four strip lights
Cylinder(0.5, 30.0, world, transform=translate(15, 5, 8) * rotate(90, 0, 0), material=UniformSurfaceEmitter(d65_white, 1.0))
Cylinder(0.5, 30.0, world, transform=translate(15, 5, 6) * rotate(90, 0, 0), material=UniformSurfaceEmitter(d65_white, 1.0))
Cylinder(0.5, 30.0, world, transform=translate(15, 5, 4) * rotate(90, 0, 0), material=UniformSurfaceEmitter(d65_white, 1.0))
Cylinder(0.5, 30.0, world, transform=translate(15, 5, 2) * rotate(90, 0, 0), material=UniformSurfaceEmitter(d65_white, 1.0))

# observer
camera = PinholeCamera(parent=world, transform=translate(0, 3.3, 0) * rotate(0, -47, 0), fov=42)
camera.ray_min_depth = 3
camera.ray_max_depth = 500
camera.ray_extinction_prob = 0.01
camera.rays = 1
camera.spectral_samples = 15
camera.pixels = (1024, 512)
camera.pixel_samples = 100
camera.display_progress = True
camera.display_update_time = 10
camera.accumulate = True

# start ray tracing
ion()
for p in range(1, 1000):
    print("Rendering pass {} ({} samples/pixel)...".format(p, camera.accumulated_samples + camera.pixel_samples * camera.rays))
    camera.observe()
    camera.save("demo_roughen_{}_samples.png".format(camera.accumulated_samples))
    print()

# display final result
ioff()
camera.display()
show()
