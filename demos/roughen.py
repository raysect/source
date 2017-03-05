from raysect.optical import World, translate, rotate, Point3D, d65_white
from raysect.primitive import Sphere, Box, Cylinder
from raysect.optical.observer import PinholeCamera, RGBPipeline2D
from raysect.optical.material import Lambert, UniformSurfaceEmitter, Roughen
from raysect.optical.library import *
from matplotlib.pyplot import *

rough_max = 0.5

world = World()

# glass spheres
angle = 6
distance = 3
radius = 0.15

Sphere(radius, world, transform=rotate(-4 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=Aluminium())
Sphere(radius, world, transform=rotate(-3 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=RoughAluminium(0.125 * rough_max))
Sphere(radius, world, transform=rotate(-2 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=RoughAluminium(0.25 * rough_max))
Sphere(radius, world, transform=rotate(-1 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=RoughAluminium(0.375 * rough_max))
Sphere(radius, world, transform=rotate(0 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=RoughAluminium(0.5 * rough_max))
Sphere(radius, world, transform=rotate(1 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=RoughAluminium(0.625 * rough_max))
Sphere(radius, world, transform=rotate(2 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=RoughAluminium(0.750 * rough_max))
Sphere(radius, world, transform=rotate(3 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=RoughAluminium(0.875 * rough_max))
Sphere(radius, world, transform=rotate(4 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=RoughAluminium(1.0 * rough_max))

# metal spheres
angle = 6
distance = 3.6
radius = 0.15

Sphere(radius, world, transform=rotate(-4 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=Copper())
Sphere(radius, world, transform=rotate(-3 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=RoughCopper(0.125 * rough_max))
Sphere(radius, world, transform=rotate(-2 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=RoughCopper(0.25 * rough_max))
Sphere(radius, world, transform=rotate(-1 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=RoughCopper(0.375 * rough_max))
Sphere(radius, world, transform=rotate(0 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=RoughCopper(0.5 * rough_max))
Sphere(radius, world, transform=rotate(1 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=RoughCopper(0.625 * rough_max))
Sphere(radius, world, transform=rotate(2 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=RoughCopper(0.75 * rough_max))
Sphere(radius, world, transform=rotate(3 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=RoughCopper(0.875 * rough_max))
Sphere(radius, world, transform=rotate(4 * angle, 0, 0) * translate(0, radius + 0.00001, distance), material=RoughCopper(1.0 * rough_max))

# rough titanium ground plane
# Box(Point3D(-100, -0.1, -100), Point3D(100, 0, 100), world, material=Roughen(Titanium(), 0.1))
Box(Point3D(-100, -0.1, -100), Point3D(100, 0, 100), world, material=RoughTungsten(0.2))

# four strip lights
Cylinder(0.5, 1.0, world, transform=translate(0.5, 5, 8) * rotate(90, 0, 0), material=UniformSurfaceEmitter(d65_white, 1.0))
Cylinder(0.5, 1.0, world, transform=translate(0.5, 5, 6) * rotate(90, 0, 0), material=UniformSurfaceEmitter(d65_white, 1.0))
Cylinder(0.5, 1.0, world, transform=translate(0.5, 5, 4) * rotate(90, 0, 0), material=UniformSurfaceEmitter(d65_white, 1.0))
Cylinder(0.5, 1.0, world, transform=translate(0.5, 5, 2) * rotate(90, 0, 0), material=UniformSurfaceEmitter(d65_white, 1.0))

rgb = RGBPipeline2D(display_unsaturated_fraction=0.96, name="sRGB")
# sampler = RGBAdaptiveSampler2D(rgb, ratio=10, fraction=0.2, min_samples=500, cutoff=0.05)

# observer
camera = PinholeCamera((1024, 512), fov=42, pipelines=[rgb], transform=translate(0, 3.3, 0) * rotate(0, -47, 0), parent=world)
camera.ray_max_depth = 500
camera.ray_extinction_prob = 0.01
camera.spectral_rays = 1
camera.spectral_bins = 15
camera.pixel_samples = 50


# start ray tracing
ion()
for p in range(1, 1000):
    print("Rendering pass {}".format(p))
    camera.observe()
    camera.pipelines[0].save("demo_roughen_{}_samples.png".format(p))
    print()

# display final result
ioff()
camera.pipelines[0].display()
show()
