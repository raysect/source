from raysect.optical import World, Node, Point3D, translate, rotate, d65_white, ConstantSF
from raysect.optical.observer import PinholeCamera
from raysect.optical.material import Lambert, NullVolume
from raysect.optical.material.emitter import UniformSurfaceEmitter
from raysect.optical.library import schott
from raysect.primitive import Intersect, Subtract, Box, Cylinder, Sphere
from math import tan, pi
from matplotlib.pyplot import *

class DirectionalEmitterSurface(NullVolume):

    def __init__(self):

        super().__init__()
        self.importance = 1

    def evaluate_surface(self, world, ray, primitive, hit_point, exiting, inside_point, outside_point, normal, world_to_primitive, primitive_to_world):

        spectrum = ray.new_spectrum()
        if -normal.dot(ray.direction.transform(world_to_primitive)) > 0.99:
            spectrum.samples = 1000 * d65_white.sample_multiple(spectrum.min_wavelength, spectrum.max_wavelength, spectrum.num_samples)
        return spectrum


def equilateral_prism(parent=None, transform=None, material=None):

    mid_point = 0.05 * tan(60/180*pi) / 2
    centre = Box(Point3D(-0.05, 0, 0), Point3D(0.05, 0.1, 0.1))
    left = Box(Point3D(0, -0.001, -0.001), Point3D(0.1, 0.101, 0.15), transform=translate(0.05, 0, 0) * rotate(30, 0, 0))
    right = Box(Point3D(-0.1, -0.001, -0.001), Point3D(0.0, 0.101, 0.15), transform=translate(-0.05, 0, 0) * rotate(-30, 0, 0))
    return Subtract(Subtract(centre, left), right, parent=parent, transform=transform * translate(0, 0, -mid_point), material=material)


def light_box(parent, transform=None):

    node = Node(parent=parent, transform=transform)
    outer = Box(Point3D(-0.01, 0, -0.05), Point3D(0.01, 0.15, 0.0))
    slit = Box(Point3D(-0.005, 0.01, -0.045), Point3D(0.005, 0.014, 0.0001))
    Subtract(outer, slit, parent=node, material=Lambert(reflectivity=ConstantSF(0.1)))
    Box(Point3D(-0.005, 0.01, -0.045), Point3D(0.005, 0.014, -0.04), parent=node, material=UniformSurfaceEmitter(d65_white, 1000))
    return node


world = World()

floor = Box(Point3D(-1000, -0.1, -1000), Point3D(1000, 0, 1000), parent=world, material=Lambert())

prism_mount = Cylinder(0.025, 0.01, parent=world, material=Lambert(), transform=rotate(0, 90, 0))
prism = equilateral_prism(parent=world, material=schott("SF11"), transform=translate(0, 0.01 + 1e-6, 0))

screen = Intersect(
    Box(Point3D(-10, -10, -10), Point3D(10, 10, 0)),
    Subtract(Cylinder(0.22, 0.15), Cylinder(0.20, 0.16, transform=translate(0, 0, -0.005)), transform=rotate(0, 90, 0)),
    parent=world,
    material=Lambert(ConstantSF(1.0))
)


lightbox = light_box(parent=world, transform=translate(0.025, 0, 0) * rotate(-35.5, 0, 0) *translate(0.10, 0, 0) * rotate(90, 0, 0))

light = Sphere(0.5, parent=world, material=UniformSurfaceEmitter(d65_white, scale=2), transform=translate(0, 2, -1))

prism.material.importance = 5

camera = PinholeCamera(parent=world, transform=translate(0, 0.05, -0.05) * rotate(180, -65, 0) * translate(0, 0, -0.75), fov=45)
camera.ray_importance_sampling = True
camera.ray_important_path_weight = 0.5
camera.ray_min_depth = 3
camera.ray_max_depth = 500
camera.ray_extinction_prob = 0.01
camera.spectral_rays = 1
camera.spectral_samples = 15
camera.pixels = (128, 64)
camera.pixel_samples = 50
camera.display_progress = True
camera.display_update_time = 10
camera.accumulate = True

# start ray tracing
ion()
for p in range(0, 1000):
    print("Rendering pass {} ({} samples/pixel)...".format(p+1, camera.accumulated_samples + camera.pixel_samples * camera.spectral_rays))
    camera.observe()
    camera.save("prisms_{}_samples.png".format(camera.accumulated_samples))
    print()


# display final result
ioff()
camera.display()
show()



