
# External module imports
from math import tan, pi
import matplotlib.pyplot as plt

# Raysect imports
from raysect.optical import World, Node, Point3D, translate, rotate, d65_white, ConstantSF
from raysect.optical.observer import PinholeCamera, RGBPipeline2D
from raysect.optical.material import Lambert
from raysect.optical.material.emitter import UniformSurfaceEmitter
from raysect.optical.library import schott
from raysect.primitive import Intersect, Subtract, Box, Cylinder, Sphere


# Utility method to construct a glass prism from CSG operations
def equilateral_prism(width=1.0, height=1.0, parent=None, transform=None, material=None):

    half_width = width / 2
    mid_point = half_width * tan(60/180*pi) / 2

    centre = Box(Point3D(-half_width * 1.001, 0, 0), Point3D(half_width * 1.001, height, width))

    left = Box(Point3D(0, -height * 0.001, -width * 0.001),
               Point3D(width, height * 1.001, 2 * width),
               transform=translate(half_width, 0, 0) * rotate(30, 0, 0))

    right = Box(Point3D(-width, -height * 0.001, -width * 0.001),
                Point3D(0.0, height * 1.001, 2 * width),
                transform=translate(-half_width, 0, 0) * rotate(-30, 0, 0))

    csg_prism = Subtract(Subtract(centre, left), right,
                     parent=parent, transform=transform * translate(0, 0, -mid_point),
                     material=material)

    return csg_prism


# Utility method to construct a box with a slit emitting white light
def light_box(parent, transform=None):

    # Notice that this function is creating and returning a parent node which holds references
    # to the underlying primitives.
    node = Node(parent=parent, transform=transform)

    outer = Box(Point3D(-0.01, 0, -0.05), Point3D(0.01, 0.15, 0.0))
    slit = Box(Point3D(-0.0015, 0.03, -0.045), Point3D(0.0015, 0.12, 0.0001))

    Subtract(outer, slit, parent=node, material=Lambert(reflectivity=ConstantSF(0.1)))

    Box(Point3D(-0.0015, 0.03, -0.045),
        Point3D(0.0015, 0.12, -0.04),
        parent=node,
        material=UniformSurfaceEmitter(d65_white, 250))

    return node


world = World()


# construct diffuse floor surface
floor = Box(Point3D(-1000, -0.1, -1000), Point3D(1000, 0, 1000),
            parent=world, material=Lambert())


# construct prism from utility method
prism = equilateral_prism(0.06, 0.15, parent=world,
                          material=schott("SF11"), transform=translate(0, 0.0 + 1e-6, 0))


# Curved target screen for collecting rainbow light
screen = Intersect(
    Box(Point3D(-10, -10, -10), Point3D(10, 10, 0)),
    Subtract(Cylinder(0.22, 0.15),
             Cylinder(0.20, 0.16, transform=translate(0, 0, -0.005)),
             transform=rotate(0, 90, 0)),
    parent=world,
    material=Lambert()
)


# construct main collimated light source
prism_light = light_box(parent=world,
                        transform=rotate(-35.5, 0, 0) * translate(0.10, 0, 0) * rotate(90, 0, 0))


# background light source
top_light = Sphere(0.5, parent=world, transform=translate(0, 2, -1),
                   material=UniformSurfaceEmitter(d65_white, scale=2))


# Give the prism a high importance to ensure adequate sampling
prism.material.importance = 9

rgb = RGBPipeline2D()

# create and setup the camera
camera = PinholeCamera((512, 256), fov=45, parent=world, pipelines=[rgb])
camera.transform = translate(0, 0.05, -0.05) * rotate(180, -65, 0) * translate(0, 0, -0.75)
camera.ray_importance_sampling = True
camera.ray_important_path_weight = 0.75
camera.ray_max_depth = 500
camera.ray_extinction_prob = 0.01
camera.spectral_bins = 32
camera.spectral_rays = 32
camera.pixel_samples = 100

# start ray tracing
plt.ion()
for p in range(0, 1000):
    print("Rendering pass {}".format(p+1))
    camera.observe()
    rgb.save("prisms_{}.png".format(p+1))
    print()

# display final result
plt.ioff()
rgb.display()
plt.show()
