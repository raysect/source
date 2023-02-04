import sys
from numpy import array
from matplotlib.pyplot import *

from raysect.primitive import Sphere, Box, Cylinder, Subtract
from raysect.optical import World, Node, translate, rotate, Point3D, d65_white, InterpolatedSF, Vector3D, ConstantSF
from raysect.optical.material import Lambert, UniformSurfaceEmitter, UniformVolumeEmitter, UniformAttenuator, UniformDiattenuator, UnitySurfaceEmitter
from raysect.optical.library import *
from raysect.optical.observer import PinholeCamera
from raysect.optical.observer import RGBPipeline2D
from raysect.optical.observer import RGBAdaptiveSampler2D

start = int(sys.argv[1])
end = int(sys.argv[2])
step = int(sys.argv[3])

"""
Cornell Box Demo
================

This demo renders a variant of the classic Cornell Box scene.

To the original Cornell Box see:

    http://www.graphics.cornell.edu/online/box/data.html

The wall colours and light spectrum used in this demo are the values measured
for the physical Cornell Box.
"""


def polariser(parent=None, transform=None):

    assembly = Node(parent=parent, transform=transform)

    polariser = Cylinder(
        radius=0.101, height=0.002, parent=assembly, transform=translate(0, 0, -0.001),
        material=UniformDiattenuator(Vector3D(0, 1, 0), ConstantSF(1), ConstantSF(0))
    )

    body = Subtract(
        Cylinder(radius=0.11, height=0.010),
        Cylinder(radius=0.1, height=0.011, transform=translate(0, 0, -0.0005)),
        parent=assembly,
        transform=translate(0, 0, -0.005),
        material=Lambert(ConstantSF(0.05))
    )

    handle = Cylinder(
        radius=0.004, height=0.021, parent=assembly,
        transform=translate(0, 0.109, 0)*rotate(0, 90, 0),
        material=Lambert(ConstantSF(0.05))
    )

    return assembly

# define reflectivity for box surfaces
wavelengths = array(
    [400, 404, 408, 412, 416, 420, 424, 428, 432, 436, 440, 444, 448, 452, 456, 460, 464, 468, 472, 476, 480, 484, 488,
     492, 496, 500, 504, 508, 512, 516, 520, 524, 528, 532, 536, 540, 544, 548, 552, 556, 560, 564, 568, 572, 576, 580,
     584, 588, 592, 596, 600, 604, 608, 612, 616, 620, 624, 628, 632, 636, 640, 644, 648, 652, 656, 660, 664, 668, 672,
     676, 680, 684, 688, 692, 696, 700])

white = array(
    [0.343, 0.445, 0.551, 0.624, 0.665, 0.687, 0.708, 0.723, 0.715, 0.71, 0.745, 0.758, 0.739, 0.767, 0.777, 0.765,
     0.751, 0.745, 0.748, 0.729, 0.745, 0.757, 0.753, 0.75, 0.746, 0.747, 0.735, 0.732, 0.739, 0.734, 0.725, 0.721,
     0.733, 0.725, 0.732, 0.743, 0.744, 0.748, 0.728, 0.716, 0.733, 0.726, 0.713, 0.74, 0.754, 0.764, 0.752, 0.736,
     0.734, 0.741, 0.74, 0.732, 0.745, 0.755, 0.751, 0.744, 0.731, 0.733, 0.744, 0.731, 0.712, 0.708, 0.729, 0.73,
     0.727, 0.707, 0.703, 0.729, 0.75, 0.76, 0.751, 0.739, 0.724, 0.73, 0.74, 0.737])

green = array(
    [0.092, 0.096, 0.098, 0.097, 0.098, 0.095, 0.095, 0.097, 0.095, 0.094, 0.097, 0.098, 0.096, 0.101, 0.103, 0.104,
     0.107, 0.109, 0.112, 0.115, 0.125, 0.14, 0.16, 0.187, 0.229, 0.285, 0.343, 0.39, 0.435, 0.464, 0.472, 0.476, 0.481,
     0.462, 0.447, 0.441, 0.426, 0.406, 0.373, 0.347, 0.337, 0.314, 0.285, 0.277, 0.266, 0.25, 0.23, 0.207, 0.186,
     0.171, 0.16, 0.148, 0.141, 0.136, 0.13, 0.126, 0.123, 0.121, 0.122, 0.119, 0.114, 0.115, 0.117, 0.117, 0.118, 0.12,
     0.122, 0.128, 0.132, 0.139, 0.144, 0.146, 0.15, 0.152, 0.157, 0.159])

red = array(
    [0.04, 0.046, 0.048, 0.053, 0.049, 0.05, 0.053, 0.055, 0.057, 0.056, 0.059, 0.057, 0.061, 0.061, 0.06, 0.062, 0.062,
     0.062, 0.061, 0.062, 0.06, 0.059, 0.057, 0.058, 0.058, 0.058, 0.056, 0.055, 0.056, 0.059, 0.057, 0.055, 0.059,
     0.059, 0.058, 0.059, 0.061, 0.061, 0.063, 0.063, 0.067, 0.068, 0.072, 0.08, 0.09, 0.099, 0.124, 0.154, 0.192,
     0.255, 0.287, 0.349, 0.402, 0.443, 0.487, 0.513, 0.558, 0.584, 0.62, 0.606, 0.609, 0.651, 0.612, 0.61, 0.65, 0.638,
     0.627, 0.62, 0.63, 0.628, 0.642, 0.639, 0.657, 0.639, 0.635, 0.642])

white_reflectivity = InterpolatedSF(wavelengths, white)
red_reflectivity = InterpolatedSF(wavelengths, red)
green_reflectivity = InterpolatedSF(wavelengths, green)

# define light spectrum
light_spectrum = InterpolatedSF(array([400, 500, 600, 700]), array([0.0, 8.0, 15.6, 18.4]))

# set-up scenegraph
world = World()

# enclosing box
enclosure = Node(world)

e_back = Box(Point3D(-1, -1, 0), Point3D(1, 1, 0),
             parent=enclosure,
             transform=translate(0, 0, 1) * rotate(0, 0, 0),
             material=Lambert(white_reflectivity))

e_bottom = Box(Point3D(-1, -1, 0), Point3D(1, 1, 0),
               parent=enclosure,
               transform=translate(0, -1, 0) * rotate(0, -90, 0),
               # material=m)
               material=Lambert(white_reflectivity))

e_top = Box(Point3D(-1, -1, 0), Point3D(1, 1, 0),
            parent=enclosure,
            transform=translate(0, 1, 0) * rotate(0, 90, 0),
            material=Lambert(white_reflectivity))

e_left = Box(Point3D(-1, -1, 0), Point3D(1, 1, 0),
             parent=enclosure,
             transform=translate(1, 0, 0) * rotate(-90, 0, 0),
             material=Lambert(red_reflectivity))

e_right = Box(Point3D(-1, -1, 0), Point3D(1, 1, 0),
              parent=enclosure,
              transform=translate(-1, 0, 0) * rotate(90, 0, 0),
              material=Lambert(green_reflectivity))

# ceiling light
light = Box(Point3D(-0.4, -0.4, -0.01), Point3D(0.4, 0.4, 0.0),
            parent=enclosure,
            transform=translate(0, 1, 0) * rotate(0, 90, 0),
            # material=UniformSurfaceEmitter(d65_white, 2))
            material=UniformSurfaceEmitter(light_spectrum, 2))

# objects in enclosure
box = Box(
    Point3D(-0.4, 0, -0.4), Point3D(0.3, 1.4, 0.3),
    parent=world,
    transform=translate(0.4, -1 + 1e-6, 0.4)*rotate(30, 0, 0),
    material=schott("N-BK7")
)

sphere = Sphere(
    0.4,
    parent=world,
    transform=translate(-0.4, -0.6 + 1e-6, -0.4)*rotate(0, 0, 0),
    material=schott("N-BK7")
)

# polarisers
p1 = polariser(parent=world, transform=translate(0, 0, -2.96)*rotate(0, 0, 0))
# p2 = polariser(parent=world, transform=translate(0, 0, -2.98)*rotate(0, 0, 45))
# p3 = polariser(parent=world, transform=translate(0, 0, -3)*rotate(0, 0, 90))



# from raysect.core.workflow import SerialEngine
# camera.render_engine = SerialEngine()

# start ray tracing
ion()

for i in range(start, end, step):

    p1.transform = translate(0, 0, -2.96)*rotate(0, 0, i)

    rgb = RGBPipeline2D(display_unsaturated_fraction=0.96, name="sRGB")
    rgb.display_progress = False

    sampler = RGBAdaptiveSampler2D(rgb, ratio=100, fraction=1.0, min_samples=1000, max_samples=10000, cutoff=0.1)

    camera = PinholeCamera((512, 512), parent=world, transform=translate(0, 0, -3.3) * rotate(0, 0, 00), pipelines=[rgb])
    camera.frame_sampler = sampler
    camera.spectral_rays = 1
    camera.spectral_bins = 15
    camera.pixel_samples = 1000
    camera.ray_importance_sampling = True
    camera.ray_important_path_weight = 0.25
    camera.ray_max_depth = 500
    camera.ray_extinction_min_depth = 3
    camera.ray_extinction_prob = 0.01

    name = 'cornell_box_polariser_anim_test'
    # timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    render_pass = 1
    while not camera.render_complete:

        print("Rendering angle {}, pass {}...".format(i, render_pass))
        camera.observe()
        print()
        render_pass += 1

    rgb.save("{}_angle_{}.png".format(name, i))

    camera.parent = None
    del camera
    del rgb
    del sampler


ioff()
# rgb.display()

