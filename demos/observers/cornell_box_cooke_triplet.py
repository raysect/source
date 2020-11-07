from matplotlib.pyplot import *
from raysect.primitive import Sphere, Box, Cylinder, Subtract
from raysect.primitive.lens import Meniscus, BiConcave, BiConvex
from raysect.optical import World, Node, translate, rotate, Point3D
from raysect.optical.material import Lambert, UniformSurfaceEmitter, AbsorbingSurface, Checkerboard
from raysect.optical.library import *
from raysect.optical.observer import TargettedCCDArray
from raysect.optical.observer import RGBPipeline2D, BayerPipeline2D, PowerPipeline2D
from raysect.optical.observer import RGBAdaptiveSampler2D
from raysect.core.math import mm

"""
Cornell Box With Cooke Triplet Camera Lens System and CCD
=========================================================

This demo renders a variant of the classic Cornell Box scene.

To the original Cornell Box see:

    http://www.graphics.cornell.edu/online/box/data.html

The wall colours and light spectrum used in this demo are the values measured
for the physical Cornell Box.

None of the lenses have anti-reflection coatings.
"""

# THE SCENE ------------------------------------------------------------------------------------------------------------

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
            material=UniformSurfaceEmitter(light_spectrum, 2))

# alternate light #1
# light = Box(Point3D(-0.4, -0.4, -0.01), Point3D(0.4, 0.4, 0.0),
#             parent=enclosure,
#             transform=translate(0, 1, 0) * rotate(0, 90, 0),
#             material=UniformSurfaceEmitter(d65_white, 2))

# alternate light #2
# back_light = Sphere(0.1,
#     parent=enclosure,
#     transform=translate(0.80, -0.85, 0.80)*rotate(0, 0, 0),
#     material=UniformSurfaceEmitter(light_spectrum, 10.0))

# objects in enclosure
box = Box(Point3D(-0.4, 0, -0.4), Point3D(0.3, 1.4, 0.3),
          parent=world,
          transform=translate(0.4, -1 + 1e-6, 0.4)*rotate(30, 0, 0),
          material=schott("N-BK7"))

sphere = Sphere(0.4,
                parent=world,
                transform=translate(-0.4, -0.6 + 1e-6, -0.4)*rotate(0, 0, 0),
                material=schott("N-BK7"))

# THE CAMERA -----------------------------------------------------------------------------------------------------------

# The camera assembly uses a Cooke triplet lens system. The lens system based on design in:
# https://wp.optics.arizona.edu/jsasian/wp-content/uploads/sites/33/2016/03/L20_OPTI517_Cooke_triplet.pdf

camera = Node(parent=world, transform=translate(0, 0, -3.8)*rotate(0, 0, 0))

# uncomment to show a 0.5m checkerboard ruler
#ruler = Box(Point3D(-0.1, -0.005, 0.0), Point3D(0.1, 0.005, 5.0), parent=camera, transform=translate(0, -0.1, 0.0), material=Checkerboard(width=0.5, scale1=0, scale2=1.0))

# Cooke triplet system
lenses = Node(parent=camera)
l1 = Meniscus(mm(21), mm(4.831), mm(23.713), mm(7331.288), parent=lenses, transform=translate(0, 0, mm(-4.831)), material=schott("N-LAK9"))
l2 = BiConcave(mm(13), mm(0.975), mm(24.456), mm(21.896), parent=l1, transform=translate(0, 0, mm(-6.835)), material=schott("SF5"))
l3 = BiConvex(mm(18), mm(3.127), mm(86.759), mm(20.4942), parent=l2, transform=translate(0, 0, mm(-7.949)), material=schott("N-LAK9"))
image_plane = Node(parent=l3, transform=translate(0, 0, mm(-41.5)))  # tweaked position gives a sharper image (original: 41.10346 mm)

# disable importance sampling of the lenses (enabled by default for dielectrics and emitters)
l1.material.importance = 0.0
l2.material.importance = 0.0
l3.material.importance = 0.0

# cylinder body holding the lenses and CCD
body = Subtract(
    Cylinder(mm(26), mm(80.0), transform=translate(0, 0, mm(-63))),
    Cylinder(mm(25), mm(79.1), transform=translate(0, 0, mm(-62))),
    parent=camera, transform=translate(0, 0, 0), material=AbsorbingSurface()
)

# L1 lens mount
l1_mount = Subtract(
    Cylinder(mm(25.5), mm(5.0), transform=translate(0, 0, mm(0))),
    Cylinder(mm(21 / 2 + 0.01), mm(5.1), transform=translate(0, 0, mm(-0.05))),
    parent=l1, transform=translate(0, 0, mm(0)), material=AbsorbingSurface()
)

# L2 lens mount
l2_mount = Subtract(
    Cylinder(mm(25.5), mm(4.0), transform=translate(0, 0, mm(0))),
    Cylinder(mm(13 / 2 + 0.01), mm(4.1), transform=translate(0, 0, mm(-0.05))),
    parent=l2, transform=translate(0, 0, mm(0)), material=AbsorbingSurface()
)

# L2-L3 stop
stop = Subtract(
    Cylinder(mm(25.5), mm(1.0), transform=translate(0, 0, mm(0))),
    Cylinder(mm(12 / 2 + 0.01), mm(1.1), transform=translate(0, 0, mm(-0.05))),
    parent=l2, transform=translate(0, 0, mm(-2)), material=AbsorbingSurface()
)

# L3 lens mount
l3_mount = Subtract(
    Cylinder(mm(25.5), mm(4.0), transform=translate(0, 0, mm(0))),
    Cylinder(mm(18 / 2 + 0.01), mm(4.1), transform=translate(0, 0, mm(-0.05))),
    parent=l3, transform=translate(0, 0, mm(0)), material=AbsorbingSurface()
)

# CCD pipeline and sampler
rgb = RGBPipeline2D(display_unsaturated_fraction=0.96, name="sRGB")
sampler = RGBAdaptiveSampler2D(rgb, ratio=10, fraction=0.2, min_samples=1000, cutoff=0.01)

# CCD targetting all rays at last lens element for speed
ccd = TargettedCCDArray(
    targetted_path_prob=1.0, targets=[l3],
    width=mm(35), pixels=(512, 512),
    parent=image_plane, transform=translate(0, 0, 0)*rotate(0, 0, 180),
    pipelines=[rgb]
)
ccd.frame_sampler = sampler
ccd.spectral_rays = 1
ccd.spectral_bins = 15
ccd.pixel_samples = 250
ccd.ray_importance_sampling = True
ccd.ray_important_path_weight = 0.25
ccd.ray_max_depth = 500
ccd.ray_extinction_min_depth = 3
ccd.ray_extinction_prob = 0.01

# start ray tracing
ion()
name = 'cornell_box_cooke_triplet'
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
render_pass = 1
while not ccd.render_complete:

    print("Rendering pass {}...".format(render_pass))
    ccd.observe()
    rgb.save("{}_{}_pass_{}.png".format(name, timestamp, render_pass))
    print()

    render_pass += 1

ioff()
rgb.display()

