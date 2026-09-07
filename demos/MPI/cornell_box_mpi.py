"""
Cornell Box MPI Demo
====================

This demo renders a variant of the classic Cornell Box scene.

For the original Cornell Box see:

    http://www.graphics.cornell.edu/online/box/data.html

The wall colours and light spectrum used in this demo are the values measured
for the physical Cornell Box.

The MPI (message passing interface) render engine is used for this demo.
To run, use `mpirun -n <nprocesses> python cornell_box_mpi.py`. An
MPI library must be installed on your system, and the mpi4py Python
package must be installed in the Python environment.

There are some small differences between this demo and the shared-memory
equilvalent cornell_box.py, due to details of the MPI implementation meaning
we need to communicate the results of one render pass to all the worker
processes before the next pass. Compare the last section of the two demo
scripts to see how this is done.
"""

from numpy import array

from raysect.primitive import Sphere, Box
from raysect.optical import World, Node, translate, rotate, Point3D
from raysect.optical.material import Lambert, UniformSurfaceEmitter
from raysect.optical.library import InterpolatedSF, schott
from raysect.optical.observer import PinholeCamera
from raysect.optical.observer import RGBPipeline2D, BayerPipeline2D, PowerPipeline2D
from raysect.optical.observer import RGBAdaptiveSampler2D
from raysect.core.workflow import MPIEngine


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


filter_red = InterpolatedSF([100, 650, 660, 670, 680, 800], [0, 0, 1, 1, 0, 0])
filter_green = InterpolatedSF([100, 530, 540, 550, 560, 800], [0, 0, 1, 1, 0, 0])
filter_blue = InterpolatedSF([100, 480, 490, 500, 510, 800], [0, 0, 1, 1, 0, 0])

# create and setup the camera
power_unfiltered = PowerPipeline2D(display_unsaturated_fraction=0.96, name="Unfiltered")
power_unfiltered.display_update_time = 15

power_green = PowerPipeline2D(filter=filter_green, display_unsaturated_fraction=0.96, name="Green Filter")
power_green.display_update_time = 15

power_red = PowerPipeline2D(filter=filter_red, display_unsaturated_fraction=0.96, name="Red Filter")
power_red.display_update_time = 15

rgb = RGBPipeline2D(display_unsaturated_fraction=0.96, name="sRGB")

bayer = BayerPipeline2D(filter_red, filter_green, filter_blue, display_unsaturated_fraction=0.96, name="Bayer Filter")
bayer.display_update_time = 15

pipelines = [rgb, power_unfiltered, power_green, power_red, bayer]

sampler = RGBAdaptiveSampler2D(rgb, ratio=10, fraction=0.2, min_samples=500, cutoff=0.01)


camera = PinholeCamera((1024, 1024), parent=world, transform=translate(0, 0, -3.3) * rotate(0, 0, 0), pipelines=pipelines)
camera.frame_sampler = sampler
camera.spectral_rays = 1
camera.spectral_bins = 15
camera.pixel_samples = 250
camera.ray_importance_sampling = True
camera.ray_important_path_weight = 0.25
camera.ray_max_depth = 500
camera.ray_extinction_min_depth = 3
camera.ray_extinction_prob = 0.01

# Speedups for testing purposes.
# Fewer camera pixels for faster runtime per pass.
camera.pixels = (256, 256)
# Less strict cutoff to finish with fewer passes.
sampler.cutoff = 0.05


camera.render_engine = MPIEngine()

# Don't make plots during the render as MPI is typically run non-interactively.
for pipeline in camera.pipelines:
    pipeline.display_progress = False

# Worker processes do not have accurate statistics, only the root process has
# all the render results. So don't bother outputting statistics on the workers.
if camera.render_engine.rank != 0:
    camera.quiet = True

# start ray tracing.
print(f"Starting ray tracing on rank {camera.render_engine.rank}")
render_pass = 1
while not camera.render_complete:
    if camera.render_engine.rank == 0:
        print(f"Rendering pass {render_pass}...")
    camera.observe()
    # Rank 0 processes the ray tracing results so is the only process which
    # knows the true progress of the render. In order to know when the render
    # is complete, we need to ensure the frame sampler on each of the worker
    # processes has an up-to-date copy of the pipeline used for sampling.
    # We also need to use a different variable name for the received broadcast
    # object else it is not broadcast properly.
    root_rgb = camera.render_engine.comm.bcast(rgb, root=0)
    camera.frame_sampler.pipeline = root_rgb
    camera.render_engine.comm.Barrier()
    render_pass += 1
print(f"Finished ray tracing on rank {camera.render_engine.rank}")

# Again, only rank 0 has all the ray tracing results so is the only one which
# can produce a correct image.
if camera.render_engine.rank == 0:
    rgb.save('CornellMPI_rgb.png')
