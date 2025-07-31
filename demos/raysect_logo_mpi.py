"""
Renders the raysect logo: a top-down view of 6 coloured slabs.

MPI (message passing interface) is used to parallelise the workflow.
Running this demo therefore requires an MPI library, such as MPICH,
OpenMPI or MicrosoftMPI. The mpi4py Python package must also
be installed.

To run this demo, call `mpirun -n <nprocesses> python raysect_logo_mpi.py`.
If a batch job scheduler is used on a cluster, the render may be spread
across multiple nodes of the cluster. If run interactively at the command
line, all rendering will be done on the same machine as the command line.
"""

from matplotlib.pyplot import *
from numpy import array

from raysect.primitive import Sphere, Box

from raysect.optical import World, Node, translate, rotate, Point3D, d65_white, ConstantSF, InterpolatedSF
from raysect.optical.observer import PinholeCamera
from raysect.optical.material.emitter import UniformSurfaceEmitter
from raysect.optical.material.dielectric import Dielectric
from raysect.core.workflow import MPIEngine


world = World()

wavelengths = array([300, 490, 510, 590, 610, 800])
red_attn = array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0]) * 0.98
green_attn = array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]) * 0.85
blue_attn = array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]) * 0.98
yellow_attn = array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]) * 0.85
cyan_attn = array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]) * 0.85
purple_attn = array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0]) * 0.95

red_glass = Dielectric(index=ConstantSF(1.4), transmission=InterpolatedSF(wavelengths, red_attn))
green_glass = Dielectric(index=ConstantSF(1.4), transmission=InterpolatedSF(wavelengths, green_attn))
blue_glass = Dielectric(index=ConstantSF(1.4), transmission=InterpolatedSF(wavelengths, blue_attn))
yellow_glass = Dielectric(index=ConstantSF(1.4), transmission=InterpolatedSF(wavelengths, yellow_attn))
cyan_glass = Dielectric(index=ConstantSF(1.4), transmission=InterpolatedSF(wavelengths, cyan_attn))
purple_glass = Dielectric(index=ConstantSF(1.4), transmission=InterpolatedSF(wavelengths, purple_attn))

Sphere(1000, world, material=UniformSurfaceEmitter(d65_white, 1.0))

node = Node(parent=world, transform=rotate(0, 0, 90))
Box(Point3D(-0.5, 0, -2.5), Point3D(0.5, 0.25, 0.5), node, rotate(0, 0, 0) * translate(0, 1, -0.500001), red_glass)
Box(Point3D(-0.5, 0, -2.5), Point3D(0.5, 0.25, 0.5), node, rotate(0, 0, 60) * translate(0, 1, -0.500001), yellow_glass)
Box(Point3D(-0.5, 0, -2.5), Point3D(0.5, 0.25, 0.5), node, rotate(0, 0, 120) * translate(0, 1, -0.500001), green_glass)
Box(Point3D(-0.5, 0, -2.5), Point3D(0.5, 0.25, 0.5), node, rotate(0, 0, 180) * translate(0, 1, -0.500001), cyan_glass)
Box(Point3D(-0.5, 0, -2.5), Point3D(0.5, 0.25, 0.5), node, rotate(0, 0, 240) * translate(0, 1, -0.500001), blue_glass)
Box(Point3D(-0.5, 0, -2.5), Point3D(0.5, 0.25, 0.5), node, rotate(0, 0, 300) * translate(0, 1, -0.500001), purple_glass)

camera = PinholeCamera((256, 256), fov=45, parent=world, transform=translate(0, 0, -6.5) * rotate(0, 0, 0))

camera.ray_max_depth = 500
camera.ray_extinction_prob = 0.01
camera.pixel_samples = 100
camera.spectral_rays = 1
camera.spectral_bins = 21
camera.render_engine = MPIEngine()

# With the MPI engine, only the rank 0 process has all the results available
# to generate sampling statistics. So don't output stats on the workers.
if camera.render_engine.rank != 0:
    camera.quiet = True

# Don't try making plots during the render: this will likely fail when run
# through a non-interactive job scheduler.
for pipeline in camera.pipelines:
    pipeline.display_progress = False

camera.observe()

# Again, only rank 0 has all the results, so this is the only process that
# should produce any output.
if camera.render_engine.rank == 0:
    # Comment out if you don't want to save the result to a file:
    camera.pipelines[0].save("raysect_logo.png")
    # Uncomment if running interactively for a plot of the result:
    camera.pipelines[0].display()
