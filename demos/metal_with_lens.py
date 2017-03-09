
from raysect.optical import World, translate, rotate, Point3D, d65_white, ConstantSF
from raysect.primitive import Sphere, Box, Cylinder, Subtract
from raysect.primitive.lens.spherical import BiConvex
from raysect.optical.library.metal import Gold, Silver, Copper, Titanium, Aluminium, Beryllium
from raysect.optical.material import Lambert, UniformSurfaceEmitter, AbsorbingSurface
from raysect.optical.library import schott
from matplotlib.pyplot import *
from raysect.optical.observer import RGBPipeline2D, BayerPipeline2D, CCDArray, RGBAdaptiveSampler2D
from raysect.optical.colour import ciexyz_x, ciexyz_y, ciexyz_z

world = World()

Sphere(0.5, world, transform=translate(1.2, 0.5001, 0.6), material=Gold())
Sphere(0.5, world, transform=translate(0.6, 0.5001, -0.6), material=Silver())
Sphere(0.5, world, transform=translate(0, 0.5001, 0.6), material=Copper())
Sphere(0.5, world, transform=translate(-0.6, 0.5001, -0.6), material=Titanium())
Sphere(0.5, world, transform=translate(-1.2, 0.5001, 0.6), material=Aluminium())
Sphere(0.5, world, transform=translate(0, 0.5001, -1.8), material=Beryllium())

Box(Point3D(-100, -0.1, -100), Point3D(100, 0, 100), world, material=Lambert(ConstantSF(1.0)))
Cylinder(3.0, 8.0, world, transform=translate(4, 8, 0) * rotate(90, 0, 0), material=UniformSurfaceEmitter(d65_white, 1.0))

rgb = RGBPipeline2D(display_unsaturated_fraction=0.98, name="sRGB")
bayer = BayerPipeline2D(ciexyz_x, ciexyz_y, ciexyz_z, display_unsaturated_fraction=0.98, name="Bayer Filter")

sampler = RGBAdaptiveSampler2D(rgb, ratio=10, fraction=0.2, min_samples=500, cutoff=0.05)
pipelines = [rgb, bayer]

camera = CCDArray(parent=world, transform=translate(0, 4, -3.5) * rotate(0, -48, 180), pipelines=pipelines)
camera.frame_sampler = sampler
camera.pixels = (180*5, 120*5)  # (360, 240)
camera.pixel_samples = 250
camera.spectral_rays = 1
camera.spectral_bins = 20

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

# start ray tracing
ion()
p = 1
while not camera.render_complete:
    print("Rendering pass {}...".format(p))
    camera.observe()
    # camera.pipelines[0].save("demo_metal_lens_{}.png".format(p))
    print()
    p += 1
