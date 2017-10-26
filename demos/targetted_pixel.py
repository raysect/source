from matplotlib.pyplot import ion, ioff, show
from raysect.optical import World, translate, Point3D
from raysect.optical.observer import Pixel, TargetedPixel, PowerPipeline0D
from raysect.optical.material import UnitySurfaceEmitter
from raysect.primitive import Box


SAMPLES = 100000

world = World()

# create a small emitting box, simulating a 1x10 mm slit 100 mm from a 10x10 mm pixel surface, 20mm off axis
emitter = Box(Point3D(-0.005, -0.0005, -0.0005), Point3D(0.005, 0.0005, 0.0005), world, translate(0.02, 0, 0.10), UnitySurfaceEmitter())

# setup basic pixel
basic_pipeline = PowerPipeline0D(name="Basic Pixel Observer")
basic_pixel = Pixel(parent=world, pixel_samples=SAMPLES, pipelines=[basic_pipeline])

# setup targeted pixel
targeted_pipeline = PowerPipeline0D(name="Targeted Pixel Observer")
targeted_pixel = TargetedPixel(parent=world, target=emitter, pixel_samples=SAMPLES, pipelines=[targeted_pipeline])

# render
ion()
basic_pixel.observe()
print()
targeted_pixel.observe()

ioff()
show()

