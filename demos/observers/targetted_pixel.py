
from matplotlib.pyplot import ion, ioff, show

from raysect.primitive import Box
from raysect.optical import World, translate, Point3D, Node
from raysect.optical.observer import Pixel, TargettedPixel, PowerPipeline0D
from raysect.optical.material import UnitySurfaceEmitter


SAMPLES = 100000

world = World()


# # create a small emitting box, simulating a 1x10 mm slit 100 mm from a 10x10 mm pixel surface, 20mm off axis
# emitter = Box(Point3D(-0.005, -0.0005, -0.0005), Point3D(0.005, 0.0005, 0.0005), world, translate(0.02, 0, 0.10), UnitySurfaceEmitter())
# targets = [emitter]

# create a small emitting box, same as above, but split into 10 1x1x1mm cubes so that the bounding spheres are a tighter fit to the slit.
emitter = Node(parent=world, transform=translate(0.02, 0, 0.10))
targets = []
for i in range(10):
    section = Box(Point3D(-0.0005, -0.0005, -0.0005), Point3D(0.0005, 0.0005, 0.0005), emitter, translate(0.001 * i - 0.0045, 0, 0), UnitySurfaceEmitter())
    targets.append(section)


# setup basic pixel
basic_pipeline = PowerPipeline0D(name="Basic Pixel Observer")
basic_pixel = Pixel(parent=world, pixel_samples=SAMPLES, pipelines=[basic_pipeline])

# setup targetted pixel
targetted_pipeline = PowerPipeline0D(name="Targeted Pixel Observer")
targetted_pixel = TargettedPixel(parent=world, targets=targets, pixel_samples=SAMPLES, pipelines=[targetted_pipeline])
targetted_pixel.targetted_path_prob = 1

# render
ion()
basic_pixel.observe()
print()
targetted_pixel.observe()

ioff()
show()
