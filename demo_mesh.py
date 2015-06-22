from raysect.core.acceleration import Unaccelerated
from raysect.optical import World, translate, rotate, Point, Vector, Ray, d65_white, ConstantSF, SampledSF
from raysect.optical.observer.pinholecamera import PinholeCamera
from raysect.optical.material.emitter import UniformVolumeEmitter, UniformSurfaceEmitter, Checkerboard
from raysect.optical.material import debug
from raysect.primitive import Box, Mesh, Sphere
from matplotlib.pyplot import *
from numpy import array

world = World()

# test data:
vertices = [
    (1.4, -1, -1),   # left
    (0, -1., 1.4),   # forward
    (-1.4, -1, -1),  # right
    (0, 1, 0)        # up
]

polygons = [
    (0, 1, 2),
    (0, 2, 3),
    (0, 3, 1),
    (2, 1, 3)
]

Mesh(vertices, polygons, world, translate(0, 0, 0)*rotate(40, 0, 0), debug.Light(Vector(1.0, -0.5, 1.0)))
# Mesh(vertices, polygons, world, translate(0, 0, 0)*rotate(0, 90, 0), debug.Normal('+x'))
# Mesh(vertices, polygons, world, translate(0, 0, 0)*rotate(0, 90, 0), debug.Exiting())

Box(Point(-50, -50, 50), Point(50, 50, 50.1), world, material=Checkerboard(4, d65_white, d65_white, 0.4, 0.8))
Box(Point(-100, -100, -100), Point(100, 100, 100), world, material=UniformSurfaceEmitter(d65_white, 0.1))

ion()
camera = PinholeCamera(fov=45, parent=world, transform=translate(0, 0, -4) * rotate(0, 0, 0), process_count=3)
camera.ray_max_depth = 15
camera.rays = 1
camera.spectral_samples = 15
camera.pixels = (256, 256)
camera.display_progress = True
camera.display_update_time = 5
camera.super_samples = 1
camera.observe()

ioff()
camera.save("demo_mesh_render.png")
camera.display()
show()

