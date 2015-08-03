
from raysect.optical import World, translate, rotate, Point, Vector, d65_white
from raysect.optical.observer.camera import PinholeCamera
from raysect.optical.material.emitter import UniformSurfaceEmitter, Checkerboard

from raysect.primitive import Box
from raysect.primitive._cone import Cone
from matplotlib.pyplot import *
from raysect.optical.material.debug import Light

import sys
sys.ps1 = 'SOMETHING'

world = World()

cone = Cone(material=Light(Vector(0.2, -0.1, 1.0)), parent=world, transform=translate(1.5, 1.5, 0) * rotate(0, 0, 0))
cone = Cone(material=Light(Vector(0.2, -0.1, 1.0)), parent=world, transform=translate(0, 1.5, 0) * rotate(0, 45, 0))
cone = Cone(material=Light(Vector(0.2, -0.1, 1.0)), parent=world, transform=translate(-1.5, 1.5, 0) * rotate(0, 90, 0))
cone = Cone(material=Light(Vector(0.2, -0.1, 1.0)), parent=world, transform=translate(1.5, -0, 0) * rotate(0, 150, 0))
cone = Cone(material=Light(Vector(0.2, -0.1, 1.0)), parent=world, transform=translate(0, -0, 0) * rotate(0, 180, 0))
cone = Cone(material=Light(Vector(0.2, -0.1, 1.0)), parent=world, transform=translate(-1.5, -0, 0) * rotate(0, 225, 0))
cone = Cone(material=Light(Vector(0.2, -0.1, 1.0)), parent=world, transform=translate(1.5, -1.5, 0) * rotate(20, 0, 0))
cone = Cone(material=Light(Vector(0.2, -0.1, 1.0)), parent=world, transform=translate(0, -1.5, 0) * rotate(45, 0, 0))
cone = Cone(material=Light(Vector(0.2, -0.1, 1.0)), parent=world, transform=translate(-1.5, -1.5, 0) * rotate(90, 0, 0))

Box(Point(-50, -50, 50), Point(50, 50, 50.1), world, material=Checkerboard(4, d65_white, d65_white, 0.4, 0.8))
Box(Point(-100, -100, -100), Point(100, 100, 100), world, material=UniformSurfaceEmitter(d65_white, 0.1))

ion()
camera = PinholeCamera(fov=45, parent=world, transform=translate(0, 0, -8) * rotate(0, 0, 0),
                       spectral_samples=3, super_samples=5)
camera.ray_max_depth = 10
camera.rays = 1
camera.spectral_samples = 1
camera.pixels = (512, 512)
camera.display_progress = True
camera.display_update_time = 10
camera.observe()

ioff()
camera.save("render.png")
camera.display()
show()

