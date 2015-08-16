from raysect.optical import World, Node, translate, rotate, Point, d65_white, ConstantSF, InterpolatedSF
from raysect.optical.observer.camera import PinholeCamera
from raysect.optical.material.emitter import UniformVolumeEmitter, UniformSurfaceEmitter, Checkerboard
from raysect.optical.material.lambert import Lambert
from raysect.optical.material.dielectric import Dielectric, Sellmeier
from raysect.optical.material.glass_libraries import schott
from raysect.primitive import Sphere, Box, Cylinder, Union, Intersect, Subtract
from matplotlib.pyplot import *
from numpy import array

world = World()

# sky
Sphere(1000, world, material=UniformSurfaceEmitter(d65_white, 1.0))

# floor
Box(Point(-10, -10, 0), Point(10, 10, 1), world, material=Lambert(ConstantSF(1)))

# LOGO VARIANT 1
# wavelengths = array([300, 490, 510, 590, 610, 800])
# red_attn = array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0]) * 0.98
# green_attn = array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]) * 0.88
# blue_attn = array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]) * 0.98
#
# red_glass = Dielectric(index=ConstantSF(1.4), transmission=InterpolatedSF(wavelengths, red_attn))
# green_glass = Dielectric(index=ConstantSF(1.4), transmission=InterpolatedSF(wavelengths, green_attn))
# blue_glass = Dielectric(index=ConstantSF(1.4), transmission=InterpolatedSF(wavelengths, blue_attn))
#
# node = Node(parent=world, transform=rotate(0, 0, 210))
# Box(Point(-0.5, 0, -2.5), Point(0.5, 0.18, 0.5), node, rotate(0, 0, 0) * translate(0, 0.288675136, -0.500001), red_glass)
# Box(Point(-0.5, 0, -2.5), Point(0.5, 0.18, 0.5), node, rotate(0, 0, 120) * translate(0, 0.288675136, -0.500001), green_glass)
# Box(Point(-0.5, 0, -2.5), Point(0.5, 0.18, 0.5), node, rotate(0, 0, 240) * translate(0, 0.288675136, -0.500001), blue_glass)
# camera = PinholeCamera(fov=45, parent=world, transform=translate(0, 0, -5) * rotate(0, 0, 0))

# LOGO VARIANT 2
# wavelengths = array([300, 490, 510, 590, 610, 800])
# red_attn = array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0]) * 1.0
# green_attn = array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]) * 0.9
# blue_attn = array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]) * 1.0
#
# red_glass = Dielectric(index=ConstantSF(1.4), transmission=InterpolatedSF(wavelengths, red_attn))
# green_glass = Dielectric(index=ConstantSF(1.4), transmission=InterpolatedSF(wavelengths, green_attn))
# blue_glass = Dielectric(index=ConstantSF(1.4), transmission=InterpolatedSF(wavelengths, blue_attn))
#
# node = Node(parent=world, transform=rotate(0, 0, 210))
# Box(Point(-0.5, 0, -2.5), Point(0.5, 0.18, 0.5), node, rotate(0, 0, 0) * translate(0, 0.288675136, -0.500001), red_glass)
# Box(Point(-0.5, 0, -2.5), Point(0.5, 0.18, 0.5), node, rotate(0, 0, 120) * translate(0, 0.288675136, -0.500001), green_glass)
# Box(Point(-0.5, 0, -2.5), Point(0.5, 0.18, 0.5), node, rotate(0, 0, 240) * translate(0, 0.288675136, -0.500001), blue_glass)
# camera = PinholeCamera(fov=45, parent=world, transform=translate(0, 0, -5) * rotate(0, 0, 0))

# LOGO VARIANT 3
# wavelengths = array([300, 490, 510, 590, 610, 800])
# red_attn = array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0]) * 0.95
# green_attn = array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]) * 0.80
# blue_attn = array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]) * 0.95
#
# red_glass = Dielectric(index=ConstantSF(1.4), transmission=InterpolatedSF(wavelengths, red_attn))
# green_glass = Dielectric(index=ConstantSF(1.4), transmission=InterpolatedSF(wavelengths, green_attn))
# blue_glass = Dielectric(index=ConstantSF(1.4), transmission=InterpolatedSF(wavelengths, blue_attn))
#
# yellow_attn = array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]) * 0.80
# cyan_attn = array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]) * 0.80
# purple_attn = array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0]) * 0.90
#
# yellow_glass = Dielectric(index=ConstantSF(1.4), transmission=InterpolatedSF(wavelengths, yellow_attn))
# cyan_glass = Dielectric(index=ConstantSF(1.4), transmission=InterpolatedSF(wavelengths, cyan_attn))
# purple_glass = Dielectric(index=ConstantSF(1.4), transmission=InterpolatedSF(wavelengths, purple_attn))
#
# node = Node(parent=world, transform=rotate(0, 0, 0))
# Box(Point(-0.5, 0, -0.75), Point(0.5, 0.5, 0.5), node, rotate(0, 0, 0) * translate(0, 1, -0.500001), red_glass)
# Box(Point(-0.5, 0, -0.75), Point(0.5, 0.5, 0.5), node, rotate(0, 0, 60) * translate(0, 1, -0.500001), yellow_glass)
# Box(Point(-0.5, 0, -0.75), Point(0.5, 0.5, 0.5), node, rotate(0, 0, 120) * translate(0, 1, -0.500001), green_glass)
# Box(Point(-0.5, 0, -0.75), Point(0.5, 0.5, 0.5), node, rotate(0, 0, 180) * translate(0, 1, -0.500001), cyan_glass)
# Box(Point(-0.5, 0, -0.75), Point(0.5, 0.5, 0.5), node, rotate(0, 0, 240) * translate(0, 1, -0.500001), blue_glass)
# Box(Point(-0.5, 0, -0.75), Point(0.5, 0.5, 0.5), node, rotate(0, 0, 300) * translate(0, 1, -0.500001), purple_glass)
# camera = PinholeCamera(fov=45, parent=world, transform=translate(0, 0, -6) * rotate(0, 0, 0))

# LOGO VARIANT 4
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

node = Node(parent=world, transform=rotate(0, 0, 90))
Box(Point(-0.5, 0, -2.5), Point(0.5, 0.25, 0.5), node, rotate(0, 0, 0) * translate(0, 1, -0.500001), red_glass)
Box(Point(-0.5, 0, -2.5), Point(0.5, 0.25, 0.5), node, rotate(0, 0, 60) * translate(0, 1, -0.500001), yellow_glass)
Box(Point(-0.5, 0, -2.5), Point(0.5, 0.25, 0.5), node, rotate(0, 0, 120) * translate(0, 1, -0.500001), green_glass)
Box(Point(-0.5, 0, -2.5), Point(0.5, 0.25, 0.5), node, rotate(0, 0, 180) * translate(0, 1, -0.500001), cyan_glass)
Box(Point(-0.5, 0, -2.5), Point(0.5, 0.25, 0.5), node, rotate(0, 0, 240) * translate(0, 1, -0.500001), blue_glass)
Box(Point(-0.5, 0, -2.5), Point(0.5, 0.25, 0.5), node, rotate(0, 0, 300) * translate(0, 1, -0.500001), purple_glass)
camera = PinholeCamera(fov=45, parent=world, transform=translate(0, 0, -6.5) * rotate(0, 0, 0))

ion()
camera.ray_min_depth = 3
camera.ray_max_depth = 500
camera.ray_extinction_prob = 0.01
camera.pixel_samples = 100
camera.rays = 1
camera.spectral_samples = 21
camera.pixels = (512, 512)
camera.display_progress = True
camera.display_update_time = 10
camera.sub_sample = True
camera.observe()

ioff()
camera.save("raysect_logo.png")
camera.display()
show()

