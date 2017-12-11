
# External imports
import matplotlib.pyplot as plt
import os

# Raysect imports
from raysect.optical import World, translate, rotate, Point3D, d65_white
from raysect.optical.observer import PinholeCamera
from raysect.optical.material import UniformSurfaceEmitter
from raysect.optical.library import RoughAluminium
from raysect.primitive import Box
from raysect.optical.material import AnisotropicSurfaceEmitter


# Construct our anisotropic emitter
class BlueYellowEmitter(AnisotropicSurfaceEmitter):

    def emission_function(self, spectrum, cosine, back_face):

        spectrum.samples[:] = 1.0
        spectrum.samples[spectrum.wavelengths > 500] = 2 * cosine
        return spectrum


# 1. Create Primitives
# --------------------

# Box defining the floor
ground = Box(lower=Point3D(-100, -14.2, -100), upper=Point3D(100, -14.1, 100), material=RoughAluminium(0.05))

# Box defining the isotropic emitter
emitterIsotropic = Box(lower=Point3D(-25, -10, -10), upper=Point3D(-5, 10, 10),
              material=UniformSurfaceEmitter(d65_white), transform=rotate(0, 0, 0))
# Box defining the anisotropic emitter
emitterAnisotropic = Box(lower=Point3D(5, -10, -10), upper=Point3D(25, 10, 10),
              material=BlueYellowEmitter(), transform=rotate(0, 0, 0))


# 2. Add Observer
# ---------------

# camera
camera = PinholeCamera((384, 384), transform=translate(0, 0, -80))
camera.fov = 45
camera.pixel_samples = 250
camera.pipelines[0].accumulate = False


# 3. Build Scenegraph
# -------------------

world = World()
ground.parent = world
emitterIsotropic.parent = world
emitterAnisotropic.parent = world
camera.parent = world


# 4. Observe and animate
# ----------------------

if not os.path.isdir("anim/aniso/"):
    os.makedirs("anim/aniso/")

plt.ion()
num_frames = 45
for frame in range(num_frames):
    print("Rendering frame {}:".format(frame))
    emitterIsotropic.transform = rotate(0, 2*frame, 0)
    emitterAnisotropic.transform = rotate(0, 2*frame, 0)
    camera.observe()
    camera.pipelines[0].save("anim/aniso/frame_%03d.png"%frame)
    camera.pipelines[0].display()
    plt.show()

plt.ioff()

# create gif animation
# os.system("convert --loop anim/aniso/*.png anisotropic_emitter.gif")