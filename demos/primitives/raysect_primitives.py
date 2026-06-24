# External imports
from time import strftime

from matplotlib import pyplot as plt

# Raysect imports
from raysect.optical import Point3D, World, d65_white, rotate, translate
from raysect.optical.library import schott
from raysect.optical.material import Checkerboard, Lambert
from raysect.optical.observer import PinholeCamera, RGBAdaptiveSampler2D, RGBPipeline2D
from raysect.primitive import Box, Cone, Cylinder, Parabola, Sphere, Torus

# 1. Create Primitives
# --------------------

# Box defining the ground plane
ground = Box(
    lower=Point3D(-50, -0.01, -50), upper=Point3D(50, 0.0, 50), material=Lambert()
)

# checker board wall that acts as emitter
emitter = Box(
    lower=Point3D(-100, -100, 10),
    upper=Point3D(100, 100, 10.1),
    material=Checkerboard(4, d65_white, d65_white, 0.1, 2.0),
)

# Primitive showcasing all geometric features
# Note that the primitives must be displaced slightly above the ground plane to prevent numerically issues that could
# cause a light leak at the intersection between the objects and the ground.
cylinder = Cylinder(
    radius=1.5,
    height=3.0,
    transform=translate(1.5 * 3 + 1.0, 0.0001, 0) * rotate(0, 90, 0),
    material=schott("N-BK7"),
)
cone = Cone(
    radius=1.5,
    height=3.0,
    transform=translate(1.5 + 0.2, 0.0001, 0) * rotate(0, 90, 0),
    material=schott("N-BK7"),
)
sphere = Sphere(
    radius=1.5,
    transform=translate(-1.5 - 0.2, 1.5 + 0.0001, 0),
    material=schott("N-BK7"),
)
box = Box(
    lower=Point3D(-1.5, 0.0, -1.5),
    upper=Point3D(1.5, 3.0, 1.5),
    transform=translate(-1.5 * 3 - 1.0, 0.0001, 0),
    material=schott("N-BK7"),
)
parabola = Parabola(
    radius=2.0,
    height=1.0,
    transform=translate(2.5, 1.0 + 0.0001, -5.0) * rotate(0, -90, 0),
    material=schott("N-BK7"),
)
torus = Torus(
    major_radius=1.0,
    minor_radius=0.5,
    transform=translate(-2.5, 0.5 + 0.0001, -5.0) * rotate(0, 90, 0),
    material=schott("N-BK7"),
)


# 2. Add Observer
# ---------------

# Process the ray-traced spectra with the RGB pipeline.
rgb = RGBPipeline2D(display_unsaturated_fraction=0.96)
sampler = RGBAdaptiveSampler2D(
    rgb, ratio=10, fraction=0.2, min_samples=2000, cutoff=0.01
)

# camera
camera = PinholeCamera(
    (512, 512), pipelines=[rgb], transform=translate(-7, 12, -15) * rotate(-25, -40, 0)
)

# camera - pixel sampling settings
camera.fov = 45
camera.pixel_samples = 250

# camera - ray sampling settings
camera.spectral_rays = 15
camera.spectral_bins = 15
camera.ray_max_depth = 100
camera.ray_extinction_prob = 0.1
camera.min_wavelength = 375.0
camera.max_wavelength = 740.0


# 3. Build Scenegraph
# -------------------

world = World()

ground.parent = world
emitter.parent = world
camera.parent = world
cylinder.parent = world
cone.parent = world
sphere.parent = world
box.parent = world
parabola.parent = world
torus.parent = world

# 4. Observe()
# ------------
name = "raysect_primitives"
timestamp = strftime("%Y-%m-%d_%H-%M-%S")
render_pass = 1
plt.ion()
while not camera.render_complete:
    print(f"Rendering pass {render_pass}...")
    camera.observe()
    rgb.save(f"{name}_{timestamp}_pass_{render_pass}.png")
    render_pass += 1
    print()

# display final result
plt.ioff()
rgb.display()
plt.show()
