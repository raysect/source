
# External imports
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Raysect imports
from raysect.optical import World, translate, rotate, Point3D, d65_white, Ray, Vector3D
from raysect.optical.material.absorber import AbsorbingSurface
from raysect.optical.library import schott
from raysect.primitive import Sphere, Box
from raysect.optical.loggingray import LoggingRay
from raysect.primitive.lens.spherical import *


world = World()

# Create a glass BiConvex lens we want to study
lens_glass = schott("N-BK7")
lens_glass.transmission_only = True
lens = BiConvex(0.0254, 0.0052, 0.0506, 0.0506, parent=world, material=lens_glass)

# Create a target plane behind the lens.
target = Box(lower=Point3D(-50, -50, -0), upper=Point3D(50, 50, 0), material=AbsorbingSurface(),
             transform=translate(0, 0, 0.1), parent=world)


# for each sample direction trace a logging ray and plot the ray trajectory
plt.ion()
fig = plt.figure()
ax = fig.gca(projection='3d')

for u in np.linspace(-0.006, 0.006, 5):
    for v in np.linspace(-0.012, 0.012, 11):

        start = Point3D(v, u, -0.05)
        log_ray = LoggingRay(start, Vector3D(0, 0, 1))
        log_ray.trace(world)

        p = [(start.x, start.y, start.z)]
        for point in log_ray.log:
            p.append((point.x, point.y, point.z))
        p = np.array(p)

        ax.plot(p[:, 0], p[:, 1], p[:, 2], 'k-')
        ax.plot(p[:, 0], p[:, 1], p[:, 2], 'r.')

plt.ioff()
plt.show()
