
# External imports
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
plt.ion()

# Raysect imports
from raysect.optical import World, translate, rotate_x, Point3D, d65_white, Ray, Vector3D
from raysect.optical.material.absorber import AbsorbingSurface
from raysect.optical.library import schott
from raysect.primitive import Sphere, Box, Cylinder, Subtract
from raysect.optical.observer import FibreOptic, PowerPipeline0D, SpectralPipeline0D
from raysect.optical.loggingray import LoggingRay
from raysect.primitive.lens.spherical import *


world = World()

sk4_lens_glass = schott("N-SK4")
sk4_lens_glass.transmission_only = True
f4_lens_glass = schott("F4")
f4_lens_glass.transmission_only = True


# Example double gauss lens design from Robert Jördens Rayopt package:
# http://nbviewer.jupyter.org/github/jordens/rayopt-notebooks/blob/master/double_gauss.ipynb

lens1 = Meniscus(0.2, 0.05083, 0.25907, 1.47341, parent=world, material=sk4_lens_glass, transform=translate(0, 0, 0.1 + 0.05083)*rotate_x(180))

lens2 = Meniscus(0.15, 0.01694, 0.34804, 0.1734, parent=world, material=f4_lens_glass, transform=translate(0, 0, 0.17438 + 0.01694)*rotate_x(180))

aperture_stop = Subtract(Box(lower=Point3D(-1, -1, -0), upper=Point3D(1, 1, 0.0001)),
                         Cylinder(radius=0.0515, height=0.001),
                         material=AbsorbingSurface(), transform=translate(0, 0, 0.21674), parent=world)

lens3 = Meniscus(0.15, 0.01694, 0.34804, 0.1734, parent=world, material=f4_lens_glass, transform=translate(0, 0, 0.24214))

lens4 = Meniscus(0.2, 0.05083, 0.25907, 1.47341, parent=world, material=sk4_lens_glass, transform=translate(0, 0, 0.28263))

# Create a target plane behind the lens.
target = Box(lower=Point3D(-50, -50, -0), upper=Point3D(50, 50, 0.001), material=AbsorbingSurface(),
             transform=translate(0, 0, 1.24), parent=world)


# for each sample direction trace a logging ray and plot the ray trajectory
trajectories_2d = []
trajectories_3d = []
for v in np.linspace(-0.05, 0.05, 5):

    start = Point3D(v, 0, 0)
    log_ray = LoggingRay(start, Vector3D(0, 0, 1))
    log_ray.trace(world)

    hit_points_3d = [(start.x, start.y, start.z)]
    hit_points_2d = [(start.x, start.z)]
    for point in log_ray.log:
        hit_points_3d.append((point.x, point.y, point.z))
        hit_points_2d.append((point.x, point.z))

    trajectories_3d.append(np.array(hit_points_3d))
    trajectories_2d.append(np.array(hit_points_2d))

plt.figure()
for trajectory_2d in trajectories_2d:
    plt.plot(trajectory_2d[:, 1], trajectory_2d[:, 0], 'k')
    plt.plot(trajectory_2d[:, 1], trajectory_2d[:, 0], 'r.')

plt.axis('equal')

fig = plt.figure()
ax_3d = fig.gca(projection='3d')
for trajectory_3d in trajectories_3d:
    ax_3d.plot(trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2], 'k')
    ax_3d.plot(trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2], 'r.')

ax_3d.set_xlim(-0.1, 0.1)
ax_3d.set_ylim(-0.1, 0.1)
ax_3d.set_zlim(0, 0.5)
plt.show()
