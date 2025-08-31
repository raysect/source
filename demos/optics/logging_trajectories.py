# External imports
import matplotlib.pyplot as plt
import numpy as np

# Raysect imports
from raysect.optical import Point3D, Vector3D, World, translate
from raysect.optical.library import schott
from raysect.optical.loggingray import LoggingRay
from raysect.optical.material.absorber import AbsorbingSurface
from raysect.primitive import Box
from raysect.primitive.lens.spherical import BiConvex

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
ax = plt.axes(projection='3d')

for u in np.linspace(-0.006, 0.006, 5):
    for v in np.linspace(-0.012, 0.012, 11):

        start = Point3D(v, u, -0.05)
        log_ray = LoggingRay(start, Vector3D(0, 0, 1))
        log_ray.trace(world)

        p = [(start.x, start.y, start.z)]
        for intersection in log_ray.log:
            point = intersection.hit_point
            p.append((point.x, point.y, point.z))
        p = np.array(p)

        ax.plot(p[:, 0], p[:, 1], p[:, 2], 'k-')
        ax.plot(p[:, 0], p[:, 1], p[:, 2], 'r.')

ax.set(
    xlabel="$X$",
    ylabel="$Y$",
    zlabel="$Z$",
)
plt.ioff()
plt.show()
