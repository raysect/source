
import matplotlib.pyplot as plt
from math import atan2, sqrt, degrees

from raysect.core.math import Point3D, TargettedHemisphereSampler, TargettedSphereSampler


def display_samples(samples, title):

    # convert samples into theta and phi angles
    theta = []
    phi = []
    for sample in samples:
        theta.append(degrees(atan2(sample.z, sqrt(sample.x**2 + sample.y**2))))
        phi.append(degrees(atan2(sample.y, sample.x)))

    plt.figure()
    plt.plot(phi, theta, '.', markersize=1)
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.title(title)


samples = 100000

# Target tuple: (sphere_centre, sphere_radius, target_weight)
targets = [
    (Point3D(0, 1, 5), 2, 1),
    (Point3D(-5, -3, 0), 3, 1),
    (Point3D(4, 6, 1.5), 1, 1),
    (Point3D(50, -25, 70), 1, 1),
]

observation_point = Point3D(0, 0, 0)

# generate samplers
hemisphere = TargettedHemisphereSampler(targets)
sphere = TargettedSphereSampler(targets)

# sample for origin point and point at (0, 0, -10)
h_samples = hemisphere(observation_point, samples=samples)
s_samples = sphere(observation_point, samples=samples)

# display
display_samples(h_samples, 'Hemisphere Samples at ({}, {}, {})'.format(*observation_point))
display_samples(s_samples, 'Sphere Samples at ({}, {}, {})'.format(*observation_point))
plt.show()
