
import numpy as np
import matplotlib.pyplot as plt

from raysect.core import Point3D, Vector3D, rotate_basis, translate, Ray as CoreRay
from raysect.core.math.sampler import DiskSampler3D, RectangleSampler3D, TargettedHemisphereSampler
from raysect.optical import World
from raysect.primitive import Box, Cylinder, Subtract
from raysect.optical.material import AbsorbingSurface, NullMaterial

R_2_PI = 1 / (2 * np.pi)


world = World()

# Setup pinhole
target_plane = Box(Point3D(-10, -10, -0.000001), Point3D(10, 10, 0.000001))
hole = Cylinder(0.001, 0.001, transform=translate(0, 0, -0.0005))
pinhole = Subtract(target_plane, hole, parent=world, material=AbsorbingSurface())

target = Cylinder(0.0012, 0.001, transform=translate(0, 0, -0.0011), parent=world, material=NullMaterial())


def analytic_etendue(area_det, area_slit, distance, alpha, gamma):

    return area_det * area_slit * np.cos(alpha/360 * (2*np.pi)) * np.cos(gamma/360 * (2*np.pi)) / distance**2


def raytraced_etendue(distance, detector_radius=0.001, ray_count=100000, batches=10):

    # generate the transform to the detector position and orientation
    detector_transform = translate(0, 0, distance) * rotate_basis(Vector3D(0, 0, -1), Vector3D(0, -1, 0))

    # generate bounding sphere and convert to local coordinate system
    sphere = target.bounding_sphere()
    spheres = [(sphere.centre.transform(detector_transform), sphere.radius, 1.0)]

    # instance targetted pixel sampler
    targetted_sampler = TargettedHemisphereSampler(spheres)

    point_sampler = DiskSampler3D(detector_radius)

    detector_area = detector_radius**2 * np.pi
    solid_angle = 2 * np.pi
    etendue_sampled = solid_angle * detector_area

    etendues = []
    for i in range(batches):

        # sample pixel origins
        origins = point_sampler(samples=ray_count)

        passed = 0.0
        for origin in origins:

            # obtain targetted vector sample
            direction, pdf = targetted_sampler(origin, pdf=True)
            path_weight = R_2_PI * direction.z/pdf

            origin = origin.transform(detector_transform)
            direction = direction.transform(detector_transform)

            while True:

                # Find the next intersection point of the ray with the world
                intersection = world.hit(CoreRay(origin, direction))

                if intersection is None:
                    passed += 1 * path_weight
                    break

                elif isinstance(intersection.primitive.material, NullMaterial):
                    hit_point = intersection.hit_point.transform(intersection.primitive_to_world)
                    origin = hit_point + direction * 1E-9
                    continue

                else:
                    break

        if passed == 0:
            raise ValueError("Something is wrong with the scene-graph, calculated etendue should not zero.")

        etendue_fraction = passed / ray_count
        etendues.append(etendue_sampled * etendue_fraction)

    etendue = np.mean(etendues)
    etendue_error = np.std(etendues)

    return etendue, etendue_error


area = 0.001**2 * np.pi

detector_etendue = 0.001**2 * np.pi * np.pi  # etendue = A * omega * 1/2


distance_samples = [10**i for i in np.arange(-4, -1.1, 0.10)]

analytic_values = []
raytraced_values = []
raytraced_errors = []

for d in distance_samples:
    analytic_values.append(analytic_etendue(area, area, d, 0, 0))
    value, error = raytraced_etendue(d)
    raytraced_values.append(value)
    raytraced_errors.append(error)

analytic_values = np.array(analytic_values)
raytraced_values = np.array(raytraced_values)
raytraced_errors = np.array(raytraced_errors)

plt.ion()

plt.figure()
ax = plt.gca()
ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')
plt.axhline(y=detector_etendue, linestyle='--', color='k', label='detector etendue')
plt.plot(distance_samples, analytic_values, label='analytic etendue')
plt.errorbar(distance_samples, raytraced_values, raytraced_errors, label='ray-traced etendue')
plt.xlabel('Distance between slit and detector (m)')
plt.ylabel('Etendue (m^2 str)')
plt.title('Ray-traced VS approximate analytic etendue')
plt.legend()

# plt.figure()
# ax = plt.gca()
# ax.set_xscale("log", nonposx='clip')
# plt.errorbar(distance_samples, np.abs(raytraced_values-analytic_values)/raytraced_values, raytraced_errors/raytraced_values)
# plt.xlim(0.001, 0.1)
# plt.ylim(0, 0.5)
# plt.xlabel('Distance between slit and detector (m)')
# plt.ylabel('Fractional error')
# plt.show()


