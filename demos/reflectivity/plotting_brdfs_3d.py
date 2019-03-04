
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from raysect.core import Point3D, Vector3D
from raysect.optical.library.metal import RoughAluminium


plt.ion()

origin = Point3D(0, 0, 0)

aluminium = RoughAluminium(0.25)

def calculate_brdf_surface(light_vector):

    thetas = np.arange(0, 91, step=5)
    num_thetas = len(thetas)
    phis = np.arange(0, 361, step=10)
    num_phis = len(phis)
    thetas, phis = np.meshgrid(thetas, phis)

    X = np.zeros((num_phis, num_thetas))
    Y = np.zeros((num_phis, num_thetas))
    Z = np.zeros((num_phis, num_thetas))

    for i in range(num_phis):
        for j in range(num_thetas):

            theta = np.deg2rad(thetas[i, j])
            phi = np.deg2rad(phis[i, j])
            outgoing = Vector3D(np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta))

            radius = aluminium.bsdf(light_vector, outgoing, 500)
            X[i, j] = radius * np.cos(phi) * np.sin(theta)
            Y[i, j] = radius * np.sin(phi) * np.sin(theta)
            Z[i, j] = radius * np.cos(theta)

    return X, Y, Z


light_angle = 0
light_position = Point3D(np.sin(np.deg2rad(light_angle)), 0, np.cos(np.deg2rad(light_angle)))
light_direction = origin.vector_to(light_position).normalise()
X, Y, Z = calculate_brdf_surface(light_direction)

plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
plt.title("Light angle {} degrees".format(light_angle))


light_angle = -10
light_position = Point3D(np.sin(np.deg2rad(light_angle)), 0, np.cos(np.deg2rad(light_angle)))
light_direction = origin.vector_to(light_position).normalise()
X, Y, Z = calculate_brdf_surface(light_direction)

plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
plt.title("Light angle {} degrees".format(light_angle))


light_angle = -25
light_position = Point3D(np.sin(np.deg2rad(light_angle)), 0, np.cos(np.deg2rad(light_angle)))
light_direction = origin.vector_to(light_position).normalise()
X, Y, Z = calculate_brdf_surface(light_direction)

plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
plt.title("Light angle {} degrees".format(light_angle))


light_angle = -45
light_position = Point3D(np.sin(np.deg2rad(light_angle)), 0, np.cos(np.deg2rad(light_angle)))
light_direction = origin.vector_to(light_position).normalise()
X, Y, Z = calculate_brdf_surface(light_direction)

plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
plt.title("Light angle {} degrees".format(light_angle))

plt.show()
