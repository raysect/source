
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from raysect.optical.observer.point_generator import *
from raysect.optical.observer.vector_generators import *


plt.ion()


sp = SinglePointGenerator()
points = sp.sample(1000)
x = [p.x for p in points]
y = [p.y for p in points]
plt.figure()
plt.plot(x, y, '.')
plt.axis('equal')
plt.title('SinglePointGenerator')
plt.show()


sp = CircularPointGenerator()
points = sp.sample(1000)
x = [p.x for p in points]
y = [p.y for p in points]
plt.figure()
plt.plot(x, y, '.')
plt.axis('equal')
plt.title('CircularPointGenerator')
plt.show()


sp = RectangularPointGenerator()
points = sp.sample(1000)
x = [p.x for p in points]
y = [p.y for p in points]
plt.figure()
plt.plot(x, y, '.')
plt.axis('equal')
plt.title('RectangularPointGenerator')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sp = SingleRay()
vectors = sp.sample(1000)
for i in range(1000):
    v = vectors[i]
    ax.scatter(v.x, v.y, v.z, marker='.')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_zlim(-1.2, 1.2)
plt.title('SingleRay')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sp = Cone(acceptance_angle=PI/8)
vectors = sp.sample(1000)
for i in range(1000):
    v = vectors[i]
    ax.scatter(v.x, v.y, v.z, marker='.')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_zlim(-1.2, 1.2)
plt.title('Cone')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sp = Hemisphere()
vectors = sp.sample(1000)
for i in range(1000):
    v = vectors[i]
    ax.scatter(v.x, v.y, v.z, marker='.')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_zlim(-1.2, 1.2)
plt.title('Hemisphere')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sp = CosineHemisphere()
vectors = sp.sample(1000)
for i in range(1000):
    v = vectors[i]
    ax.scatter(v.x, v.y, v.z, marker='.')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_zlim(-1.2, 1.2)
plt.title('CosineHemisphere')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sp = CosineHemisphereWithForwardBias(forward_bias=0.75)
vectors = sp.sample(1000)
for i in range(1000):
    v = vectors[i]
    ax.scatter(v.x, v.y, v.z, marker='.')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_zlim(-1.2, 1.2)
plt.title('CosineHemisphereWithForwardBias')
plt.show()

