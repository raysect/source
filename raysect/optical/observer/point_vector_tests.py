
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from raysect.optical.observer.point_generator import *
from raysect.optical.observer.vector_generators import *


sp = SinglePointGenerator()
points = sp.sample(1000)
x = [p.x for p in points]
y = [p.y for p in points]
plt.plot(x, y, '.')
plt.axis('equal')
plt.title('SinglePointGenerator')
plt.show()


sp = CircularPointGenerator()
points = sp.sample(1000)
x = [p.x for p in points]
y = [p.y for p in points]
plt.plot(x, y, '.')
plt.axis('equal')
plt.title('CircularPointGenerator')
plt.show()


sp = RectangularPointGenerator()
points = sp.sample(1000)
x = [p.x for p in points]
y = [p.y for p in points]
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
plt.axis('equal')
plt.title('Cone')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sp = Cone()
vectors = sp.sample(1000)
for i in range(1000):
    v = vectors[i]
    ax.scatter(v.x, v.y, v.z, marker='.')
plt.axis('equal')
plt.title('Cone')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sp = Hemisphere()
vectors = sp.sample(1000)
for i in range(1000):
    v = vectors[i]
    ax.scatter(v.x, v.y, v.z, marker='.')
plt.axis('equal')
plt.title('Hemisphere')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sp = CosineHemisphere()
vectors = sp.sample(1000)
for i in range(1000):
    v = vectors[i]
    ax.scatter(v.x, v.y, v.z, marker='.')
plt.axis('equal')
plt.title('CosineHemisphere')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sp = CosineHemisphereWithForwardBias(forward_bias=2.0)
vectors = sp.sample(1000)
for i in range(1000):
    v = vectors[i]
    ax.scatter(v.x, v.y, v.z, marker='.')
plt.axis('equal')
plt.title('CosineHemisphereWithForwardBias')
plt.show()

