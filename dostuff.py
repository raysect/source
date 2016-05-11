from matplotlib.pylab import *
from raysect.optical import *
from raysect.optical.scenegraph.world import ImportanceManager
from raysect.primitive import Sphere
from raysect.optical.material import Lambert
from raysect.core.math.random import vector_sphere

w = World()
s1 = Sphere(1, parent=w, transform=translate(10,0,0), material=Lambert())
s2 = Sphere(1, parent=w, transform=translate(0,8,0), material=Lambert())
s3 = Sphere(1, parent=w, transform=translate(0,0,0), material=Lambert())
p = [s1, s2, s3]
p[0].material.importance = 90
p[1].material.importance = 50
p[2].material.importance = 60
im = ImportanceManager(p)

sum = 0
for i in range(100000):
    sum += im.pdf(Point3D(3, 0, 0), vector_sphere())

print(sum / 100000 * 4 * 3.141)


f = figure(1)
for i in range(1000):
    v = im.sample(Point3D(3, 0, 0))
    plot([0, v.x], [0, v.y])

show()
