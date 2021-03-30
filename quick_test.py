import timeit
from raysect.optical import *
from raysect.optical.unpolarised import Ray as URay
from raysect.optical.polarised import Ray as PRay
from raysect.optical.material import UnitySurfaceEmitter, UnityVolumeEmitter, Lambert
from raysect.primitive import Sphere

w = World()
# Sphere(1.0, parent=w, material=UnitySurfaceEmitter())
Sphere(1.0, parent=w, material=Lambert(d65_white))
Sphere(50.0, parent=w, material=UnitySurfaceEmitter())

# print(timeit.timeit('r.sample(w, 100000).samples', setup='r = URay(origin=Point3D(0, 0, -5))', globals=globals(), number=1))
# print(timeit.timeit('r.sample(w, 100000).samples', setup='r = PRay(origin=Point3D(0, 0, -5))', globals=globals(), number=1))

r = URay(origin=Point3D(0, 0, -5), bins=5)
print(r.trace(w).samples[:])

r = PRay(origin=Point3D(0, 0, -5), bins=5)
print(r.trace(w).samples[:,:])


