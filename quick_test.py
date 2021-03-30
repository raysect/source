import timeit
from raysect.optical import *
from raysect.optical.unpolarised import Ray as URay
from raysect.optical.polarised import Ray as PRay
from raysect.optical.material import UnitySurfaceEmitter, UnityVolumeEmitter
from raysect.primitive import Sphere

w = World()
# Sphere(1.0, parent=w, material=UnitySurfaceEmitter())
Sphere(1.0, parent=w, material=UnityVolumeEmitter())

print(timeit.timeit('r.sample(w, 1000000).samples', setup='r = URay()', globals=globals(), number=1))
print(timeit.timeit('r.sample(w, 1000000).samples', setup='r = PRay()', globals=globals(), number=1))

r = PRay()
print(r.trace(w).samples)