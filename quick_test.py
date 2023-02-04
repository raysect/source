import timeit
from raysect.optical import *
from raysect.optical.material import UnitySurfaceEmitter, UnityVolumeEmitter, Lambert, Retarder
from raysect.primitive import Sphere, Cylinder, Subtract


def polariser(parent=None, transform=None):

    assembly = Node(parent=parent, transform=transform)

    polariser = Cylinder(
        radius=0.101, height=0.002, parent=assembly, transform=translate(0, 0, -0.001),
        material=LinearPolariser(Vector3D(0, 1, 0))
    )

    body = Subtract(
        Cylinder(radius=0.11, height=0.010),
        Cylinder(radius=0.1, height=0.011, transform=translate(0, 0, -0.0005)),
        parent=assembly,
        transform=translate(0, 0, -0.005),
        material=Lambert(ConstantSF(0.05))
    )

    handle = Cylinder(
        radius=0.004, height=0.021, parent=assembly,
        transform=translate(0, 0.109, 0)*rotate(0, 90, 0),
        material=Lambert(ConstantSF(0.05))
    )

    return assembly


w = World()
Sphere(50.0, parent=w, material=UnitySurfaceEmitter())
polariser(w, transform=translate(0, 0, 0.02) * rotate(0, 0, 45))
Cylinder(0.5, 0.01, parent=w, material=Retarder(90 / 0.01, axis=Vector3D(0, 1, 1)))

r = Ray(origin=Point3D(0, 0, -5), orientation=Vector3D(0, 1, 0), bins=1)
s = StokesVector(*r.trace(w).samples[0, :])
print(f's0: {s[0]:0.4f}\ns1: {s[1]:0.4f}\ns2: {s[2]:0.4f}\ns3: {s[3]:0.4f}')
print(s.polarised_fraction())



