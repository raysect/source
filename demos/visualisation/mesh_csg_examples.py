
from mayavi import mlab

from raysect.core import translate, Point3D, rotate_basis, Vector3D, rotate
from raysect.optical import World
from raysect.primitive import Box, Sphere, Cylinder, Cone
from raysect.primitive import Sphere, Mesh, Intersect, Subtract, Union

from raysect.visualisation import visualise_scenegraph


########################################################################################################################
# Spheres


s1 = Sphere(0.5, transform=translate(-0.25, 0, 0), name='s1')
s2 = Sphere(0.5, transform=translate(0.25, 0, 0), name='s2')


world = World()
Union(s1, s2, parent=world)
visualise_scenegraph(world)
input('pause...')


world = World()
Intersect(s1, s2, parent=world)
visualise_scenegraph(world)
input('pause...')


world = World()
Subtract(s1, s2, parent=world)
visualise_scenegraph(world)
input('pause...')


########################################################################################################################
# Cubes

b1 = Box(Point3D(0, 0, 0), Point3D(1, 1, 1))
b2 = Box(Point3D(0, 0, 0), Point3D(1, 1, 1), transform=translate(0.6, 0.6, 0.6))

world = World()
Union(b1, b2, parent=world)
visualise_scenegraph(world)
input('pause...')

world = World()
Intersect(b1, b2, parent=world)
visualise_scenegraph(world)
input('pause...')

world = World()
Subtract(b1, b2, parent=world)
visualise_scenegraph(world)
input('pause...')


########################################################################################################################
# Cylinders

c1 = Cylinder(0.5, 2)
c2 = Cylinder(0.5, 2, transform=translate(0.15, 0, 0.4))

world = World()
Union(c1, c2, parent=world)
visualise_scenegraph(world)
input('pause...')

world = World()
Intersect(c1, c2, parent=world)
visualise_scenegraph(world)
input('pause...')

world = World()
Subtract(c1, c2, parent=world)
visualise_scenegraph(world)
input('pause...')


########################################################################################################################
# Box and Sphere

s1 = Sphere(0.5)
b2 = Box(Point3D(-0.5, -0.5, -0.5), Point3D(0.5, 0.5, 0.5), transform=translate(-0.2, 0, -0.2))

world = World()
Union(s1, b2, parent=world)
visualise_scenegraph(world)
input('pause...')

world = World()
Intersect(s1, b2, parent=world)
visualise_scenegraph(world)
input('pause...')

world = World()
Subtract(s1, b2, parent=world)
visualise_scenegraph(world)
input('pause...')


########################################################################################################################
# Cone and Cylinder

c1 = Cone(0.15, 1)
c2 = Cylinder(0.15, 0.5, transform=translate(-0.25, 0, 0.5)*rotate_basis(Vector3D(1, 0, 0), Vector3D(0, 0, 1)))

world = World()
Union(c1, c2, parent=world)
visualise_scenegraph(world)
input('pause...')

world = World()
Intersect(c1, c2, parent=world)
visualise_scenegraph(world)
input('pause...')

world = World()
Subtract(c1, c2, parent=world)
visualise_scenegraph(world)
input('pause...')


########################################################################################################################
# CSG Hell

world = World()
cyl_x = Cylinder(1, 4.2, transform=rotate(90, 0, 0)*translate(0, 0, -2.1))
cyl_y = Cylinder(1, 4.2, transform=rotate(0, 90, 0)*translate(0, 0, -2.1))
cyl_z = Cylinder(1, 4.2, transform=rotate(0, 0, 0)*translate(0, 0, -2.1))
cube = Box(Point3D(-1.5, -1.5, -1.5), Point3D(1.5, 1.5, 1.5))
sphere = Sphere(2.0)

csg = Intersect(sphere, Subtract(cube, Union(Union(cyl_x, cyl_y), cyl_z)), world)

visualise_scenegraph(world)
input('pause...')
