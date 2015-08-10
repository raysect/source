from raysect.core.acceleration.boundingbox import BoundingBox
from random import random
from raysect.core import Point, Ray, Vector, World, rotate
from raysect.primitive.mesh import Triangle, Mesh
import sys
from time import time

from raysect.primitive.mesh import import_obj

# vertices = [
#     Point(-0.5, -0.5, -0.5),
#     Point(0.5, -0.5, -0.5),
#     Point(-0.5, 0.5, -0.5),
#     Point(0.5, 0.5, -0.5),
#     Point(-0.5, -0.5, 0.5),
#     Point(0.5, -0.5, 0.5),
#     Point(-0.5, 0.5, 0.5),
#     Point(0.5, 0.5, 0.5)
# ]
#
# triangles = [
#     Triangle(vertices[0], vertices[3], vertices[1]),
#     Triangle(vertices[4], vertices[7], vertices[5])
# ]

triangles = []
for i in range(int(sys.argv[1])):
    triangles.append(Triangle(Point(-0.5, -0.5, i), Point(0.5, -0.5, i), Point(0, 0.5, i)))



w = World()
m = import_obj("/home/alex/Shared/alex/work/MAST/mesh/MAST-M9/obj/MAST-M9-BEAM DUMPS + GDC.obj", scaling=1e-3, parent=w, transform=rotate(0,90,0))
#m = Mesh(triangles, parent=w)
# m.debug_print_all()

