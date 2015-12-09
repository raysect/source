from raysect.primitive import Mesh
from raysect.primitive import import_obj
import time

print("read obj")
t = time.time()
# m = import_obj("demos/resources/stanford_bunny.obj")
m = import_obj("/home/alex/Shared/alex/work/ccfe/JET/full/obj/LBSRP_Tiles.obj")
print(time.time() - t)

print("write rsm")
t = time.time()
m.save("newmesh.rsm")
print(time.time() - t)

print("load rsm")
t = time.time()
m2 = Mesh.from_file("newmesh.rsm")
print(time.time() - t)

print("write rsm")
t = time.time()
m2.save("repeat.rsm")
print(time.time() - t)