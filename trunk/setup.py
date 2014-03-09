from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("raysect.core.classes", ["raysect/core/classes.pyx"]),
    Extension("raysect.core.acceleration.accelerator", ["raysect/core/acceleration/accelerator.pyx"]),
    Extension("raysect.core.acceleration.unaccelerated", ["raysect/core/acceleration/unaccelerated.pyx"]),
    Extension("raysect.core.acceleration.boundingbox", ["raysect/core/acceleration/boundingbox.pyx"]),
    Extension("raysect.core.math._vec3", ["raysect/core/math/_vec3.pyx"]),
    Extension("raysect.core.math._mat4", ["raysect/core/math/_mat4.pyx"]),
    Extension("raysect.core.math.vector", ["raysect/core/math/vector.pyx"]),
    Extension("raysect.core.math.normal", ["raysect/core/math/normal.pyx"]),
    Extension("raysect.core.math.point", ["raysect/core/math/point.pyx"]),
    Extension("raysect.core.math.affinematrix", ["raysect/core/math/affinematrix.pyx"]),
    Extension("raysect.core.scenegraph._nodebase",  ["raysect/core/scenegraph/_nodebase.pyx"]),
    Extension("raysect.core.scenegraph.node",  ["raysect/core/scenegraph/node.pyx"]),
    Extension("raysect.core.scenegraph.primitive",  ["raysect/core/scenegraph/primitive.pyx"]),
    Extension("raysect.core.scenegraph.observer",  ["raysect/core/scenegraph/observer.pyx"]),
    Extension("raysect.core.scenegraph.world",  ["raysect/core/scenegraph/world.pyx"]),
    Extension("raysect.primitives.sphere",  ["raysect/primitives/sphere.pyx"]),
    Extension("raysect.tests.speed_test_functions", ["raysect/tests/speed_test_functions.pyx"]),
    Extension("raysect.demo.cython", ["raysect/demo/cython.pyx"])
    ]

setup(
    ext_modules = cythonize(extensions)
)