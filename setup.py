from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

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
    Extension("raysect.core.math.function", ["raysect/core/math/function.pyx"]),
    Extension("raysect.core.math.utility", ["raysect/core/math/utility.pyx"]),
    Extension("raysect.core.scenegraph._nodebase",  ["raysect/core/scenegraph/_nodebase.pyx"]),
    Extension("raysect.core.scenegraph.node",  ["raysect/core/scenegraph/node.pyx"]),
    Extension("raysect.core.scenegraph.primitive",  ["raysect/core/scenegraph/primitive.pyx"]),
    Extension("raysect.core.scenegraph.observer",  ["raysect/core/scenegraph/observer.pyx"]),
    Extension("raysect.core.scenegraph.world",  ["raysect/core/scenegraph/world.pyx"]),
    Extension("raysect.primitive.sphere",  ["raysect/primitive/sphere.pyx"]),
    Extension("raysect.tests.speed_test_functions", ["raysect/tests/speed_test_functions.pyx"]),
    Extension("raysect.demo.material", ["raysect/demo/material.pyx"]),
    Extension("raysect.demo.ray", ["raysect/demo/ray.pyx"]),
    Extension("raysect.demo.support", ["raysect/demo/support.pyx"]),
    Extension("raysect.optical.ray", ["raysect/optical/ray.pyx"]),
    Extension("raysect.optical.colour", ["raysect/optical/colour.pyx"]),
    Extension("raysect.optical.spectrum", ["raysect/optical/spectrum.pyx"], include_dirs=[".", numpy.get_include()]),
    Extension("raysect.optical.material.material", ["raysect/optical/material/material.pyx"]),
    Extension("raysect.optical.material.emitter", ["raysect/optical/material/emitter.pyx"]),
    Extension("raysect.optical.material.demo", ["raysect/optical/material/demo.pyx"])
    ]

setup(
    ext_modules = cythonize(extensions)
    #ext_modules = cythonize(extensions, force = True)
    #ext_modules = cythonize(extensions, force = True, compiler_directives = {"profile": True})
    #ext_modules = cythonize(extensions, compiler_directives = {"profile": True})
)