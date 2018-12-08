
============
Introduction
============


What is raysect
~~~~~~~~~~~~~~~

Raysect is an open-source python framework for geometrical optical simulations. It is designed
to be a physically realistic ray-tracer generally applicable to research problems in science
and engineering. A core philosophy at the heart of Raysect's design is that scientific robustness
and flexibility takes precedence over speed. For other use cases where speed is more important
there are plenty of suitable ray-tracers, such as Povray for example. Raysect has been designed
to be easy to extend by a physicist/engineer.

Feature set
~~~~~~~~~~~

- Path tracer for efficient configurable ray-tracing.
- Full scenegraph for managing complex geometry and coordinate transforms.
- A complete set of geometric primitives, lens types, meshes and CSG operations.
- A wide range of simulated physical observers such as CCDs, cameras, fibreoptics, etc.
- Advanced optical material models, an associated material library (BRDFs), metals, glasses.
- Supports serial or multi-core operation. MPI not currently supported.
- Geometric optics => ray simulations with prisims, lenses, etc.

Structure/Architecture
~~~~~~~~~~~~~~~~~~~~~~

- Raysect is an OOP framework written in a combination of python and cython. All major functionality
  is accessible from python. It is possible to extend all components from python, however to gain full
  speed, the cython api should be used.
- The core of raysect is actually completely generalised and can be used for other ray-tracing applications
  such as neutron transport, etc. However, at the present time the optical model is the only application
  which has been implemented.
- The core of Raysect is a generalised kernel for calculating interactions with rays and or volumes onto
  which physics models that require raytracing (such as geometric optics) can be built.

Contributions
~~~~~~~~~~~~~

Anyone is welcome to make contributions to Raysect using the standard OpenSource git workflow.
Contributing developers should fork Raysect into their own github account and develop on their own
local branches. Code can be periodically merged through pull requests. Anyone can contribute bug
reports and documentation ideas through our `github issue queue <https://github.com/raysect/source/issues>`_.
