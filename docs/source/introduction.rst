
Introduction
------------

- What is raysect

    - An open-source python framework for geometrical optical simulations.

- Where to use raysect?
    - Why instead of povray, etc?
    - Science/engineering perspective.
    - Robustness over speed, philosophy.
    - Designed to be easy to extend by a physicist/engineer.

- Feature set
    - Path tracer
    - Full scenegraph for managing geometry and coordinate transforms.
    - Set of geometric primitives, lens types, meshes and CSG.
    - Simulated Physical Observers => CCDs, cameras, fibreoptics.
    - Optical materials, associated material library (BRDFs), metals, glasses.
    - multi-core.
    - geometric optics => lenses, blah.

- Structure/Architecture
    - OOP framework written in a combination of python and cython. All major functionality is accessible from python. It
      is possible to extend all components from python, however to gain full speed, the cython api should be used.
    - The core of raysect is actually completely generalised and can be used for other ray-tracing applications
      such as neutron transport, etc. However, at the present time the optical model is the only application which has
      been implemented.
    - The core of Raysect is a generalised kernel for calculating interactions with rays and or volumes onto which
      physics models that require raytracing (such as geometric optics) can be built.

- Contributions
    - Welcome, but...