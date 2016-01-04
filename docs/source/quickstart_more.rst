==============
Camera example
==============
- create a world
- place a primitive (sphere) in a scene with a lambert material.
- Primitive = anything that rays interact with. Only rays. Convention => object => primitive.
- put sphere on something => a ground plane (lambert).
- need to illuminate it => checkerboard
- scene but nothing to oberve. Anything that fires rays and samples the scene.
- Add a camera
- Move to right place, explain affine transforms.
- Render!
- oh its all noisy
- explain statistical sampling and path tracer
- turn up number of samples
- set camera to accumulate.
- re-sample
- change color of sphere? Define reflectivity function
- change material to glass


================================
Spectral Examples based on scene
================================
- reset sphere to white
- manual sampling of scene
- sample d65 white spectra
- convert sphere to coloured lambert, reflected absorption curve.
- convert sphere to coloured glass, define refractive index and absorption curve.


- have a look at flask quickstart