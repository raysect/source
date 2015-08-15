Raysect Python Raytracing Package
=================================

A raytracing framework for optical/non-optical physics simulations.

The aims of the Raysect project are as follows:
* develop a ray-tracer that is easy for scientists and engineers to use and extend
* the raytracer must be robust and high precision (double floating point throughout) 

The general development philosopy is ease of use > performance, but performance is not to be ignored.

Please note, this code is currently alpha quality and subject to significant change.

If you would like to experiment with the code you can use our development scripts. The setup.py does not currently install a working module (it's coming soon!). Raysect is written in Cython, so you will need Cython installed. To setup the dev version, clone the repository and run the dev/build.sh script while in the raysect root directory. This will build the package in-place. The package can then be added to your python path and used.

Raysect currently only supports Linux. Windows support will come at a later stage.

Please note, for legal reasons we require the copyright to any contributed code to be passed to the Raysect project.
