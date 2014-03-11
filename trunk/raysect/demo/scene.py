# cython: language_level=3

# Copyright (c) 2014, Dr Alex Meakins, Raysect Project
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the Raysect Project nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from time import time
from matplotlib.pyplot import imshow, imsave, show, ion, ioff, clf, figure
from raysect.core import World, translate, rotate
from raysect.primitive import Sphere
from raysect.demo.material import GlowGaussianBeam
from raysect.demo.camera import PinholeCamera
from raysect.demo.support import RGB

def run_demo(pixels = (500, 500), display_progress = True):

    print("Raysect demo script")

    world = World("The World")

    sphere_r = Sphere(world, translate(0, 1, 5), GlowGaussianBeam(RGB(0.25, 0, 0), 0.2, 0.05), 2.0, "Red Object")
    sphere_g = Sphere(world, translate(0.707, -0.707, 5), GlowGaussianBeam(RGB(0, 0.25, 0), 0.2, 0.05), 2.0, "Green Object")
    sphere_b = Sphere(world, translate(-0.707, -0.707, 5), GlowGaussianBeam(RGB(0, 0, 0.25), 0.2, 0.05), 2.0, "Blue Object")

    #sphere_world = Sphere(100.0, world, rotate(0, 0, 0), Checkerboard())
    #sphere_world.material.scale = 10

    camera = PinholeCamera(world, translate(0, 0, 0) * rotate(0, 0, 0), pixels, 45, "Camera")

    ion()

    camera.display_progress = display_progress

    t = time()
    camera.observe()
    t = time() - t
    print("Render time was " + str(t) + " seconds")

    camera.display()

    return (world, camera)