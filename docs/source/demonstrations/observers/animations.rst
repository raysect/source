
.. _demo_making_animations:

Making Animations
=================

This example demonstrates how to render a series of raysect images and turn them into an animation for web.

.. literalinclude:: ../../../../demos/animation.py

.. image:: animation.gif
   :align: center

You can use `ImageMagic's <https://www.imagemagick.org/>`_ convert command to make a gif.

::

   $> convert -delay 20 -loop 0 *.png myimage.gif
::

