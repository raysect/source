
Core Functionality
==================

.. list-table:: Core API examples
   :widths: 28 50 22
   :header-rows: 1

   * - Name
     - Description
     - Preview
   * - :ref:`Ray Intersection Points <demo_ray_intersection_points>`
     - Tracking of and visualisation of where rays intersect with objects in the scene.
     - .. image:: ray_intersection_points_fig1.png
          :height: 150px
          :width: 150px
   * - :ref:`Point Inside A Material <demo_point_inside_material>`
     - Finding all primitives which contain a test point.
     - .. image:: test_point_inside_material.png
          :height: 150px
          :width: 150px

Materials
=========

.. list-table:: Examples of the different material properties available
   :widths: 28 50 22
   :header-rows: 1

   * - Name
     - Description
     - Preview
   * - :ref:`Emissive colours <demo_emissive_colours>`
     - Simple coloured emissive materials.
     - .. image:: emissive_colours.png
          :height: 150px
          :width: 150px
   * - :ref:`Diffuse colours <demo_diffuse_colours>`
     - Simple diffuse coloured materials.
     - .. image:: diffuse_colours.png
          :height: 150px
          :width: 150px
   * - :ref:`Surface roughness <demo_surface_roughness_scan>`
     - Material properties can be varied from smooth to rough with a material roughness modifier.
     - .. image:: surface_roughness.jpg
          :height: 150px
          :width: 150px
   * - :ref:`Anisotropic surface emitters <demo_anisotropic_emitters>`
     - Make an anisotropic material with a custom emission function.
     - .. image:: anisotropic_emitters_preview.png
          :height: 150px
          :width: 150px


Examples Gallery
================

.. list-table:: Example scenes
   :widths: 28 50 22
   :header-rows: 1

   * - Name
     - Description
     - Preview
   * - :ref:`Cornell Box <demo_cornell_box>`
     - An industry standard test scene for benchmarking ray-tracers.
     - .. image:: cornell_box_mis_1550_samples.png
          :height: 150px
          :width: 150px
   * - :ref:`Prism dispersion <demo_prism_dispersion>`
     - White light is split into its component colours as it passes through a glass prism.
     - .. image:: prism_720x360.jpg
          :height: 150px
          :width: 150px
   * - :ref:`Making animations <demo_making_animations>`
     - Looping over the observe loop whilst changing the position of primitives generates an animation.
     - .. image:: animation_preview.jpg
          :height: 150px
          :width: 150px
   * - :ref:`Energy conservation <demo_energy_conservation>`
     - Checking all emitted light is collected correctly on a 3D enclosing surface.
     -



