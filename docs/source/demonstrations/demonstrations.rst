
Core Functionality
==================

.. list-table:: Core functionality examples
   :widths: 28 50 22
   :header-rows: 1

   * - Name
     - Description
     - Preview
   * - :ref:`Ray Intersection Points <demo_ray_intersection_points>`
     - Tracking of and visualisation of where rays intersect with objects in the scene.
     - .. image:: core/ray_intersection_points_fig1.png
          :height: 150px
          :width: 150px
   * - :ref:`Point Inside A Material <demo_point_inside_material>`
     - Finding all primitives which contain a test point.
     - .. image:: core/test_point_inside_material.png
          :height: 150px
          :width: 150px
   * - :ref:`Energy conservation <demo_energy_conservation>`
     - Checking all emitted light is collected correctly on a 3D enclosing surface.
     -
   * - :ref:`Multiple Importance Sampling <demo_multiple_importance_sampling>`
     - The classic multiple importance sampling demo re-implemented from E. Veach's PhD thesis.
     - .. image:: core/multiple_importance_sampling.jpg
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
     - .. image:: materials/emissive_colours.png
          :height: 150px
          :width: 150px
   * - :ref:`Diffuse colours <demo_diffuse_colours>`
     - Simple diffuse coloured materials.
     - .. image:: materials/diffuse_colours.png
          :height: 150px
          :width: 150px
   * - :ref:`Metals <demo_metal_materials>`
     - Loading metal materials from the library.
     - .. image:: materials/metal_balls.png
          :height: 150px
          :width: 150px
   * - :ref:`Glass <demo_glass_bunny>`
     - Loading glass materials from the library.
     - .. image:: materials/glass_bunny.jpg
          :height: 150px
          :width: 150px
   * - :ref:`Diamond <demo_diamond_material>`
     - Making a diamond material.
     - .. image:: materials/diamond.jpg
          :height: 150px
          :width: 150px
   * - :ref:`Surface roughness <demo_surface_roughness_scan>`
     - Material properties can be varied from smooth to rough with a material roughness modifier.
     - .. image:: materials/surface_roughness.jpg
          :height: 150px
          :width: 150px
   * - :ref:`Anisotropic surface emitters <demo_anisotropic_emitters>`
     - Make an anisotropic material with a custom emission function.
     - .. image:: materials/anisotropic_emitters_preview.png
          :height: 150px
          :width: 150px
   * - :ref:`Custom volume emitters <demo_custom_volume_emitters>`
     - Make a custom volume emitter with your own 3D function.
     - .. image:: materials/volume_inhomogeneous.png
          :height: 150px
          :width: 150px
   * - :ref:`Tetrahedral volume mesh emitters <demo_tetrahedra_mesh_emitters>`
     - Make a tetrahedral volume mesh emitter.
     - .. image:: materials/tetrahedra_mesh_emitter.png
          :height: 150px
          :width: 150px
   * - :ref:`Polar BRDF plots <demo_polar_brdf_plots>`
     - Producing polar plots of material BRDFs.
     - .. image:: materials/brdf_polar_plots.png
          :height: 150px
          :width: 150px
   * - :ref:`3D surface BRDF plots <demo_surface3d_brdf_plots>`
     - Producing 3D surface plots of material BRDFs.
     - .. image:: materials/brdf_surface3d_plots.png
          :height: 150px
          :width: 150px

Observers
=========

.. list-table:: Examples of the different types of observers
   :widths: 28 50 22
   :header-rows: 1

   * - Name
     - Description
     - Preview
   * - :ref:`Cornell Box <demo_cornell_box>`
     - An industry standard test scene for benchmarking ray-tracers.
       Also demonstrates how to setup and configure a basic pinhole
       camera in Raysect.
     - .. image:: observers/cornell_box_mis_1550_samples.png
          :height: 150px
          :width: 150px
   * - :ref:`Orthographic camera <demo_orthographic_camera>`
     - Using the orthographic camera.
     - .. image:: observers/orthographic_camera.png
          :height: 150px
          :width: 150px
   * - :ref:`Optical fibre <demo_optical_fibre>`
     - Spectral observations from an optical fibre.
     - .. image:: observers/optical_fibre_power.png
          :height: 150px
          :width: 150px
   * - :ref:`Mesh Observers <demo_mesh_observers>`
     - Making observations on a mesh surface.
     - .. image:: observers/mesh_observers.jpg
          :height: 150px
          :width: 150px
   * - :ref:`Making a camera 1 <demo_cornell_box_with_camera>`
     - Making a camera from components.
     - .. image:: observers/cornell_box_real_lens.png
          :height: 150px
          :width: 150px
   * - :ref:`Making a camera 2 <demo_metal_ball_with_lens>`
     - Another example of making a camera from components.
     - .. image:: observers/metal_balls_with_lens.png
          :height: 150px
          :width: 150px
   * - :ref:`Making animations <demo_making_animations>`
     - Looping over the observe loop whilst changing the position of primitives generates an animation.
     - .. image:: observers/animation_preview.jpg
          :height: 150px
          :width: 150px


Optics
======

.. list-table:: Example scenes
   :widths: 28 50 22
   :header-rows: 1

   * - Name
     - Description
     - Preview
   * - :ref:`Prism dispersion <demo_prism_dispersion>`
     - White light is split into its component colours as it passes through a glass prism.
     - .. image:: optics/prism_720x405.png
          :height: 150px
          :width: 150px
   * - :ref:`Etendue <demo_etendue_of_pinhole>`
     - Calculating the etendue of a pinhole with ray-tracing.
     - .. image:: optics/etendue_of_pinhole.png
          :height: 150px
          :width: 150px



