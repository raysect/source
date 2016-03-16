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

# from raysect.core.scenegraph.signal import GEOMETRY

from raysect.core.scenegraph.world cimport World as CoreWorld
# from raysect.core.scenegraph.primitive cimport Primitive
# from raysect.core.scenegraph.observer cimport Observer
# from raysect.core.scenegraph.signal cimport ChangeSignal

cdef class ImportanceManager:

    cdef:
        ndarray importance_cdf
        list importance_primitives



# TODO: update docstrings
cdef class World(CoreWorld):
    """
    The root node of the optical scene-graph.

    The world node tracks all primitives and observers in the world. It maintains acceleration structures to speed up
    the ray-tracing calculations. The particular acceleration algorithm used is selectable. The default acceleration
    structure is a kd-tree.

    :param name: A string defining the node name.
    """


    def __init__(self, str name=None):
        super().__init__(name)
        # TODO: add items

    cpdef build_importance(self, bint force=False):
        # """
        # This method manually triggers a rebuild of the Acceleration object.
        #
        # If the Acceleration object is already in a consistent state this method
        # will do nothing unless the force keyword option is set to True.
        #
        # The Acceleration object is used to accelerate hit() and contains()
        # calculations, typically using a spatial sub-division method. If changes are
        # made to the scene-graph structure, transforms or to a primitive's
        # geometry the acceleration structures may no longer represent the
        # geometry of the scene and hence must be rebuilt. This process is
        # usually performed automatically as part of the first call to hit() or
        # contains() following a change in the scene-graph. As calculating these
        # structures can take some time, this method provides the option of
        # triggering a rebuild outside of hit() and contains() in case the user wants
        # to be able to perform a benchmark without including the overhead of the
        # Acceleration object rebuild.
        #
        # :param bint force: If set to True, forces rebuilding of acceleration structure.
        # """

        if self._rebuild_importance or force:
            # TODO: write me!
            self._rebuild_importance = False

    def _change(self, _NodeBase node, ChangeSignal change not None):
        # """
        # Notifies the World of a change to the scene-graph.
        #
        # This method must be called is a change occurs that may have invalidated
        # any acceleration structures held by the World.
        #
        # The node on which the change occurs and a ChangeSignal must be
        # provided. The ChangeSignal must specify the nature of the change to the
        # scene-graph.
        #
        # The core World object only recognises the GEOMETRY signal. When a
        # GEOMETRY signal is received, the world will be instructed to rebuild
        # it's spatial acceleration structures on the next call to any method
        # that interacts with the scene-graph geometry.
        # """

        super()._change()
        if change is MATERIAL:
            self._rebuild_importance = True

