# cython: language_level=3

# Copyright (c) 2014-2023, Dr Alex Meakins, Raysect Project
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

from raysect.core.scenegraph.signal import GEOMETRY
from raysect.core.scenegraph.signal cimport ChangeSignal


cdef class _NodeBase:
    """
    The base class from which all scene-graph objects are derived.

    Defines the core attributes and common functionality of the scene-graph
    node objects.
    """

    def __init__(self, str name=None):
        """Base class constructor."""

        self._name = name
        self._parent = None
        self.children = []
        self.root = self
        self._transform = AffineMatrix3D()
        self._root_transform = AffineMatrix3D()
        self._root_transform_inverse = AffineMatrix3D()
        self._track_modifications = True

        # user meta data dictionary
        self.meta = {}

    def __str__(self):
        """String representation."""

        s = "{} at {}".format(self.__class__.__name__, str(hex(id(self))))
        if self.name:
            return "<{}: {}>".format(self.name, s)
        else:
            return "<{}>".format(s)

    def _check_parent(self, _NodeBase parent):
        """
        Raises an exception if this node or its descendants are passed.

        The purpose of this function is to enforce the structure of the scene-
        graph. A scene-graph is logically a tree and so cannot contain cyclic
        references.
        """

        if parent is self:
            raise ValueError("A node cannot be parented to itself or one of it's descendants.")

        for child in self.children:
            child._check_parent(parent)

    def _update(self):
        """
        Instructs the node to recalculate the root transforms for its section of
        the scene graph. Automatically registers/deregisters the node with the
        root node (if the methods are implemented). Propagates a reference to
        the root node through the tree.

        This method is called automatically when the scenegraph above this node
        changes or if the node transform or parent are modified. This method
        should never be called manually.
        """

        if self._parent is None:

            if self.root is not self:

                # node has been disconnected from the scenegraph, de-register with old root node
                self.root._deregister(self)
                self.root = self

            # this node is now a root node
            self._root_transform = AffineMatrix3D()
            self._root_transform_inverse = AffineMatrix3D()

            # report root transforms have changed
            if self._track_modifications:
                self._modified()

        else:

            # is node connecting to a different scenegraph?
            if self.root is not self._parent.root:

                # de-register with old root and register with new root
                self.root._deregister(self)
                self.root = self._parent.root
                self._parent.root._register(self)

            # update root transforms
            self._root_transform = (<_NodeBase> self._parent)._root_transform.mul(self._transform)
            self._root_transform_inverse = self._root_transform.inverse()

            # report root transforms have changed
            if self._track_modifications:
                self._modified()

        # inform root node of change to scene-graph
        self.root._change(self, GEOMETRY)

        # propagate changes to children
        for child in self.children:
            child._update()

    def _register(self, _NodeBase node):
        """
        When implemented by root nodes this method allows nodes in the
        scene-graph to register themselves with the root node for special
        handling.

        Virtual method call.

        For use in conjunction with _deregister()
        """

        pass

    def _deregister(self, _NodeBase node):
        """
        When implemented by root nodes this method allows nodes in the
        scene-graph to deregister themselves with the root node.

        Virtual method call.

        For use in conjunction with _register()
        """

        pass

    def _change(self, _NodeBase node, ChangeSignal change not None):
        """
        When implemented by root nodes this method allows nodes in the
        scene-graph to inform the root node of any change to scene-graph
        structure or to the nodes themselves.

        A ChangeSignal object specifying the nature of the change.

        Virtual method call.
        """

        pass

    def _modified(self):
        """
        This method is called when a scene-graph change occurs that modifies
        the node's root transforms. This will occur if the node's transform is
        modified, a parent node's transform is modified or if the node's
        section of scene-graph is re-parented.

        Virtual method call.
        """

        pass