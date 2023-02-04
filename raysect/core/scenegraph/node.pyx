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

cdef class Node(_NodeBase):
    """
    The scene-graph node class.

    The basic constituent of a scene-graph tree. Nodes can be linked together
    by parenting one Node to another to form a tree structure. Each node in a
    scene-graph represents a distinct co-ordinate system. An affine transform
    associated with each node describes the relationship between a node and its
    parent's coordinate system. By combining the transforms (and inverse
    transforms) along the path between two nodes in the tree, the direct
    transform between any two arbitrary nodes, and thus their co-ordinate
    systems, can be calculated. Using this transform it is then possible to
    transform vectors and points between the two co-ordinate systems.

    :param Node parent: Assigns the Node's parent to the specified scene-graph object.
    :param AffineMatrix3D transform: Sets the affine transform associated with the Node.
    :param str name: A string defining the node name.

    :ivar list children: A list of child nodes for which this node is the parent.
    :ivar dict meta: A dictionary for the storage of any extra user specified meta data.
    :ivar Node root: A reference to the root node of this node's scene-graph
      (i.e. the parent of all parents.
    """

    def __init__(self, object parent=None, AffineMatrix3D transform=None, str name=None):

        super().__init__(name)

        if transform is None:
            transform = AffineMatrix3D()

        # prevent _modified() being called during initialisation
        self._track_modifications = False

        self._name = name
        self._transform = transform
        self.parent = parent

        # re-enable _modified() calls
        self._track_modifications = True

    @property
    def parent(self):
        """
        The parent of this node in the scenegraph.

        :rtype: Node
        """
        return self._parent

    @parent.setter
    def parent(self, object value):

        if self._parent is value:
            # the parent is unchanged, do nothing
            return

        if value is None:

            # _parent cannot be None (due to above if statement) so it must be a node, disconnect from current parent
            self._parent.children.remove(self)
            self._parent = None
            self._update()

        else:

            if not isinstance(value, _NodeBase):
                raise TypeError("The specified parent is not a scene-graph node or None (unparented).")

            # prevent cyclic connections
            self._check_parent(value)

            if self._parent is None:

                # connect to parent
                self._parent = value
                self._parent.children.append(self)

                # propagate new state
                self._update()

            else:

                # disconnect from current parent
                self._parent.children.remove(self)

                # connect to parent
                self._parent = value
                self._parent.children.append(self)

                # propagate new state
                self._update()

    @property
    def transform(self):
        """
        The transform for this node's coordinate system in relation to the parent node.

        :rtype: AffineMatrix3D
        """
        return self._transform

    @transform.setter
    def transform(self, AffineMatrix3D value not None):
        self._transform = value
        self._update()

    @property
    def name(self):
        """
        The name of this node.

        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, str value not None):
        self._name = value

    cpdef AffineMatrix3D to(self, _NodeBase node):
        """
        Returns an affine transform that, when applied to a vector or point,
        transforms the vector or point from the co-ordinate space of the calling
        node to the co-ordinate space of the target node.

        For example, if space B is translated +100 in x compared to space A and
        A.to(B) is called then the matrix returned would represent a translation
        of -100 in x. Applied to point (0,0,0) in A, this would produce the
        point (-100,0,0) in B as B is translated +100 in x compared to A.

        :param _NodeBase node: The target node.
        :return: An AffineMatrix3D describing the coordinate transform.
        :rtyoe: AffineMatrix3D
        """

        if self.root is node.root:
            return node._root_transform_inverse.mul(self._root_transform)
        else:
            raise ValueError("The target node must be in the same scene-graph.")

    cpdef AffineMatrix3D to_local(self):
        """
        Returns an affine transform from world space into this nodes local
        coordinate space.

        :rtype: AffineMatrix3D
        """
        return self._root_transform_inverse

    cpdef AffineMatrix3D to_root(self):
        """
        Returns an affine transform from local space into the parent node's
        coordinate space.

        :rtype: AffineMatrix3D
        """
        return self._root_transform
