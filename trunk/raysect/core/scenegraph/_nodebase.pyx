# cython: language_level=3

#Copyright (c) 2014, Dr Alex Meakins, Raysect Project
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

cdef class _NodeBase:
    """
    The base class from which all scene-graph objects are derived.
    
    Defines the core attributes and common functionality of the scene-graph
    node objects.
    """
    
    def __init__(self):
        """Base class constructor."""

        self._parent = None
        self.children = []
        self.root = self
        self._transform = AffineMatrix()
        self._root_transform = AffineMatrix()
        self._root_transform_inverse = AffineMatrix()
        
    cpdef AffineMatrix to(self, _NodeBase node):
        """
        Returns an affine transform that, when applied to a vector or point, 
        transforms the vector or point from the co-ordinate space of the calling
        node to the co-ordinate space of the target node.
        
        For example, if space B is translated +100 in x compared to space A and
        A.to(B) is called then the matrix returned would represent a translation
        of -100 in x. Applied to point (0,0,0) in A, this would produce the
        point (-100,0,0) in B as B is translated +100 in x compared to A.
        """
        
        if self.root == node.root:

            return node._root_transform_inverse.mul(self._root_transform)
        
        else:
                
            raise ValueError("The target node must be in the same scenegraph.")

    def _check_parent(self, _NodeBase parent):
        """
        Raises an exception if the this node or its decendents are passed.
        
        The purpose of this function is to enforce the structure of the scene-
        graph. A scene-graph is logically a tree and so cannot contain cyclic
        references.
        """
    
        if parent is self:
            
            raise ValueError("A node cannot be parented to itself or one of it's decendants.")
        
        for child in self.children:
            
            child._check_parent(parent)
        
    def _update(self):
        """
        Instructs the node to recalculate the root transforms for its section of
        the scene graph. Automatically registers/deregisters the node with the
        root node (if the methods are implemented). Propagates a reference to
        the root node through the tree.
        
        This method is called automatically when the scenegraph below this node
        changes or if the node transform or parent are modified. This method
        should never be called manually.
        """
        
        if self._parent is None:
            
            if self.root is not self:
            
                # node has need disconnected from a scenegraph, de-register from old root node                
                self.root._deregister(self)
                self.root = self

            # this node is now a root node
            self._root_transform = AffineMatrix()
            self._root_transform_inverse = AffineMatrix()
        
        else:

            # is node connecting to a different scenegraph?
            if self.root is not self._parent.root:
                
                # de-register from old root and register with new root
                self.root._deregister(self)
                self.root = self._parent.root
                self._parent.root._register(self)

            # update root transforms
            self._root_transform = (<_NodeBase> self._parent)._root_transform.mul(self._transform)
            self._root_transform_inverse = self._root_transform.inverse()
            
        # propagate changes to children
        for child in self.children:
            
            child._update()
        
    def _register(self, _NodeBase node):
        """
        When implemented by root nodes this method allows nodes in the
        scene-graph to register themselves for special handling.
        
        Virtual method call.
        
        For use in conjunction with _deregister()
        """        
    
        pass
    
    def _deregister(self, _NodeBase node):
        """
        When implemented by root nodes this method allows nodes in the
        scene-graph to deregister themselves.
        
        Virtual method call.
        
        For use in conjunction with _register()
        """        
    
        pass        

