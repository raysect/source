# cython: language_level=3

# Copyright (c) 2014-2016, Dr Alex Meakins, Raysect Project
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

from raysect.core.scenegraph import signal

from raysect.core.scenegraph._nodebase cimport _NodeBase
from raysect.core.scenegraph.node cimport Node
from raysect.core.scenegraph.signal cimport ChangeSignal

cdef class BridgeNode(Node):
    """
    Specialised scene-graph root node that propagates geometry notifications.
    """

    def __init__(self, _NodeBase destination):

        super().__init__()
        self.destination = destination

    def _change(self, _NodeBase node, ChangeSignal change not None):
        """
        Handles a scene-graph node change handler.

        Propagates change notifications to the specified node and it's
        scene-graph.
        """

        # propagate change notifications from local scene-graph to target's scene-graph
        self.destination.root._change(self.destination, change)


def print_scenegraph(node):
    """
    Pretty-prints a scene-graph.

    This function will print the scene-graph that contains the specified node.
    The specified node will be highlighted in the tree by post-fixing the node
    with the string: "[referring node]".

    :param _NodeBase node: The target node.
    """

    # start from root node
    root = node.root

    # print node
    if root is node:
        print(str(root) + " [referring node]")
    else:
        print(str(root))

    # print children
    n = len(root.children)
    for i in range(0, n):
        if i < (n-1):
            _print_node(root.children[i], "", " |  ", node)
        else:
            _print_node(root.children[i], "", "    ", node)


def _print_node(node, indent, link, highlight):
    """
    Internal function called recursively to print a scene-graph.
    """

    # print node
    print(indent + " |  ")

    if node is highlight:
        print(indent + " |_ " + str(node) + " [referring node]")
    else:
        print(indent + " |_ " + str(node))

    # print children
    n = len(node.children)
    for i in range(0, n):
        if i < (n-1):
            _print_node(node.children[i], indent + link, " |  ", highlight)
        else:
            _print_node(node.children[i], indent + link, "    ", highlight)