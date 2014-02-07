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

cdef class Primitive(Node):

    def __str__(self):
        """String representation."""
    
        if self.name == "":
            
            return "<Primitive at " + str(hex(id(self))) + ">"
        
        else:
            
            return self.name + " <Primitive at " + str(hex(id(self))) + ">"
    
    cpdef Intersection hit(self, Ray ray):
        """
        Virtual method - to be implemented by derived classes.
        
        Calculates the closest intersection of the Ray with the Primitive 
        surface, if such an intersection exists.
        
        Must return an Intersection object. If no intersection occurs the
        Intersection attribute hit is set to False. If hit is True then the 
        other attributes of the Intersection object will be filled with the 
        calculated values related to the intersection.
        """
    
        raise NotImplementedError("Virtual method hit() has not been implemented.")
    
    cpdef bint inside(self, Point p):
        """
        Virtual method - to be implemented by derived classes.
        
        Returns True if the Point lies within the boundary of the surface
        defined by the Primitive. False is returned otherwise.
        """        
    
        raise NotImplementedError("Virtual method inside() has not been implemented.")
    
    #cpdef BoundingBox bounding_box(self):
    
        #pass

