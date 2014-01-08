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

"""
Unit tests for the Point object.
"""

import unittest
from ..point import Point
from ..vector import Vector
from math import sqrt

# TODO: Port to Cython to allow testing of the Cython API

class TestPoint(unittest.TestCase):
    
    def test_initialise(self):
        
        # default initialisation, point at local origin
        v = Point()
        self.assertEqual(v.x, 0.0, "Default initialisation is not (0,0,0) [X].")
        self.assertEqual(v.y, 0.0, "Default initialisation is not (0,0,0) [Y].")
        self.assertEqual(v.z, 0.0, "Default initialisation is not (0,0,0) [Z].")

        # initialisation with an indexable
        v = Point([1.0, 2.0, 3.0])
        self.assertEqual(v.x, 1.0, "Initialisation with indexable failed [X].")
        self.assertEqual(v.y, 2.0, "Initialisation with indexable failed [Y].")
        self.assertEqual(v.z, 3.0, "Initialisation with indexable failed [Z].")
        
        # invalid initialisation
        with self.assertRaises(TypeError, msg="Initialised with a string."):
            Point("spoon")
        
        with self.assertRaises(TypeError, msg="Initialised with a list containing too few items."):
            Point([1.0, 2.0])
            
    def test_x(self):
        
        v = Point([2.5, 6.7, -4.6])
        
        # get x attribute
        self.assertEqual(v.x, 2.5, "Getting x attribute failed.")
        
        # set x attribute
        v.x = 10.0
        self.assertEqual(v.x, 10.0, "Setting x attribute failed.")
    
    def test_y(self):
        
        v = Point([2.5, 6.7, -4.6])
        
        # get y attribute
        self.assertEqual(v.y, 6.7, "Getting y attribute failed.")
        
        # set y attribute
        v.y = -7.1
        self.assertEqual(v.y, -7.1, "Setting y attribute failed.")

    def test_z(self):
        
        v = Point([2.5, 6.7, -4.6])
        
        # get z attribute
        self.assertEqual(v.z, -4.6, "Getting z attribute failed.")
        
        # set z attribute
        v.z = 157.3
        self.assertEqual(v.z, 157.3, "Setting z attribute failed.")

    def test_indexing(self):
        
        v = Point([2.5, 6.7, -4.6])
        
        # check valid indexes
        self.assertEqual(v[0], 2.50, "Indexing failed [X].")
        self.assertEqual(v[1], 6.70, "Indexing failed [Y].")
        self.assertEqual(v[2], -4.6, "Indexing failed [Z].")
        
        # check invalid indexes
        with self.assertRaises(IndexError, msg="Invalid positive index did not raise IndexError."):
            
            r = v[4]

        with self.assertRaises(IndexError, msg="Invalid negative index did not raise IndexError."):
            
            r = v[-1]            
            
    def test_add(self):
        
        # adding points is undefined
        with self.assertRaises(TypeError, msg="Point addition did not raise a TypeError."):

            Point() + Point()

    def test_subtract(self):
        
        # subtracting points is undefined
        with self.assertRaises(TypeError, msg="Point subtraction did not raise a TypeError."):

            Point() - Point()

    def test_distance_to(self):
        
        a = Point([-1, 5, 26])
        b = Point([9, 4, -1])
        v = a.distance_to(b)
        r = sqrt((9 + 1)**2 + (4 - 5)**2 + (-1 - 26)**2)
        self.assertEqual(v, r, "Point to Point distance is incorrect.")
    
    def test_vector_to(self):
        
        a = Point([-1, 5, 26])
        b = Point([9, 4, -1])
        v = a.vector_to(b)
        self.assertTrue(isinstance(v, Vector), "Vector_to did not return a Vector.")
        self.assertEqual(v.x, 9 + 1, "Vector_to failed [X].")
        self.assertEqual(v.y, 4 - 5, "Vector_to failed [Y].")
        self.assertEqual(v.z, -1 - 26, "Vector_to failed [Z].")
        
    #def test_transform(self):
        
        ## TODO: add test
        #pass        
