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
Unit tests for the Vector object.
"""

from ..vector import Vector
import unittest

#things a vector needs to do
# initialised
# coords set/get - including iterate
# add
# subtract
# multiply with scalar (change length)
# dot product
# cross product
# normalise

# transform - apply an affine matrix transform to self (return new object)


class TestVector(unittest.TestCase):
    
    def test_initialise(self):

        # default initialisation, unit vector pointing along x-axis
        v = Vector()
        self.assertEqual(v.x, 1.0, "Default initialisation is not (1,0,0) [X].")
        self.assertEqual(v.y, 0.0, "Default initialisation is not (1,0,0) [Y].")
        self.assertEqual(v.z, 0.0, "Default initialisation is not (1,0,0) [Z].")

        # initialisation with an iterable
        v = Vector([1.0, 2.0, 3.0])
        self.assertEqual(v.x, 1.0, "Initialisation with iterable failed [X].")
        self.assertEqual(v.y, 2.0, "Initialisation with iterable failed [Y].")
        self.assertEqual(v.z, 3.0, "Initialisation with iterable failed [Z].")
        
        # invalid initialisation
        with self.assertRaises(TypeError, msg="Initialised with a string."):
            Vector("spoon")
        
        with self.assertRaises(TypeError, msg="Initialised with a list containing too few items."):
            Vector([1.0, 2.0])
        
        # TODO: add special test for initialisation with a point and a normal

    def test_x(self):
        
        v = Vector([2.5, 6.7, -4.6])
        
        # get x attribute
        self.assertEqual(v.x, 2.5, "Getting x attribute failed.")
        
        # set x attribute
        v.x = 10.0
        self.assertEqual(v.x, 10.0, "Setting x attribute failed.")
    
    def test_y(self):
        
        v = Vector([2.5, 6.7, -4.6])
        
        # get y attribute
        self.assertEqual(v.y, 6.7, "Getting y attribute failed.")
        
        # set y attribute
        v.y = -7.1
        self.assertEqual(v.y, -7.1, "Setting y attribute failed.")

    def test_z(self):
        
        v = Vector([2.5, 6.7, -4.6])
        
        # get z attribute
        self.assertEqual(v.z, -4.6, "Getting z attribute failed.")
        
        # set z attribute
        v.z = 157.3
        self.assertEqual(v.z, 157.3, "Setting z attribute failed.")

    def test_indexing(self):
        
        v = Vector([2.5, 6.7, -4.6])
        
        # check valid indexes
        self.assertEqual(v[0], 2.5, "Indexing failed [X].")
        self.assertEqual(v[1], 6.7, "Indexing failed [Y].")
        self.assertEqual(v[2], -4.6, "Indexing failed [Z].")
        
        # check invalid indexes
        with self.assertRaises(IndexError, msg="Invalid positive index did not raise IndexError."):
            
            r = v[4]

        with self.assertRaises(IndexError, msg="Invalid negative index did not raise IndexError."):
            
            r = v[-1]            

    def test_length(self):
        
        self.assertTrue(False)
        
    def test_add(self):
        
        self.assertTrue(False)
        
    def test_subtract(self):
        
        self.assertTrue(False)
    
    def test_multiply(self):
        
        self.assertTrue(False)
    
    def test_divide(self):
        
        self.assertTrue(False)
    
    def test_negate(self):
        
        r = -Vector([2.5, 6.7, -4.6])
        self.assertEqual(r.x, -2.5, "Negation failed [X].")
        self.assertEqual(r.y, -6.7, "Negation failed [Y].")
        self.assertEqual(r.z, 4.6, "Negation failed [Z].")
    
    def test_normalise(self):
        
        self.assertTrue(False)

    def test_dot_product(self):
        
        self.assertTrue(False)
    
    def test_cross_product(self):
        
        self.assertTrue(False)

    
if __name__ == "__main__":
    unittest.main()
    
    
    
    