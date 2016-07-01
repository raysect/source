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

"""
The data used to define the the following metal materials was sourced from http://refractiveindex.info.

This data is licensed as public domain (CC0 1.0 - https://creativecommons.org/publicdomain/zero/1.0/).
"""

import sys
from os import path
import json
from numpy import array
from raysect.optical import InterpolatedSF
from raysect.optical.material import Conductor


# a list of classes to dynamically create in the module and the files in which they will find their data
_MATERIALS = [
    ('Silver', 'silver'),
    ('Aluminium', 'aluminium'),
    ('Gold', 'gold'),
    ('Beryllium', 'beryllium'),
    ('Cobolt', 'cobolt'),
    ('Copper', 'copper'),
    ('Iron', 'iron'),
    ('Mercury', 'mercury'),
    ('Lithium', 'lithium'),
    ('Magnesium', 'magnesium'),
    ('Manganese', 'manganese'),
    ('Sodium', 'sodium'),
    ('Nickel', 'nickel'),
    ('Palladium', 'palladium'),
    ('Platinum', 'platinum'),
    ('Silicon', 'silicon'),
    ('Titanium', 'titanium'),
    ('Tungsten', 'tungsten'),
]


def init_factory(filename):

    def init(self):
        with open(path.join(path.dirname(__file__), "data", filename + ".json")) as f:
            data = json.load(f)

        wavelength = array(data['wavelength'])
        index = InterpolatedSF(wavelength, array(data['index']))
        extinction = InterpolatedSF(wavelength, array(data['extinction']))

        Conductor.__init__(self, index, extinction)

    return init


for clsname, filename in _MATERIALS:
    cls = type(clsname, (Conductor, ), {"__init__": init_factory(filename)})
    setattr(sys.modules[__name__], clsname, cls)
