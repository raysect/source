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

"""
The data used to define the the following metal materials was sourced from http://refractiveindex.info.

This data is licensed as public domain (CC0 1.0 - https://creativecommons.org/publicdomain/zero/1.0/).
"""

from os import path
import json
from numpy import array
from raysect.optical import InterpolatedSF
from raysect.optical.material import RoughConductor


class _DataLoader(RoughConductor):

    def __init__(self, filename, roughness):

        with open(path.join(path.dirname(__file__), "data", filename + ".json")) as f:
            data = json.load(f)

        wavelength = array(data['wavelength'])
        index = InterpolatedSF(wavelength, array(data['index']))
        extinction = InterpolatedSF(wavelength, array(data['extinction']))

        super().__init__(index, extinction, roughness)


class RoughAluminium(_DataLoader):
    """Aluminium metal material."""
    def __init__(self, roughness):
        super().__init__("aluminium", roughness)


class RoughBeryllium(_DataLoader):
    """Beryllium metal material."""
    def __init__(self, roughness):
        super().__init__("beryllium", roughness)


class RoughCobolt(_DataLoader):
    """Cobolt metal material."""
    def __init__(self, roughness):
        super().__init__("cobolt", roughness)


class RoughCopper(_DataLoader):
    """Copper metal material."""
    def __init__(self, roughness):
        super().__init__("copper", roughness)


class RoughGold(_DataLoader):
    """Gold metal material."""
    def __init__(self, roughness):
        super().__init__("gold", roughness)


class RoughIron(_DataLoader):
    """Iron metal material."""
    def __init__(self, roughness):
        super().__init__("iron", roughness)


class RoughLithium(_DataLoader):
    """Lithium metal material."""
    def __init__(self, roughness):
        super().__init__("lithium", roughness)


class RoughMagnesium(_DataLoader):
    """Magnesium metal material."""
    def __init__(self, roughness):
        super().__init__("magnesium", roughness)


class RoughManganese(_DataLoader):
    """Manganese metal material."""
    def __init__(self, roughness):
        super().__init__("manganese", roughness)


class RoughMercury(_DataLoader):
    """Mercury metal material."""
    def __init__(self, roughness):
        super().__init__("mercury", roughness)


class RoughNickel(_DataLoader):
    """Nickel metal material."""
    def __init__(self, roughness):
        super().__init__("nickel", roughness)


class RoughPalladium(_DataLoader):
    """Palladium metal material."""
    def __init__(self, roughness):
        super().__init__("palladium", roughness)


class RoughPlatinum(_DataLoader):
    """Platinum metal material."""
    def __init__(self, roughness):
        super().__init__("platinum", roughness)


class RoughSilicon(_DataLoader):
    """Silicon metal material."""
    def __init__(self, roughness):
        super().__init__("silicon", roughness)


class RoughSilver(_DataLoader):
    """Silver metal material."""
    def __init__(self, roughness):
        super().__init__("silver", roughness)


class RoughSodium(_DataLoader):
    """Sodium metal material."""
    def __init__(self, roughness):
        super().__init__("sodium", roughness)


class RoughTitanium(_DataLoader):
    """Titanium metal material."""
    def __init__(self, roughness):
        super().__init__("titanium", roughness)


class RoughTungsten(_DataLoader):
    """Tungsten metal material."""
    def __init__(self, roughness):
        super().__init__("tungsten", roughness)
