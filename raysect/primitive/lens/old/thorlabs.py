import csv
from collections import namedtuple

from core import AffineMatrix
from optical.material.glass_libraries import Schott
from primitive.lens.old.sphericalsinglets import generic_spherical_singlet

# infinity constant
INFINITY = 1e999

# load default_glass
schott = Schott()
NBK7 = schott("N-BK7")

_plano_convex = namedtuple("plano_convex", ["diameter", "focallength", "radius", "center_thickness",
                                            "edge_thickness", "back_focallength"])


class Thorlabs():

    # TODO - need to separate out different types of lenses.
    def __init__(self):
        try:
            plano_convex_file = open('raysect/optical/optical_components/thorlabs_data/plano-convex.csv', 'r')
        except FileNotFoundError:
            raise ValueError('Lens catalog file could not be found.')

        self._plano_convex_data = {}

        header = plano_convex_file.readline()
        reader = csv.reader(plano_convex_file, quoting=csv.QUOTE_NONNUMERIC, quotechar='"')

        for row in reader:
            self._plano_convex_data[row[0]] = _plano_convex(*row[1:7])

    def __call__(self, lens_code, parent=None, transform=AffineMatrix):
        try:
            chosen_lens = self._plano_convex_data[lens_code]
        except KeyError:
            raise ValueError('This lens component could not be found in the available Thorlabs catalog.')

        diameter = chosen_lens[0]
        r2 = chosen_lens[2]
        thickness = chosen_lens[3]

        return generic_spherical_singlet(INFINITY, r2, diameter, thickness, material=NBK7,
                                         parent=parent, transform=transform)

    def list(self):

        return self._plano_convex_data.keys()