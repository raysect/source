from os import path
import csv
from collections import namedtuple
from numpy import array
from raysect.optical.material import Dielectric, Sellmeier
from raysect.optical.spectralfunction import InterpolatedSF

_sellmeier_disp = namedtuple("sellmeier_dispersion", ["B1", "B2", "B3", "C1", "C2", "C3"])

_taui25 = namedtuple("transmission_25mm", ["wavelength", "transmission"])

# wavelengths measured for Schott glass data
_wavelengths = array([2.500, 2.325, 1.970, 1.530, 1.060, 0.700, 0.660, 0.620, 0.580, 0.546, 0.500, 0.460, 0.436, 0.420,
                      0.405, 0.400, 0.390, 0.380, 0.370, 0.365, 0.350, 0.334, 0.320, 0.310, 0.300, 0.290, 0.280, 0.270,
                      0.260, 0.250]) * 1000

_glass_data = namedtuple("glass_data", ["name", "sellmeier", "taui25"])


# TODO: tidy up + add docstrings

class Schott():

    def __init__(self):

        try:
            data_path = path.join(path.dirname(__file__), "data/schott_catalog_2000.csv")
            schott_file = open(data_path, "r")
        except FileNotFoundError:
            raise ValueError('Schott Glass catalog file could not be found.')

        self._schott_glass_data = {}
        header = schott_file.readline()
        reader = csv.reader(schott_file, quoting=csv.QUOTE_NONNUMERIC, quotechar='"')

        for row in reader:

            # extract raw csv data into appropriate variables
            glass_name = row[0]
            sellmeier = _sellmeier_disp(*row[1:7])
            raw_trans_data = row[7:37]

            # process the transmission data (t^(1/0.025)) and remove any zero elements,
            trans_array = list(zip(_wavelengths, raw_trans_data))
            trans_data = array([(data[0], data[1]**40) for data in trans_array if data[1]]).T
            transmission_25 = _taui25(*trans_data)

            self._schott_glass_data[glass_name] = _glass_data(glass_name, sellmeier, transmission_25)

    def __call__(self, glass_name):

        try:
            chosen_glass = self._schott_glass_data[glass_name]
        except KeyError:
            raise ValueError('This glass could not be found in the available Schott catalog.')

        b1, b2, b3, c1, c2, c3 = chosen_glass.sellmeier
        wavelengths, transmission = chosen_glass.taui25
        return Dielectric(index=Sellmeier(b1, b2, b3, c1, c2, c3), transmission=InterpolatedSF(wavelengths, transmission))

    def list(self):
        return self._schott_glass_data.keys()


schott = Schott()
