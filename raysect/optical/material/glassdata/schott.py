
import csv
from collections import namedtuple

optical_properties = namedtuple("optical_properties", ["nd", "ne", "vd", "ve"])
sellmeier_disp = namedtuple("sellmeier_dispersion", ["B1", "B2", "B3", "C1", "C2", "C3"])
taui25 = namedtuple("transmission_25mm",
                    ["wv2500", "wv2325", "wv1970", "wv1530", "wv1060", "wv700", "wv660", "wv620", "wv580", "wv546",
                     "wv500", "wv460", "wv436", "wv420", "wv405", "wv400", "wv390", "wv380", "wv370", "wv365", "wv350",
                     "wv334", "wv320", "wv310", "wv300", "wv290", "wv280", "wv270", "wv260", "wv250"])
taui10 = namedtuple("transmission_10mm",
                    ["wv2500", "wv2325", "wv1970", "wv1530", "wv1060", "wv700", "wv660", "wv620", "wv580", "wv546",
                     "wv500", "wv460", "wv436", "wv420", "wv405", "wv400", "wv390", "wv380", "wv370", "wv365", "wv350",
                     "wv334", "wv320", "wv310", "wv300", "wv290", "wv280", "wv270", "wv260", "wv250"])
chemical_properties = namedtuple("chemical_properties", ["CR", "FR", "SR", "AR", "PR"])
thermal_properties = namedtuple("thermal_properties", ["Tg", "T13", "T7_6", "heat_capacity", "heat_conductivity",
                                                       "alpha_30_70", "alpha_20_300"])
mechanical_properties = namedtuple("mechanical_properties", ["Youngs_modulus", "Poisson_ratio", "Knoop_hardness",
                                                             "abrasion_hardness"])

glass_data = namedtuple("glass_data",
                        ["name", "optical", "sellmeier", "taui25", "taui10", "chemical", "thermal", "mechanical"])


def load_schott():
    try:
        schott_file = open('schott_catalog_2000.csv', 'r')
    except FileNotFoundError:
        raise ValueError('Schott Glass catalog file could not be found.')

    schott_glass_data = {}

    header = schott_file.readline()
    reader = csv.reader(schott_file, quoting=csv.QUOTE_NONNUMERIC, quotechar='"')

    for row in reader:
        glass_name = row[0]
        optical = optical_properties(*row[1:5])
        sellmeir = sellmeier_disp(*row[5:11])
        transmission_25 = taui25(*row[39:69])
        transmission_10 = taui10(*row[69:99])
        chemical = chemical_properties(*row[100:105])
        thermal = thermal_properties(*row[106:113])
        mechanical = mechanical_properties(*row[113:])
        schott_glass_data[glass_name] = glass_data(glass_name, optical, sellmeir, transmission_25, transmission_10,
                                                    chemical, thermal, mechanical)

    return schott_glass_data


schott_glasses = load_schott()


def get_schott(glass_name):
    try:
        return schott_glasses[glass_name]
    except KeyError:
        raise ValueError('Glass could not be found in Schott catalog.')