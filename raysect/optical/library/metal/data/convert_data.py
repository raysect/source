"""
Conversion script for refractiveindex.info data.

Converts refractiveindex.info data stored in yaml files into JSON format for the Raysect library.
This script must be run from the root of a copy of the refractiveindex.info database.

Only a subset of the database is exported, as required for Raysect.

* RefractiveIndex.INFO website: © 2008-2016 Mikhail Polyanskiy
* refractiveindex.info database: public domain via CC0 1.0
* NO GUARANTEE OF ACCURACY - Use on your own risk
"""

import yaml
import json


def parse_tabulated_nk(d):

    # check type
    dtype = d['type']
    if dtype != 'tabulated nk':
        raise ValueError('File does not contain tabulated nk data.')

    table = d['data']

    # convert string table to values (wavelength in nm)
    wavelength = []
    index_re = []
    index_im = []
    lines = table.strip().split('\n')
    for line in lines:
        wl, n, k = tuple(line.strip().split(' '))
        wavelength.append(float(wl) * 1000)
        index_re.append(float(n))
        index_im.append(float(k))

    return wavelength, index_re, index_im


def yaml_to_json(input_filename, output_filename):

    with open(input_filename) as f:
        d = yaml.load(f)

    wl, ir, ii = parse_tabulated_nk(d['DATA'][0])

    content = {
        'reference': d['REFERENCES'],
        'wavelength': wl,
        "index": ir,
        "extinction": ii
    }

    with open(output_filename, 'w') as f:
        json.dump(content, f, indent=4)


materials = [
    ('database/main/Ag/Rakic', 'silver'),
    ('database/main/Al/Rakic', 'aluminium'),
    ('database/main/Au/Rakic', 'gold'),
    ('database/main/Be/Rakic', 'beryllium'),
    # ('database/main/C/Hagemann', 'carbon'),
    ('database/main/Co/Johnson', 'cobolt'),
    ('database/main/Cu/Rakic', 'copper'),
    ('database/main/Fe/Johnson', 'iron'),
    ('database/main/Hg/Inagaki', 'mercury'),
    ('database/main/Li/Rasigni', 'lithium'),
    ('database/main/Mg/Hagemann', 'magnesium'),
    ('database/main/Mn/Johnson', 'manganese'),
    ('database/main/Na/Inagaki', 'sodium'),
    ('database/main/Ni/Rakic', 'nickel'),
    ('database/main/Pd/Rakic', 'palladium'),
    ('database/main/Pt/Rakic', 'platinum'),
    ('database/main/Si/Jellison', 'silicon'),
    ('database/main/Ti/Rakic', 'titanium'),
    ('database/main/W/Rakic', 'tungsten'),
]

# convert data
for input, output in materials:
    yaml_to_json(input + ".yml", output + ".json")
