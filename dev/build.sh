#!/bin/bash

echo "Rebuilding Raysect extension modules (in place)..."
python setup.py build_ext --inplace $1
