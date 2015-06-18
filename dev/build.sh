#!/bin/bash

echo "Rebuilding Raysect extension modules (in place)..."
python setup.py build_ext --inplace $1 $2 $3 $4 $5
