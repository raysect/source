#!/bin/bash
set -e  # exit if an error occurs

echo Installing package as editable...
python -m pip install --config-settings=editable-verbose=true --no-build-isolation --upgrade --editable .