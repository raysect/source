#!/bin/bash

echo Building code...

dev/build.sh

echo Building docs...

export PYTHONPATH=../:$PYTHONPATH

cd docs

make html
