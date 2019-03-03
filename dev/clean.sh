#!/bin/bash

echo Removing all .c, .so and .html files...

find raysect -type f -name '*.c' -exec rm {} +
find raysect -type f -name '*.so' -exec rm {} +
find raysect -type f -name '*.html' -exec rm {} +
rm build -rf
