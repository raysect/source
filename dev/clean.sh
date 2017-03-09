#!/bin/bash

echo Removing all .c and .so files...
find . -type f -name '*.c' -exec rm {} +
find . -type f -name '*.so' -exec rm {} +

