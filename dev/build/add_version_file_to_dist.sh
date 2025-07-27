#!/bin/bash
set -e  # exit if an error occurs

if [[ -z "$MESON_SOURCE_ROOT" ]]; then
    echo "ERROR: Must be run inside a meson build environment where the variable MESON_SOURCE_ROOT is set." 1>&2
    exit 1
fi

if [[ -z "$MESON_DIST_ROOT" ]]; then
    echo "ERROR: Must be run inside a meson build environment where the variable MESON_DIST_ROOT is set." 1>&2
    exit 1
fi

VERSION_PATH="raysect/_version.py"
SRC_PATH="$MESON_SOURCE_ROOT/$VERSION_PATH"
DIST_PATH="$MESON_DIST_ROOT/$VERSION_PATH"

echo "Copying $VERSION_PATH to distribution path $DIST_PATH"
cp $SRC_PATH $DIST_PATH
