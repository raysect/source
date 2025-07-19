#!/bin/env python

# Copyright (c) 2014-2025, Dr Alex Meakins, Raysect Project
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the Raysect Project nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from pathlib import Path

PACKAGES = ['raysect']

EXCLUDE_DIR_FILE = '.meson-exclude'
CUSTOM_MESON_BUILD = '.meson-custom'
EXCLUDED_DIRS = ['__pycache__']


def generate_meson_files(packages):
    """
    Generates the meson.build files for the project from a set of template files.

    A root build.meson will be placed in the project root and meson.build files will be generated for the specified
    package folders. This script must be executed from the root folder of the project.

    If substantive changes are needed to the meson.build files throughout the project, it will be easier to modify the
    templates and trigger the regeneration process.

    This script will remove any existing meson.build files, so be sure any changes are captured before re-running this
    script. There are two template files:

        * root-meson.build
        * subdir-meson.build

    The root-meson.build file is used to generate the root meson.build for the project. The subdir-meson.build is used
    to generate the meson.build files in the project sub-directory. The templates are read and handled as a python
    f-strings. See the script implementation for the variables available to the template. The meson.build files will
    consist of the template followed by a set of subdir() entries for each descendant of the current sub-directory
    (if not excluded).

    A sub-directory may be excluded from the generation process by placing a file in the subfolder called
    ".meson-exclude". If the exclusion file is found, the sub-directory and its descendants will be ignored during
    the generation process.

    Occasionally a meson.build file may need to be customised in the source tree. Placing a file called ".meson-custom"
    in the same directory as the meson.build file will protect the customised file from deletion or replacement by this
    script.

    :param packages: A list of package names.
    """

    root_path = Path('.')
    package_paths = [Path(package) for package in packages]

    # Walk the project folder and specified packages to remove all (non-custom) meson.build files.
    # Any stale meson.build files found in excluded directories are also removed.
    _remove_meson_files(root_path, subdirs=package_paths)

    # Add root meson.build file.
    _install_root_meson_file(root_path, subdirs=package_paths)

    # Walk the specified packages and add the sub-directory meson.build files.
    for path in package_paths:
        _install_subdir_meson_files(path)


def _remove_meson_files(path, subdirs=None):
    """
    Removes any meson.build files found under the directory tree referenced by path.

    By default, this function recurses through the entire directory tree under the supplied path. If sub-dirs is
    provided, then only the specified sub-directories will be explored.
    """

    # validate
    if not path.is_dir():
        raise ValueError('The supplied path is not a directory.')

    if subdirs and any([not subdir.is_dir() for subdir in subdirs]):
        raise ValueError('The list of sub-directories must only contain paths to valid directories.')

    # remove meson.build in this directory if it is not flagged as customised
    if not _custom_meson_file(path):
        meson_file = path / 'meson.build'
        meson_file.unlink(missing_ok=True)

    # generate a list of subdirectories if none supplied
    if not subdirs:
        subdirs = [child for child in path.iterdir() if child.is_dir()]

    # recurse into sub-directories
    for subdir in subdirs:
        _remove_meson_files(subdir)


def _install_root_meson_file(path, subdirs):

    # validate
    if not path.is_dir():
        raise ValueError('The supplied path is not a directory.')

    if any([not subdir.is_dir() for subdir in subdirs]):
        raise ValueError('The list of sub-directories must only contain paths to valid directories.')

    # write meson file
    _write_meson_file(path, _generate_root_meson_file(subdirs))


def _install_subdir_meson_files(path):

    # validate
    if not path.is_dir():
        raise ValueError('The supplied path is not a directory.')

    # generate a list of subdirectories, filtering excluded
    subdirs = [child for child in path.iterdir() if child.is_dir() and not _excluded_dir(child)]

    # write meson file
    _write_meson_file(path, _generate_subdir_meson_file(path, subdirs))

    # recurse into sub-directories
    for subdir in subdirs:
        _install_subdir_meson_files(subdir)


def _write_meson_file(path, contents):
    if not _custom_meson_file(path):
        meson_file = path / 'meson.build'
        meson_file.write_text(contents)


def _custom_meson_file(path):
    return (path / CUSTOM_MESON_BUILD).exists()


def _excluded_dir(path):
    foo = (path / EXCLUDE_DIR_FILE).exists() or path.name in EXCLUDED_DIRS
    return foo


def _generate_root_meson_file(subdirs):

    # read template
    template_path = Path(__file__).parent / 'root-meson.build'
    template = template_path.read_text()

    # start contents with a warning
    contents = (
        "# WARNING: This file is automatically generated by dev/generate_meson_files.py.\n"
        "# The template file used to generate this file is dev/root-meson.build.\n\n"
    )

    # add template
    contents += template

    # add subdir entries
    contents += '\n'
    for subdir in subdirs:
        contents += f'subdir(\'{subdir.name}\')\n'

    return contents


def _generate_subdir_meson_file(path, subdirs):

    # read template
    template_path = Path(__file__).parent / 'subdir-meson.build'
    template = template_path.read_text()

    # build file lists
    pyx = []
    pxd = []
    py = []
    data = []
    for child in path.iterdir():

        if child.is_dir():
            continue

        elif child.suffix == '.pyx':
            pyx.append(child.name)

        elif child.suffix == '.pxd':
            pxd.append(child.name)

        elif child.suffix == '.py':
            py.append(child.name)

        else:
            data.append(child.name)

    # start contents with a warning
    contents = (
        "# WARNING: This file is automatically generated by dev/generate_meson_files.py.\n"
        "# The template file used to generate this file is dev/subdir-meson.build.\n\n"
    )

    # add template, filling in the variables
    contents += template.format(
        target=f'\'{str(path)}\'',
        pyx_files=str(pyx),
        pxd_files=str(pxd),
        py_files=str(py),
        data_files=str(data)
    )

    # add subdir entries
    contents += '\n'
    for subdir in subdirs:
        contents += f'subdir(\'{subdir.name}\')\n'

    return contents


if __name__ == '__main__':
    generate_meson_files(PACKAGES)