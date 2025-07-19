#!/bin/env python

from pathlib import Path

PACKAGES = ['raysect']
SUBDIR_EXCLUSION_FILENAME = '.meson-exclude'


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

    :param packages: A list of package names.
    """

    root_path = Path('.')
    package_paths = [Path(package) for package in packages]

    # Walk the project folder and specified packages to remove all meson.build files.
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

    # remove meson.build in this directory
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

    meson_file = path / 'meson.build'
    meson_file.write_text(_generate_root_meson_file(subdirs))


def _install_subdir_meson_files(path):

    # validate
    if not path.is_dir():
        raise ValueError('The supplied path is not a directory.')

    # generate a list of subdirectories, filtering excluded
    # todo: filter pycache files etc..
    subdirs = [child for child in path.iterdir() if child.is_dir() and not (child / SUBDIR_EXCLUSION_FILENAME).exists()]

    # write meson file
    meson_file = path / 'meson.build'
    meson_file.write_text(_generate_subdir_meson_file(path, subdirs))

    # recurse into sub-directories
    for subdir in subdirs:
        _install_subdir_meson_files(subdir)


def _generate_root_meson_file(subdirs):

    # read template
    template_path = Path(__file__).parent / 'root-meson.build'
    contents = template_path.read_text()

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

    # fill in template entries
    contents = template.format(
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

    # get user confirmation


    generate_meson_files(PACKAGES)