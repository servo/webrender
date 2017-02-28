#!/usr/bin/python

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import contextlib
import os
import subprocess
import sys
from os import path
from glob import glob


@contextlib.contextmanager
def cd(new_path):
    """Context manager for changing the current working directory"""
    previous_path = os.getcwd()
    try:
        os.chdir(new_path)
        yield
    finally:
        os.chdir(previous_path)


def find_dep_path_newest(package, bin_path):
    deps_path = path.join(path.split(bin_path)[0], "build")
    with cd(deps_path):
        candidates = glob(package + '-*')
    candidates = (path.join(deps_path, c) for c in candidates)
    candidate_times = sorted(((path.getmtime(c), c) for c in candidates), reverse=True)
    if len(candidate_times) > 0:
        return candidate_times[0][1]
    return None


def is_windows():
    """ Detect windows, mingw, cygwin """
    return sys.platform == 'win32' or sys.platform == 'msys' or sys.platform == 'cygwin'


def is_macosx():
    return sys.platform == 'darwin'


def is_linux():
    return sys.platform.startswith('linux')


def set_osmesa_env(bin_path):
    """Set proper LD_LIBRARY_PATH and DRIVE for software rendering on Linux and OSX"""
    if is_linux():
        osmesa_path = path.join(find_dep_path_newest('osmesa-src', bin_path), "out", "lib", "gallium")
        print(osmesa_path)
        os.environ["LD_LIBRARY_PATH"] = osmesa_path
        os.environ["GALLIUM_DRIVER"] = "softpipe"
    elif is_macosx():
        osmesa_path = path.join(find_dep_path_newest('osmesa-src', bin_path),
                                "out", "src", "gallium", "targets", "osmesa", ".libs")
        glapi_path = path.join(find_dep_path_newest('osmesa-src', bin_path),
                               "out", "src", "mapi", "shared-glapi", ".libs")
        os.environ["DYLD_LIBRARY_PATH"] = osmesa_path + ":" + glapi_path
        os.environ["GALLIUM_DRIVER"] = "softpipe"


subprocess.check_call(['cargo', 'build', '--release', '--features', 'headless'])
set_osmesa_env('../target/release/')
subprocess.check_call(['../target/release/wrench', '-h'] + sys.argv[1:])
