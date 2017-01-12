#!/usr/bin/python

import contextlib
import os
import subprocess
import sys
import hashlib
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


set_osmesa_env('../target/release/')
subprocess.check_call(['../target/release/wrench', '-t', '1', '-h', 'show', sys.argv[1]])
subprocess.check_call(['../target/release/wrench', '-h', 'reftest'])
print('md5 = ' + hashlib.md5(open('screenshot.png', 'rb').read()).hexdigest())
