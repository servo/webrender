# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Locate or download a RenderDoc distribution to a known location.

RenderDoc is used by the WebRender debugger (wrshell) to capture frames from a
running Firefox/wrench instance. This module finds an existing RenderDoc
(`librenderdoc.so` for `LD_PRELOAD`, and `qrenderdoc` to open captures), and can
download/unpack the official Linux build into a cache directory if missing.

It is importable from `mach wrshell` and can also be run standalone:

    python3 gfx/wr/wrshell/renderdoc_bootstrap.py [--version 1.44] [--cache-dir DIR]
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path

# Pinned default; override with --version or the WR_RENDERDOC_VERSION env var.
DEFAULT_VERSION = "1.44"


def default_cache_dir():
    """Directory that holds downloaded RenderDoc builds (~/.mozbuild/renderdoc)."""
    base = os.environ.get("MOZBUILD_STATE_PATH") or os.path.join(
        os.path.expanduser("~"), ".mozbuild"
    )
    return Path(base) / "renderdoc"


def _paths_for(install_dir):
    """Return (librenderdoc.so, qrenderdoc) under an extracted RenderDoc dir."""
    lib = install_dir / "lib" / "librenderdoc.so"
    qrd = install_dir / "bin" / "qrenderdoc"
    return lib, qrd


def find(version=DEFAULT_VERSION, cache_dir=None):
    """Locate an existing RenderDoc. Returns (lib_path, bin_dir) or None.

    Search order:
      1. $WR_RENDERDOC_DIR (an extracted RenderDoc root).
      2. The cache directory (default ~/.mozbuild/renderdoc/renderdoc_<version>).
      3. A system install on $PATH (qrenderdoc) / ldconfig (librenderdoc.so).
    """
    explicit = os.environ.get("WR_RENDERDOC_DIR")
    if explicit:
        lib, qrd = _paths_for(Path(explicit))
        if lib.exists():
            return lib, qrd.parent

    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
    lib, qrd = _paths_for(cache_dir / f"renderdoc_{version}")
    if lib.exists():
        return lib, qrd.parent

    # System install: librenderdoc.so on the loader path, qrenderdoc on PATH.
    for d in os.environ.get("PATH", "").split(os.pathsep):
        cand = Path(d) / "qrenderdoc"
        if cand.exists():
            for libdir in ("/usr/lib64", "/usr/lib", "/usr/local/lib"):
                syslib = Path(libdir) / "librenderdoc.so"
                if syslib.exists():
                    return syslib, cand.parent
    return None


def ensure(version=DEFAULT_VERSION, cache_dir=None, log=print):
    """Find RenderDoc, downloading + unpacking it into the cache if missing.

    Returns (lib_path, bin_dir).
    """
    found = find(version, cache_dir)
    if found:
        log(f"RenderDoc found at {found[0]}")
        return found

    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    install_dir = cache_dir / f"renderdoc_{version}"
    url = f"https://renderdoc.org/stable/{version}/renderdoc_{version}.tar.gz"
    tarball = cache_dir / f"renderdoc_{version}.tar.gz"

    log(f"Downloading RenderDoc {version} from {url}")
    urllib.request.urlretrieve(url, tarball)
    log(f"Extracting to {cache_dir}")
    with tarfile.open(tarball) as tar:
        tar.extractall(cache_dir)
    tarball.unlink()

    lib, qrd = _paths_for(install_dir)
    if not lib.exists():
        raise RuntimeError(f"librenderdoc.so not found after extracting {url}")
    log(f"RenderDoc ready at {lib}")
    return lib, qrd.parent


def _main(argv):
    import argparse

    parser = argparse.ArgumentParser(description="Locate or download RenderDoc")
    parser.add_argument(
        "--version",
        default=os.environ.get("WR_RENDERDOC_VERSION", DEFAULT_VERSION),
    )
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args(argv)

    lib, bindir = ensure(args.version, args.cache_dir)
    print(f"librenderdoc: {lib}")
    print(f"bin dir:      {bindir}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
