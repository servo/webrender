# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import importlib.util
import os
import subprocess
import tempfile
from pathlib import Path

from mach.decorators import Command, CommandArgument


def _load_renderdoc_bootstrap(topsrcdir):
    """Import the WebRender RenderDoc bootstrap helper by path."""
    path = os.path.join(
        topsrcdir, "gfx", "wr", "wrench", "script", "renderdoc_bootstrap.py"
    )
    spec = importlib.util.spec_from_file_location("renderdoc_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@Command(
    "wrshell",
    category="devenv",
    description="Launch Firefox under the WebRender debugger (wrshell) with "
    "RenderDoc capture support.",
)
@CommandArgument(
    "url",
    nargs="?",
    default="about:blank",
    help="URL to open in the launched Firefox.",
)
@CommandArgument(
    "--gui",
    action="store_true",
    help="Run wrshell in GUI mode (default is the CLI repl).",
)
@CommandArgument(
    "--renderdoc-version",
    default=None,
    help="RenderDoc version to download/use (default: pinned version).",
)
@CommandArgument(
    "--no-firefox",
    action="store_true",
    help="Don't launch Firefox; connect wrshell to an already-running instance.",
)
def wrshell(command_context, url, gui, renderdoc_version, no_firefox):
    topsrcdir = command_context.topsrcdir
    topobjdir = command_context.topobjdir
    bootstrap = _load_renderdoc_bootstrap(topsrcdir)

    def log(msg):
        command_context.log(20, "wrshell", {}, msg)

    # Ensure RenderDoc is available (download into ~/.mozbuild/renderdoc if not).
    version = renderdoc_version or bootstrap.DEFAULT_VERSION
    lib_path, bin_dir = bootstrap.ensure(version, log=log)
    renderdoc_root = str(Path(bin_dir).parent)

    firefox_proc = None
    profile_dir = None
    if not no_firefox:
        firefox = Path(topobjdir) / "dist" / "bin" / "firefox"
        if not firefox.exists():
            command_context.log(
                40, "wrshell", {}, f"Firefox binary not found at {firefox}; run `./mach build` first."
            )
            return 1

        captures_dir = Path(topobjdir) / "tmp" / "renderdoc-captures"
        captures_dir.mkdir(parents=True, exist_ok=True)

        # A fresh, disposable profile with the GPU process disabled so WebRender
        # renders (and is captured) in the parent process.
        profile_dir = tempfile.mkdtemp(prefix="wrshell-profile-")
        with open(os.path.join(profile_dir, "user.js"), "w") as fh:
            fh.write('user_pref("layers.gpu-process.enabled", false);\n')

        env = dict(os.environ)
        env["LD_PRELOAD"] = str(lib_path)
        env["MOZ_DISABLE_CONTENT_SANDBOX"] = "1"
        env["WR_RENDERDOC_CAPTURE_PATH"] = str(captures_dir / "wr")

        log(f"Launching Firefox with RenderDoc preloaded ({lib_path})")
        firefox_proc = subprocess.Popen(
            [
                str(firefox),
                "-no-remote",
                "-new-instance",
                "-profile",
                profile_dir,
                url,
            ],
            env=env,
            stdin=subprocess.DEVNULL,
        )

    try:
        # Build and run wrshell, telling it where to find qrenderdoc.
        manifest = os.path.join(topsrcdir, "gfx", "wr", "wrshell", "Cargo.toml")
        wrshell_env = dict(os.environ)
        wrshell_env["WR_RENDERDOC_DIR"] = renderdoc_root
        mode = "gui" if gui else "repl"

        log(f"Starting wrshell ({mode}); close it to shut down.")
        return subprocess.call(
            ["cargo", "run", "--manifest-path", manifest, "--", mode],
            env=wrshell_env,
        )
    finally:
        # Tear down only the Firefox instance we launched.
        if firefox_proc is not None and firefox_proc.poll() is None:
            firefox_proc.terminate()
            try:
                firefox_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                firefox_proc.kill()
