#!/usr/bin/env bash

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */

# This must be run from the root webrender directory!

set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

pushd webrender_api
cargo test --verbose --features "ipc"
popd

pushd webrender
cargo check --verbose --no-default-features
cargo check --verbose --no-default-features --features capture
cargo check --verbose --features capture,profiler
cargo check --verbose --features replay
cargo check --verbose --no-default-features --features pathfinder
popd

pushd wrench
cargo check --verbose --features env_logger
popd

pushd examples
cargo check --verbose
popd

cargo test --all --verbose
