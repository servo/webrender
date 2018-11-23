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
cargo build --verbose --no-default-features
cargo build --verbose --no-default-features --features capture
cargo build --verbose --features capture,profiler
cargo build --verbose --features replay
popd

pushd wrench
cargo build --verbose --features env_logger
popd

pushd examples
cargo build --verbose
popd

cargo test --all --verbose
