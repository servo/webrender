#!/usr/bin/env bash

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */

# This must be run from the root webrender directory!

set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

pushd wrench
python script/headless.py reftest
cargo build --release
cargo run --release -- --precache \
    reftest reftests/clip/fixed-position-clipping.yaml
popd
