#!/bin/sh
# Usage: sync.sh <path/to/filtered>
#
# A script to sync Mozilla's git mirror of WebRender [1] to the Servo project's
# downstream fork [2]. This script runs as part of a GitHub Action in this
# repository triggered on regular intervals.
#
# The procedure for the sync is:
#
# 1. Clone a copy of the GitHub gecko-dev repository to the "_cache" directory.
# 2. Filter that repository using `git-filter-repo` and create a new local git
#    repository with the filtered contents into a directory specified by the
#    argument passed to this script. The filtered contents are determined
#    by the configuration in `.github/sync/webrender.paths`.
# 3. Reset the `upstream` branch in the target repository to the new filtered
#    version.
#
# Note that this script relies on the idea that filtering `gecko-dev` the same
# way more than once will result in the same commit hashes, otherwise multiple
# copies of the version control history will be included in the repository.
#
# [1]: <https://github.com/mozilla/gecko-dev/> mirrored from
#      <https://hg.mozilla.org/mozilla-central>
# [2]: <https://github.com/mozilla/gecko-dev/>
set -eux

root_dir=$(pwd)
cache_dir=$root_dir/_cache

# Configure git because we will be making commits.
git_name="Webrender Upstream Sync"
git_email="noreply@github.com"

step() {
    if [ "${TERM-}" != '' ]; then
        tput setaf 12
    fi
    >&2 printf '* %s\n' "$*"
    if [ "${TERM-}" != '' ]; then
        tput sgr0
    fi
}

step "Creating directory for filtered upstream repo if needed"
mkdir -p "$1"
cd -- "$1"
filtered=$(pwd)

step "Creating cache directory if needed"
mkdir -p "$cache_dir"
cd "$cache_dir"
export PATH="$PWD:$PATH"

step "Downloading git-filter-repo if needed"
if ! git filter-repo --version 2> /dev/null; then
    curl -O https://raw.githubusercontent.com/newren/git-filter-repo/v2.38.0/git-filter-repo
    chmod +x git-filter-repo

    git filter-repo --version
fi

step "Cloning upstream if needed"
if ! [ -e upstream ]; then
    git clone --bare --single-branch --progress https://github.com/mozilla/gecko-dev.git upstream
fi

step "Updating upstream"
branch=$(git -C upstream rev-parse --abbrev-ref HEAD)
git -C upstream fetch origin $branch:$branch

step "Filtering upstream"
# Cloning and filtering is much faster than git filter-repo --source --target.
git clone upstream -- "$filtered"
git -C "$filtered" filter-repo --force --paths-from-file "$root_dir/.github/sync/webrender.paths"

step "Adding filtered repository as a remote"
cd "$root_dir"
git remote add filtered-upstream "$filtered"
git fetch filtered-upstream

step "Resetting 'upstream' branch to filtered repository HEAD"
git switch -c upstream
git reset --hard filtered-upstream/master

step Pushing new 'upstream'
git push -f origin upstream
