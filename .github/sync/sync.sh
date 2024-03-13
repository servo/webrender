#!/bin/sh
# Usage: sync.sh <path/to/filtered>
set -eu

DOWNSTREAM="git@github.com:mrobinson/webrender.git"
ROOT=$(pwd)
CACHE_DIR=$ROOT/_cache

# Configure git because we will be making commits.
git config --global user.email "noreply@github.com"
git config --global user.name "Webrender Upstream Sync"

step() {
    if [ "${TERM-}" != '' ]; then
        tput setaf 12
    fi
    >&2 printf '* %s\n' "$*"
    if [ "${TERM-}" != '' ]; then
        tput sgr0
    fi
}

step "Creating work directory for sync"
mkdir -p "$1"
cd -- "$1"
filtered=$(pwd)

step "Creating cache diretory if needed"
mkdir -p "$CACHE_DIR"
cd "$CACHE_DIR"
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
git -C "$filtered" filter-repo --force --paths-from-file "$ROOT/.github/sync/webrender.paths"

step "Adding filtered repository as a remote"
cd "$ROOT"
git remote add upstream "$filtered"
git fetch upstream

hash_file=".github/sync/UPSTREAM_COMMIT"
hash=`cat $hash_file`
number_of_commits=`git log $hash..upstream/master --pretty=oneline | wc -l`

if [ $number_of_commits != '0' ]; then
    step "Applying $number_of_commits new commits"
    git cherry-pick $hash..upstream/master
    git rev-parse upstream/master > "$hash_file"
    git commit "$hash_file" -m "Syncing to upstream (`cat $hash_file`)"

    step "Pushing new main branch"
    git push origin main
else
    step "No new commits. Doing nothing."
fi
