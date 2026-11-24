#!/bin/bash
# *****************************************************************************
# Copyright (c) 2026, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

# Updates the gh-pages branch, safe against concurrent writers: each attempt
# rebuilds the commit on the latest branch tip and a rejected push is retried.
# Only the index is touched, so the site is never checked out.
#
# Usage:
#   update_gh_pages.sh publish <src-dir> <dest-dir> <message>
#       <dest-dir> "." adds <src-dir> to the site root, keeping other files;
#       any other <dest-dir> is replaced by <src-dir>
#   update_gh_pages.sh remove <message> <dir>...
#   update_gh_pages.sh prune
#       removes pull/<N> of every closed or merged pull request
#
# Environment: GH_TOKEN (push access; gh also uses it), GITHUB_REPOSITORY.
# Optional: GH_PAGES_BRANCH, GH_PAGES_REMOTE, GH_PAGES_MAX_ATTEMPTS,
# GIT_AUTHOR_NAME, GIT_AUTHOR_EMAIL.

set -euo pipefail

branch=${GH_PAGES_BRANCH:-gh-pages}
remote=${GH_PAGES_REMOTE:-https://github.com/${GITHUB_REPOSITORY}.git}
max_attempts=${GH_PAGES_MAX_ATTEMPTS:-5}

export GIT_AUTHOR_NAME=${GIT_AUTHOR_NAME:-github-actions[bot]}
export GIT_AUTHOR_EMAIL=${GIT_AUTHOR_EMAIL:-github-actions[bot]@users.noreply.github.com}
export GIT_COMMITTER_NAME=$GIT_AUTHOR_NAME
export GIT_COMMITTER_EMAIL=$GIT_AUTHOR_EMAIL

# Config from the environment keeps the token off command lines and out of files
if [[ -n "${GH_TOKEN:-}" ]]; then
    export GIT_CONFIG_COUNT=1
    export GIT_CONFIG_KEY_0="http.https://github.com/.extraheader"
    GIT_CONFIG_VALUE_0="AUTHORIZATION: basic $(printf 'x-access-token:%s' "${GH_TOKEN}" | base64 -w0)"
    export GIT_CONFIG_VALUE_0
fi

usage() {
    sed -n '/^# Usage:/,/^# Environment/p' "$0" | sed '$d; s/^# \{0,1\}//' >&2
    exit 2
}

mode=${1:-}
[[ -n "${mode}" ]] || usage
shift

# Arguments are checked before the clone
case "${mode}" in
    publish)
        [[ $# -eq 3 ]] || usage
        src=$(cd "$1" && pwd)
        dest=${2%/}
        message=$3
        # Also keeps <dest-dir> safe as a sed replacement below
        if [[ "${dest}" != "." && ! "${dest}" =~ ^[A-Za-z0-9_-][A-Za-z0-9._-]*(/[A-Za-z0-9_-][A-Za-z0-9._-]*)*$ ]]; then
            echo "::error::Invalid destination directory: ${dest}"
            exit 2
        fi
        ;;
    remove)
        [[ $# -ge 2 ]] || usage
        message=$1
        shift
        dirs=("$@")
        ;;
    prune)
        [[ $# -eq 0 ]] || usage
        ;;
    *)
        usage
        ;;
esac

work=$(mktemp -d)
trap 'rm -rf "${work}"' EXIT
export GIT_DIR="${work}/repo.git"
export GIT_WORK_TREE="${work}/empty"
mkdir "${GIT_WORK_TREE}"

# Blobless: commits and trees are enough to build the new commit
git clone -q --bare --filter=blob:none --depth=1 --single-branch \
    --branch "${branch}" "${remote}" "${GIT_DIR}"

if [[ "${mode}" == "prune" ]]; then
    if ! git cat-file -e "refs/heads/${branch}:pull" 2>/dev/null; then
        echo "No pull/ directory on ${branch}"
        exit 0
    fi
    pulls=$(git ls-tree -d --name-only "refs/heads/${branch}:pull")
    dirs=()
    lookups=0
    failed=0
    while IFS= read -r num; do
        [[ "${num}" =~ ^[0-9]+$ ]] || continue
        lookups=$((lookups + 1))
        # A failed lookup keeps the preview
        if ! state=$(gh pr view "${num}" --repo "${GITHUB_REPOSITORY}" --json state --jq .state); then
            echo "::warning::Failed to look up pull request ${num}, keeping pull/${num}"
            failed=$((failed + 1))
            continue
        fi
        if [[ "${state}" == "CLOSED" || "${state}" == "MERGED" ]]; then
            dirs+=("pull/${num}")
        fi
    done <<< "${pulls}"
    if [[ ${failed} -gt 0 && ${failed} -eq ${lookups} ]]; then
        echo "::error::Failed to look up all ${lookups} pull requests"
        exit 1
    fi
    if [[ ${#dirs[@]} -eq 0 ]]; then
        echo "No docs of closed pull requests to remove"
        exit 0
    fi
    printf 'Removing %s\n' "${dirs[@]}"
    message="Removing docs for ${#dirs[@]} closed pull requests"
    mode=remove
fi

# Pathspecs below are relative to the site root
cd "${GIT_WORK_TREE}"

if [[ "${mode}" == "publish" ]]; then
    # Hash <src-dir> in a plain repo: in the blobless one, each blob already on
    # the remote would be fetched; then expose its objects as an alternate
    src_repo="${work}/src.git"
    src_index="${work}/src.index"
    env -u GIT_DIR -u GIT_WORK_TREE git init -q --bare "${src_repo}"
    GIT_DIR="${src_repo}" GIT_INDEX_FILE="${src_index}" \
        git -C "${src}" --work-tree="${src}" add -A -f .
    echo "${src_repo}/objects" >> "${GIT_DIR}/objects/info/alternates"
fi

for ((attempt = 1; ; attempt++)); do
    if [[ ${attempt} -gt 1 ]]; then
        git fetch -q --depth=1 origin "+refs/heads/${branch}:refs/heads/${branch}"
    fi
    base=$(git rev-parse "refs/heads/${branch}")
    git read-tree "${base}"

    if [[ "${mode}" == "remove" ]]; then
        git rm -r -q --cached --ignore-unmatch -- "${dirs[@]}"
    else
        prefix=""
        if [[ "${dest}" != "." ]]; then
            git rm -r -q --cached --ignore-unmatch -- "${dest}"
            prefix="${dest}/"
        fi
        # Unlike read-tree --prefix, this doesn't fetch blobs of existing entries
        GIT_INDEX_FILE="${src_index}" git ls-files -s -z |
            sed -z "s|\t|\t${prefix}|" |
            git update-index -z --index-info
    fi

    # --missing-ok: blobs of the rest of the site are not fetched
    tree=$(git write-tree --missing-ok)
    if [[ "${tree}" == "$(git rev-parse "${base}^{tree}")" ]]; then
        echo "No changes to push to ${branch}"
        exit 0
    fi
    commit=$(git commit-tree "${tree}" -p "${base}" -m "${message}")

    # --no-thin: a thin pack would fetch old blobs as delta bases
    if git push -q --no-thin origin "${commit}:refs/heads/${branch}"; then
        echo "Pushed ${commit} to ${branch} (attempt ${attempt})"
        exit 0
    fi
    # Only a concurrent writer moves the tip; a failed ls-remote is retried
    tip=$(git ls-remote origin "refs/heads/${branch}" | cut -f1) || tip=""
    if [[ "${tip}" == "${base}" ]]; then
        echo "::error::Failed to push to ${branch}, not retrying: its tip did not move"
        exit 1
    fi
    if [[ ${attempt} -ge ${max_attempts} ]]; then
        echo "::error::Failed to push to ${branch} after ${attempt} attempts"
        exit 1
    fi
    delay=$((attempt * 5 + RANDOM % 10))
    echo "Push to ${branch} rejected, retrying in ${delay} s"
    sleep "${delay}"
done
