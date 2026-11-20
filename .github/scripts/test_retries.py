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

"""Report the tests a CI retry reruns with ``pytest --lf``.

``record`` runs before each ``--lf`` attempt: it prints the tests left in
pytest's last-failed cache as a GitHub warning annotation and, with
``TEST_RETRIES_FILE`` set, appends them there as a JSON line. A first attempt
has no cache, so it prints nothing.

``report`` collects the recorded files, downloaded as one directory per job
artifact, into the markdown of a PR comment and sets the ``retried`` step
output.
"""

import argparse
import json
import os
import pathlib

LASTFAILED = pathlib.Path(".pytest_cache", "v", "cache", "lastfailed")
MAX_ANNOTATED = 50
MAX_REPORTED = 100


def _capped(tests, limit):
    return tests[:limit] + ["..."] * (len(tests) > limit)


def record():
    # the cache is in the rootdir, which is the cwd with --pyargs
    if not LASTFAILED.exists():
        return
    tests = sorted(json.loads(LASTFAILED.read_text()))
    if not tests:
        return

    # '%' and newlines must be escaped in a workflow command
    msg = "%0A".join(
        t.replace("%", "%25") for t in _capped(tests, MAX_ANNOTATED)
    )
    print(f"::warning title=Rerunning {len(tests)} failed tests::{msg}")

    path = os.environ.get("TEST_RETRIES_FILE")
    if path:
        # SKIP_TENSOR_TESTS is set in exactly the dpnp suite runs
        tensor = not os.environ.get("SKIP_TENSOR_TESTS")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"tensor": tensor, "tests": tests}) + "\n")


def report(artifacts_dir, out_file, prefix):
    sections = []
    for d in sorted(pathlib.Path(artifacts_dir).glob("*")):
        job = d.name.removeprefix(prefix)
        for f in sorted(d.glob("*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                # node IDs are relative to the --pyargs package
                if rec["tensor"]:
                    suite, pkg = "tensor", "dpnp/tests/tensor/"
                else:
                    suite, pkg = "dpnp", "dpnp/tests/"
                tests = [pkg + t for t in rec["tests"]]
                body = "\n".join(_capped(tests, MAX_REPORTED))
                sections.append(
                    f"<details><summary>{job}, {suite} tests: "
                    f"{len(tests)} rerun</summary>\n\n"
                    f"```\n{body}\n```\n\n</details>"
                )

    if sections:
        head = (
            f"Tests retried on CI at {os.environ['HEAD_SHA']}: they failed "
            "in the full run and were rerun on their own with `--lf`. A test "
            "that passed on the rerun may depend on the tests that ran "
            "before it."
        )
        pathlib.Path(out_file).write_text(
            "\n\n".join([head] + sections) + "\n", encoding="utf-8"
        )

    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
        f.write(f"retried={'true' if sections else 'false'}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("record", help="annotate and record the tests to rerun")
    rep = sub.add_parser("report", help="compose the PR comment")
    rep.add_argument("artifacts_dir", help="directory of downloaded artifacts")
    rep.add_argument("out_file", help="markdown file to write")
    rep.add_argument(
        "--prefix", default="", help="artifact name prefix to drop"
    )
    args = parser.parse_args()

    if args.cmd == "record":
        record()
    else:
        report(args.artifacts_dir, args.out_file, args.prefix)


if __name__ == "__main__":
    main()
