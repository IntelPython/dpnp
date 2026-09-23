"""Add the temporary DPNP_TOPK_ALGO switch to a dpnp source tree.

usage: python algo_switch.py SRC        SRC is a dpnp checkout

Builds the tree afterwards to get an ALGO_ROOT, see the README. The switch
makes topk_caller read DPNP_TOPK_ALGO=select|merge|sort and force that
algorithm, so the three can be measured against each other in one build.
Anything else keeps the build's own heuristic. Don't commit the result.

This edits by matching the two places it needs instead of applying a context
patch, so it keeps working as topk.cpp changes around them. It is idempotent
and says what it did. It needs the radix select changes to be present, that
is the "if (axis_nelems >= radix_select_min_nelems)" line.
"""

import sys
from pathlib import Path

INCLUDES = ("#include <cstdint>\n", "#include <cstdint>\n#include <cstdlib>\n")
STRING = ("#include <optional>\n", "#include <optional>\n#include <string>\n")
GUARD = "        if (axis_nelems >= radix_select_min_nelems) {\n"
SWITCH = """\
        // TEMPORARY measurement switch, DPNP_TOPK_ALGO=select|merge|sort
        // forces an algorithm, anything else keeps the heuristic
        const char *algo_env = std::getenv("DPNP_TOPK_ALGO");
        const std::string algo = (algo_env) ? algo_env : "";
        if (algo == "sort") {
            using dpnp::tensor::kernels::topk_radix_impl;
            return topk_radix_impl<argTy, IndexTy>(
                exec_q, iter_nelems, axis_nelems, k, !largest, arg_cp,
                vals_cp, inds_cp, depends);
        }
        if (algo == "select" ||
            (algo != "merge" && axis_nelems >= radix_select_min_nelems)) {
"""

src = Path(sys.argv[1]) / "dpnp/tensor/libtensor/source/sorting/topk.cpp"
if not src.is_file():
    sys.exit(f"{src} not found, is {sys.argv[1]} a dpnp checkout?")
text = src.read_text()

if "DPNP_TOPK_ALGO" in text:
    print(f"{src} already has the switch, nothing to do")
    sys.exit(0)
if GUARD not in text:
    sys.exit(
        f"{src} has no radix select threshold to switch on.\n"
        "Copy the radix select changes into the tree first, they are\n"
        "uncommitted in the development checkout:\n"
        "  cd <dev checkout> && git status --short\n"
        "  git diff HEAD -- dpnp/tensor | git -C <SRC> apply -"
    )
for old, new in (INCLUDES, STRING):
    if old not in text:
        sys.exit(f"{src} has no {old.strip()!r} line to add an include after")
    text = text.replace(old, new, 1)
src.write_text(text.replace(GUARD, SWITCH, 1))
print(f"added the DPNP_TOPK_ALGO switch to {src}")
