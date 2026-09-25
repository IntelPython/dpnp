"""Add the temporary DPNP_TOPK_ALGO switch to a dpnp source tree.

usage: python algo_switch.py SRC        SRC is a dpnp checkout

Builds the tree afterwards to get an ALGO_ROOT, see the README. The switch
makes topk_caller read DPNP_TOPK_ALGO=select|merge|sort and force that
algorithm, so the three can be measured against each other in one build.
Anything else keeps the build's own heuristic. It also makes
radix_select_dispatch read DPNP_TOPK_SG_MAX_N, the longest rows given to the
sub-group per row kernel (0 sends them all to the work-group per row one),
to find where that kernel stops paying off. Don't commit the result.

This edits by matching the places it needs instead of applying a context
patch, so it keeps working as the files change around them. It is idempotent
and says what it did. It needs the radix select changes to be present: in
topk.cpp either the "if (axis_nelems >= radix_select_min_nelems)" threshold
of the builds which sent short rows to merge sort, or the plain
"if constexpr (use_radix_select<argTy>::value)" branch of those which don't.
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

SG_INCLUDE = ("#include <cstdint>\n", "#include <cstdint>\n#include <cstdlib>\n")
# the row length is "n" in older trees and "n_values" in newer ones
SG_GUARDS = ("    if (n <= std::min(sub_group_max_n,",
             "    if (n_values <=\n            std::min(sub_group_max_n,")
SG_SWITCH = """\
    // TEMPORARY measurement switch, DPNP_TOPK_SG_MAX_N overrides the longest
    // rows given to the sub-group kernel
    std::size_t sg_max_n_env = sub_group_max_n;
    if (const char *e = std::getenv("DPNP_TOPK_SG_MAX_N")) {
        sg_max_n_env = std::strtoull(e, nullptr, 10);
    }
"""

root = Path(sys.argv[1])
hpp = root / "dpnp/tensor/libtensor/include/kernels/sorting/radix_select.hpp"
if not hpp.is_file():
    sys.exit(f"{hpp} not found, is {sys.argv[1]} a dpnp checkout?")
htext = hpp.read_text()
if "DPNP_TOPK_SG_MAX_N" in htext:
    print(f"{hpp} already has the switch")
elif not any(g in htext for g in SG_GUARDS):
    print(f"{hpp} has no sub-group kernel, so no DPNP_TOPK_SG_MAX_N switch")
else:
    guard = next(g for g in SG_GUARDS if g in htext)
    new_guard = SG_SWITCH + guard.replace("sub_group_max_n", "sg_max_n_env")
    htext = htext.replace(SG_INCLUDE[0], SG_INCLUDE[1], 1)
    hpp.write_text(htext.replace(guard, new_guard, 1))
    print(f"added the DPNP_TOPK_SG_MAX_N switch to {hpp}")

# every real type goes to radix select, merge sort is only instantiated for
# complex ones, so the switch brings its own merge sort call
BRANCH = "    if constexpr (use_radix_select<argTy>::value) {\n"
BRANCH_SWITCH = BRANCH + """\
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
        if (algo == "merge") {
            using dpnp::tensor::kernels::topk_merge_impl;
            namespace rc = dpnp::tensor::rich_comparisons;
            if (largest) {
                return topk_merge_impl<argTy, IndexTy,
                                       typename rc::DescendingSorter<argTy>::type>(
                    exec_q, iter_nelems, axis_nelems, k, arg_cp, vals_cp,
                    inds_cp, depends);
            }
            return topk_merge_impl<argTy, IndexTy,
                                   typename rc::AscendingSorter<argTy>::type>(
                exec_q, iter_nelems, axis_nelems, k, arg_cp, vals_cp, inds_cp,
                depends);
        }
"""

src = root / "dpnp/tensor/libtensor/source/sorting/topk.cpp"
if not src.is_file():
    sys.exit(f"{src} not found, is {sys.argv[1]} a dpnp checkout?")
text = src.read_text()

if "DPNP_TOPK_ALGO" in text:
    print(f"{src} already has the switch, nothing to do")
    sys.exit(0)
if GUARD in text:
    text = text.replace(GUARD, SWITCH, 1)
elif BRANCH in text:
    text = text.replace(BRANCH, BRANCH_SWITCH, 1)
else:
    sys.exit(
        f"{src} has neither radix select routing to switch on.\n"
        "Copy the radix select changes into the tree first, they are\n"
        "uncommitted in the development checkout:\n"
        "  cd <dev checkout> && git status --short\n"
        "  git diff HEAD -- dpnp/tensor | git -C <SRC> apply -"
    )
for old, new in (INCLUDES, STRING):
    if old not in text:
        sys.exit(f"{src} has no {old.strip()!r} line to add an include after")
    text = text.replace(old, new, 1)
src.write_text(text)
print(f"added the DPNP_TOPK_ALGO switch to {src}")
