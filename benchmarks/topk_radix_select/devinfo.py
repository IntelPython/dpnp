"""Device facts the benchmarks size themselves by.

usage: python devinfo.py DEVICE [DTYPE]    one line for a result file header

An integrated GPU's memory is the host's RAM, a discrete GPU's is its own
VRAM, and the two behave differently enough here to be worth recording with
every measurement: on the Max 1100 dpctl's radix sort beats radix select on
rows of up to 64 one byte keys, on the integrated GPU it loses by 2 to 4x.

In C++ the query is sycl::aspect::ext_oneapi_is_integrated_gpu, which reports
correctly for both of this machine's devices. dpctl does not expose that
aspect, so is_integrated() uses the consequence instead: an integrated GPU's
global memory is the host's RAM, so the two sizes are within a few percent
(this machine reports 16.40 GiB of GPU memory for 15.34 GiB of RAM), while a
discrete GPU's is unrelated. Set INTEGRATED=0 or 1 to override.
"""

import os
import sys

# device bytes per element that top_k needs at k = n: the input, the k values
# and the k int64 indices, and about as much again for the sort or select
# scratch. On an integrated GPU the numpy source array shares this memory.
SCRATCH = 2


def is_integrated(d):
    env = os.environ.get("INTEGRATED")
    if env is not None:
        return bool(int(env))
    if not d.has_aspect_gpu:
        return False
    ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return d.global_mem_size > 0.7 * ram


def bytes_per_elem(d, itemsize):
    b = SCRATCH * (2 * itemsize + 8)
    return b + itemsize if is_integrated(d) else b


def element_cap(d, itemsize, max_elems):
    """How many elements one case may hold, the smallest of MAX_ELEMS, the
    share MEM_FRACTION of the device memory that fits, and the largest int64
    index output a single allocation can hold."""
    frac = float(os.environ.get("MEM_FRACTION", 0.4))
    return min(
        max_elems,
        int(frac * d.global_mem_size) // bytes_per_elem(d, itemsize),
        d.max_mem_alloc_size // 8,
    )


def kind(d):
    if not d.has_aspect_gpu:
        return "not a gpu"
    return "integrated gpu" if is_integrated(d) else "discrete gpu"


def describe(d, itemsize=None, max_elems=None):
    s = (
        f"{d.name}, {d.max_compute_units} CUs, {kind(d)}, "
        f"{d.global_mem_size / 2**30:.1f} GiB, "
        f"max alloc {d.max_mem_alloc_size / 2**30:.1f} GiB"
    )
    if itemsize is not None:
        cap = element_cap(d, itemsize, max_elems)
        s += f", cap {cap / 2**20:.0f}M elements of {itemsize} bytes"
    return s


if __name__ == "__main__":
    import dpnp.tensor as dpt

    dev = dpt.empty(0, device=sys.argv[1]).sycl_device
    if len(sys.argv) > 2:
        import numpy as np

        itemsize = np.dtype(sys.argv[2]).itemsize
        max_elems = int(eval(os.environ.get("MAX_ELEMS", "2**28")))
        print(describe(dev, itemsize, max_elems))
    else:
        print(describe(dev))
