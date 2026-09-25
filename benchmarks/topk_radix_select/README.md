# top_k radix select benchmarks

These scripts measure `dpnp.tensor.top_k` with radix select and check its
results. The measurements chose the design and the routing in
`topk.cpp`, which uses radix select for every real type and merge sort for
complex types. Merge sort used to take rows shorter than 16 elements as
well, until the sub-group kernel (see `sgn`) beat it on those.

| file | what it does |
| --- | --- |
| `run_all.sh` | driver that runs all suites below, configured by `config.sh` |
| `compare.sh` | compares builds or algorithms, one process per (case, k, variant) |
| `sweep.py` | the timing harness, minimum of up to `SWEEP_NIT` calls per variant |
| `devinfo.py` | integrated or discrete, and how large a case may be |
| `check_topk.py` | exact check against a stable NumPy reference, 5406 cases on all real dtypes |
| `check_long.py` | long rows (up to 2**22) with NaNs and signed zeros |
| `make_root.sh` | makes a build root, see below |
| `build_so.sh` | builds only the sorting extension of another checkout |
| `algo_switch.py` | adds the temporary `DPNP_TOPK_ALGO=select\|merge\|sort` and `DPNP_TOPK_SG_MAX_N` switches for the `algo`, `rowsn` and `sgn` suites |

## Suites

```
./run_all.sh --list
./run_all.sh                    # everything, several hours
./run_all.sh check builds       # some suites
QUICK=1 ./run_all.sh            # first dtype and case of each, checks the setup
DEVICES=gpu ./run_all.sh kn     # one device
GPU_DTYPES="i1 u1" ./run_all.sh algo3   # other dtypes (CPU_DTYPES for cpu)
ROW_SCALE=64 ./run_all.sh kn    # 64 times the rows, see "Data size"
./run_all.sh long rowsn         # the large and long-row suites
```

| suite | compares | cases |
| --- | --- | --- |
| `check` | correctness of `NEW_ROOT` | `check_topk.py <dev> set`, `check_long.py <dev>` |
| `builds` | `NEW_ROOT` against every configured old build (`MASTER`, `BASE`, `SORTED`, `FEAT_ROOT`) | 65536x64 to 1x2\*\*25, k = 1, 10, 100, 1000 |
| `kn` | new vs master, k close to n | n = 16 to 4096, k = n/2, 3n/4, 7n/8, n-1, n |
| `small` | new vs master on very short rows | n = 8 to 48, k = 1 to n |
| `long` | new vs master on a few very long rows with k near n | n = 2\*\*20 to 2\*\*28, 1 to 64 rows, k = n/2 to n |
| `algo` | radix select vs merge sort, set the former threshold of 16 | n = 4 to 32, many and few rows |
| `algo3` | radix select vs merge sort vs radix sort, including bool, i1 and u1 | n = 4 to 48; n = 64 to 256 with k = 1 to n; k near n up to n = 4096 |
| `rowsn` | the same three, varying rows and n independently | n fixed at 64/256 over 8192 to 2M rows, rows fixed at 64K/1M over n = 16 to 1024 |
| `sgn` | the sub-group per row kernel vs the work-group per row one, radix sort and merge sort | n = 4 to 128, k = 1, n/2, n |

`long` is the worst case for a selection algorithm: at k = n nothing can be
discarded, so it does a full sort's work and pays for the digit passes on top.
`rowsn` answers a question the other suites can't: radix select costs about
17 ns per row whatever n is, and radix sort a smaller per-row cost plus a
fixed floor, so for 1-byte keys the winner should follow the row count rather
than n — but every case in `algo3` holds 4M elements, where "rows above ~50000"
and "n below ~96" are the same line. `long` and `rowsn` choose their sizes on
purpose and ignore `ROW_SCALE`.

Those 17 ns are the cost of a work-group per row, which is what every row
under 2\*\*16 got: short rows were bound by scheduling work-groups, at 0.2 to
10% of peak bandwidth, and a GPU with more compute units didn't make them
faster. Rows of up to `sub_group_max_n` (64) elements now go to a kernel
that ranks each row within one sub-group instead, many rows to a
work-group. `sgn` finds where that bound belongs: its `sg` variant gives the
sub-group kernel every row it can hold (`sub_group_max_chunks` times the
device's smallest sub-group size, 128 on a device whose smallest is 16) and
`wg` gives it none. `rowsn`, `algo` and `algo3` results from before this
kernel measured the work-group kernel on those rows.

How many calls each measurement times follows from how long a call takes:
about `SWEEP_BUDGET` (3) seconds per variant, at most `SWEEP_NIT` (15) and at
least `MIN_NIT` (5). Every case that fits in a few milliseconds gets all 15,
as before, and a merge sort of one 2\*\*26-element row, which takes seconds,
gets 5. Five is enough: on an idle integrated GPU, one 2\*\*22-element merge
sort measured 209/217/215 ms over three runs of 5 calls and 214/212/211 ms
over three runs of 15. On a *busy* one the same case gave 490 to 1030 ms at
every iteration count, which is what the checks below are for.

The results go to `results/<suite>_<device>.txt` (set by `RESULTS_DIR`). Each
line has `rows n k`, the time in ms for each variant, and each variant's time
divided by the first variant's time (above 1x means the first one, the new
build or radix select, is faster). The last column is the fastest variant.
An `ERR` means that run failed or timed out; rerun it with `compare.sh`
directly to see the error. A `SKIP` means the case was too large for the
device, see "Data size".

`gpu` uses `GPU_SELECTOR`, or the default GPU if that is empty (see
Environment). `cpu` sets `ONEAPI_DEVICE_SELECTOR=$CPU_SELECTOR`, which is the
OpenCL CPU device by default. The GPU dtype lists include `f8` and `f2`. On a
device without them those runs show `ERR`.

## Data size

Most cases in `builds`, `kn`, `small` and the `algo` suites hold 4M elements,
16 MB of f4. A big GPU keeps that in its L2 cache (the Max 1100 has 108 MB),
and the fastest calls take about 0.2 ms, where launch overhead matters.
`ROW_SCALE=S` multiplies the rows of those cases by S and keeps n, so the
per-row behaviour stays comparable. `ROW_SCALE=64` takes them to 2\*\*28
elements; cases that are already that large stay about the same.

`devinfo.py` caps how many elements one case may hold, at the smallest of

* `MAX_ELEMS`, 2\*\*28 by default,
* the share `MEM_FRACTION` (0.4) of the device memory that the whole call
  fits in. At k = n `top_k` holds the input, the k values and the k int64
  indices, and about as much again in sort or select scratch, so roughly
  `2 * (2 * itemsize + 8)` bytes per element,
* `max_mem_alloc_size / 8`, the largest int64 index output a single
  allocation can hold. Level Zero limits one allocation to 4 GiB unless
  relaxed allocation limits are enabled; the Max 1100 used here reports
  45.6 GiB, so they are on.

On an integrated GPU the numpy source array is in the same memory as the
device array, so it counts towards the budget too, and `devinfo.py` adds it.
`ROW_SCALE` scaling is clamped to the cap; a case that is over the cap by
itself, like `1x2**28` of f8 on a 16 GiB device, is skipped with a message
on stderr and shows as `SKIP` in the result file (`ERR` means the run really
failed or timed out; `compare.sh` prints the reason on stderr).

One cap does not fit all three algorithms: radix sort needs an int64 index
workspace for the whole array, so it runs out of allocation before radix
select and merge sort do. On the integrated GPU here, with a 1 GiB maximum
allocation, `rowsn`'s 134M-element cases run under select and merge and fail
under sort with "Unable to allocate device_memory" — that `ERR` is the
device's limit, not a broken setup. Arrays over `HOST_CHUNK`
(2\*\*26) elements are filled on the device in row blocks so the host copy
stays small; smaller ones are generated in one call, which keeps the data of
all the existing cases unchanged.

To push a large discrete GPU, raise both the cap and the scaling:

```bash
ROW_SCALE=64 ./run_all.sh builds kn small                    # to 2**28
MAX_ELEMS=2**31 ROW_SCALE=512 MEM_FRACTION=0.5 ./run_all.sh kn algo3
```

## Integrated and discrete GPUs

An integrated GPU's memory is the host's RAM, a discrete GPU's is its own
VRAM, and they do not rank these algorithms the same way: on the Max 1100
dpctl's radix sort beats radix select on rows of up to 64 one-byte keys,
while on the integrated GPU it loses by 2 to 4x. Every result file header
says which kind the device is, so a file can't be misread later.

In C++ the query is `sycl::aspect::ext_oneapi_is_integrated_gpu`:

```cpp
const bool discrete_gpu =
    dev.is_gpu() && !dev.has(sycl::aspect::ext_oneapi_is_integrated_gpu);
```

It reports correctly for both of this machine's devices (1 for the integrated
GPU, 0 for the CPU, which is the right answer for a non-GPU). dpctl does not
expose that aspect, so `devinfo.py` uses the consequence instead: an
integrated GPU's global memory is the host's RAM, so the two sizes are within
a few percent (16.40 GiB of GPU memory for 15.34 GiB of RAM here), while a
discrete GPU's is unrelated. `INTEGRATED=0` or `1` overrides it.

## Why one process per measurement

On some GPUs a long sweep in one process slows down its later
measurements. `compare.sh` therefore starts a new Python process for every
(case, k, variant) and times only one configuration in each. Within a
process, `sweep.py` warms up, then interleaves the variants it was given
and reports the minimum time. Don't run benchmarks while something is
building.

Anything else on the device inflates these numbers, and a minimum won't hide
it, so two checks guard against the way that happens in practice. `run_all.sh`
refuses to start if a `sweep.py` is already running (`FORCE=1` overrides),
and `compare.sh` stops, killing the sweep it is waiting on, as soon as the
driver that started it is gone. Without the second one, interrupting a run
leaves its `compare.sh` loops going: three of them survived here and made
everything measured for the next half hour 2 to 3x too slow.

## Build roots

A root is a directory with a `dpnp` package. It is placed on `PYTHONPATH`,
so different builds can run from the same checkout. `make_root.sh` makes one
from a built dpnp tree, e.g. the editable development checkout. The root is
a tree of symlinks with a real copy of the chosen `_tensor_sorting_impl*.so`.
Only that extension differs between the builds compared here. The Python
side of `top_k` is the same, so everything else comes from the source tree.

```
# the build under test: the development tree itself works as a root
NEW_ROOT=~/repos/dpnp-topk

# master (or any other branch): build its sorting extension separately
git -C ~/repos/dpnp worktree add --detach /tmp/master_src master
so=$(./build_so.sh /tmp/master_src /tmp/build-master | tail -1)
./make_root.sh ~/repos/dpnp-topk "$so" /tmp/master_root

# ALGO_ROOT: the build under test with the algorithm switch
git -C ~/repos/dpnp-topk worktree add --detach /tmp/algo_src HEAD
# a worktree is at HEAD, so carry over the uncommitted changes of the build
# under test, otherwise algo_switch.py has nothing to switch on
git -C ~/repos/dpnp-topk diff HEAD -- dpnp/tensor | git -C /tmp/algo_src apply -
python algo_switch.py /tmp/algo_src
so=$(./build_so.sh /tmp/algo_src /tmp/build-algo | tail -1)
./make_root.sh ~/repos/dpnp-topk "$so" /tmp/algo_root

MASTER_ROOT=/tmp/master_root ALGO_ROOT=/tmp/algo_root ./run_all.sh
```

`build_so.sh` configures CMake the way the scikit-build install does. It uses
icx and icpx from `PATH` (the conda environment's, or oneAPI's through
`ONEAPI_SETVARS`) and the conda environment's Python, NumPy and dpctl. It
then builds only the `_tensor_sorting_impl` target, which takes about 5
minutes. For the
development tree itself, the usual rebuild is enough:
`ninja -C _skbuild/<platform>/cmake-build _tensor_sorting_impl`. After that,
copy the new `.so` into `dpnp/tensor/`.

`DPNP_TOPK_ALGO` and `DPNP_TOPK_SG_MAX_N` (the longest rows given to the
sub-group kernel, `0` for none) are only read by a build with
`algo_switch.py` applied.
`run_all.sh` refuses an `ALGO_ROOT` whose extension doesn't contain the
switch. Without the switch, all the variants would run the same code.
`algo_switch.py` edits `topk.cpp` in place by matching the two places it
needs, is idempotent, and tells you what is missing if the tree doesn't have
the radix select changes yet. Don't commit its edit.

## Environment

The scripts use the conda environment dpnp was built in: `CONDA_ENV`, which
defaults to the active environment (`$CONDA_PREFIX`). They put its `bin` first
on `PATH` and its `lib` first on `LD_LIBRARY_PATH`, because SYCL programs
don't run without the latter. With DPC++ installed from conda
(`dpcpp_linux-64`, `intel-opencl-rt` for the CPU device), nothing else is
needed. For a standalone oneAPI install, set
`ONEAPI_SETVARS=/opt/intel/oneapi/setvars.sh` and the scripts source it first.

`gpu` runs with `ONEAPI_DEVICE_SELECTOR=$GPU_SELECTOR` if one is set, otherwise
on the default GPU. On a machine with both an integrated and a discrete GPU,
the default may not be the one you want. Run `sycl-ls` and pick a device, e.g.
`GPU_SELECTOR=level_zero:1`. Every result file starts with the name of the
device it ran on and whether it is integrated or discrete.

## On another machine

```
conda activate <env with dpnp built in development mode>
cd <dpnp checkout with the radix select changes>/benchmarks/topk_radix_select
export NEW_ROOT=<that checkout> RESULTS_DIR=$PWD/results-$(hostname)
# master and ALGO_ROOT builds as in "Build roots" above, then
export MASTER_ROOT=... ALGO_ROOT=... BASE_ROOT= SORTED_ROOT= FEAT_ROOT=
QUICK=1 ./run_all.sh                    # checks the setup in a few minutes
GPU_SELECTOR=level_zero:0 ./run_all.sh
```

Every other variable in `config.sh` can also be set in the environment to
override it. The default roots point at this development machine's paths. A
root that doesn't exist is skipped with a message, and so is one set to empty.

On a large discrete GPU, the run that goes past the caches and answers what
the 4M-element cases can't:

```bash
export DEVICES=gpu MASTER_ROOT=... ALGO_ROOT=... BASE_ROOT= SORTED_ROOT= FEAT_ROOT=
./run_all.sh long rowsn                                  # the new suites
ROW_SCALE=64 RESULTS_DIR=$PWD/results-x64 ./run_all.sh builds kn small algo3
MAX_ELEMS=2**31 ROW_SCALE=512 MEM_FRACTION=0.5 \
  RESULTS_DIR=$PWD/results-x512 ./run_all.sh kn algo3
```
