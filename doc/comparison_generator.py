import importlib
import inspect


def calc_totals(base_mod, ref_mods, cls):
    base_obj, _ = import_mod(base_mod, cls)
    base_funcs = get_functions(base_obj)

    totals = [len(base_funcs)]
    for ref_mod in ref_mods:
        ref_obj, _ = import_mod(ref_mod, cls)
        ref_funcs = get_functions(ref_obj)

        totals.append(len(ref_funcs & base_funcs))

    return totals


def get_functions(obj):
    funcs = []
    for n, _ in inspect.getmembers(obj):
        if n in ["test"]:
            continue
        if not callable(getattr(obj, n)):
            continue
        if isinstance(getattr(obj, n), type):
            continue
        if not n[0].islower():
            continue
        if n.startswith("__"):
            continue
        funcs.append(n)

    return set(funcs)


def import_mod(mod, cls):
    obj = importlib.import_module(mod)
    if cls:
        obj = getattr(obj, cls)
        return obj, ":meth:`{}.{}.{{}}`".format(mod, cls)
    else:
        # ufunc is not a function
        return obj, ":obj:`{}.{{}}`".format(mod)


def generate_totals(base_mod, ref_mods, base_type, ref_types, cls):
    all_types = [base_type] + ref_types
    header = ", ".join("**{} Total**".format(t) for t in all_types)
    header = "   {}".format(header)

    totals = calc_totals(base_mod, ref_mods, cls)

    cells = ", ".join(str(t) for t in totals)
    total = "   {}".format(cells)

    return [header, total]


def generate_comparison_rst(base_mod, ref_mods, base_type, ref_types, cls):
    base_obj, base_fmt = import_mod(base_mod, cls)
    base_funcs = get_functions(base_obj)

    header = ", ".join([base_type] + ref_types)

    rows = []
    for f in sorted(base_funcs):
        base_cell = base_fmt.format(f)

        ref_cells = []
        for ref_mod in ref_mods:
            ref_obj, ref_fmt = import_mod(ref_mod, cls)
            ref_funcs = get_functions(ref_obj)

            ref_cell = r"\-"
            if f in ref_funcs:
                ref_cell = ref_fmt.format(f)

            ref_cells.append(ref_cell)

        cells = ", ".join([base_cell] + ref_cells)
        line = "   {}".format(cells)
        rows.append(line)

    totals = generate_totals(base_mod, ref_mods, base_type, ref_types, cls)

    return (
        [".. csv-table::", "   :header: {}".format(header), ""] + rows + totals
    )


def section(header, base_mod, ref_mods, base_type, ref_types, cls=None):
    comparison_rst = generate_comparison_rst(
        base_mod, ref_mods, base_type, ref_types, cls
    )

    return [header, "~" * len(header), ""] + comparison_rst + [""]


def generate_totals_numbers(header, base_mod, ref_mods, cls=None):
    base_obj, _ = import_mod(base_mod, cls)
    base_funcs = get_functions(base_obj)

    counter_funcs = [len(base_funcs)]
    for ref_mod in ref_mods:
        ref_obj, _ = import_mod(ref_mod, cls)
        ref_funcs = get_functions(ref_obj)

        counter_funcs.append(len(ref_funcs & base_funcs))

    totals = [header] + calc_totals(base_mod, ref_mods, cls)

    cells = ", ".join(str(t) for t in totals)
    total = "   {}".format(cells)

    return total, counter_funcs


def generate_table_numbers(base_mod, ref_mods, base_type, ref_types, cls=None):
    all_types = ["Name"] + [base_type] + ref_types
    header = ", ".join("**{}**".format(t) for t in all_types)
    header = "   {}".format(header)

    rows = []
    counters_funcs = []

    totals = []
    totals_, counters_funcs_ = generate_totals_numbers(
        "Module-Level", base_mod, ref_mods
    )
    totals.append(totals_)
    counters_funcs.append(counters_funcs_)
    cells = ", ".join(str(t) for t in totals)
    total = "   {}".format(cells)
    rows.append(total)

    totals = []
    totals_, counters_funcs_ = generate_totals_numbers(
        "Multi-Dimensional Array", base_mod, ref_mods, cls="ndarray"
    )
    totals.append(totals_)
    counters_funcs.append(counters_funcs_)
    cells = ", ".join(str(t) for t in totals)
    total = "   {}".format(cells)
    rows.append(total)

    totals = []
    totals_, counters_funcs_ = generate_totals_numbers(
        "Linear Algebra",
        base_mod + ".linalg",
        [m + ".linalg" for m in ref_mods],
    )
    totals.append(totals_)
    counters_funcs.append(counters_funcs_)
    cells = ", ".join(str(t) for t in totals)
    total = "   {}".format(cells)
    rows.append(total)

    totals = []
    totals_, counters_funcs_ = generate_totals_numbers(
        "Discrete Fourier Transform",
        base_mod + ".fft",
        [m + ".fft" for m in ref_mods],
    )
    totals.append(totals_)
    counters_funcs.append(counters_funcs_)
    cells = ", ".join(str(t) for t in totals)
    total = "   {}".format(cells)
    rows.append(total)

    totals = []
    totals_, counters_funcs_ = generate_totals_numbers(
        "Random Sampling",
        base_mod + ".random",
        [m + ".random" for m in ref_mods],
    )
    totals.append(totals_)
    counters_funcs.append(counters_funcs_)
    cells = ", ".join(str(t) for t in totals)
    total = "   {}".format(cells)
    rows.append(total)

    counter_functions = []
    for i in range(len(counters_funcs[0])):
        counter = 0
        for j in range(len(counters_funcs)):
            counter += counters_funcs[j][i]
        counter_functions.append("{}".format(counter))

    summary = ["Total"] + counter_functions
    cells = ", ".join(str(t) for t in summary)
    summary_total = "   {}".format(cells)
    rows.append(summary_total)

    comparison_rst = [".. csv-table::", ""] + [header] + rows

    return ["Summary", "~" * len("Summary"), ""] + comparison_rst + [""]


def generate():
    ref_mods = []
    ref_types = []
    ref_vers = []

    try:
        import dpnp

        ref_mods += ["dpnp"]
        ref_types += ["DPNP"]
        ref_vers = ["DPNP(v{})".format(dpnp.__version__)]
    except ImportError as err:
        print(f"DOCBUILD: Can't load DPNP module with error={err}")

    try:
        import cupy

        ref_mods += ["cupy"]
        ref_types += ["CuPy"]
        ref_vers += ["CuPy(v{})".format(cupy.__version__)]
    except ImportError as err:
        print(f"DOCBUILD: Can't load CuPy module with error={err}")

    try:
        import numpy

        base_mod = "numpy"  # TODO: Why string?
        base_type = "NumPy"
        base_ver = "{}(v{})".format(base_type, numpy.__version__)
    except ImportError as err:
        print(f"DOCBUILD: Can't load {base_type} module with error={err}")

    header = " / ".join([base_ver] + ref_vers) + " APIs"
    buf = ["**{}**".format(header), ""]

    buf += generate_table_numbers(base_mod, ref_mods, base_type, ref_types)
    buf += section("Module-Level", base_mod, ref_mods, base_type, ref_types)
    buf += section(
        "Multi-Dimensional Array",
        base_mod,
        ref_mods,
        base_type,
        ref_types,
        cls="ndarray",
    )
    buf += section(
        "Linear Algebra",
        base_mod + ".linalg",
        [m + ".linalg" for m in ref_mods],
        base_type,
        ref_types,
    )
    buf += section(
        "Discrete Fourier Transform",
        base_mod + ".fft",
        [m + ".fft" for m in ref_mods],
        base_type,
        ref_types,
    )
    buf += section(
        "Random Sampling",
        base_mod + ".random",
        [m + ".random" for m in ref_mods],
        base_type,
        ref_types,
    )

    return "\n".join(buf)
