import importlib
import inspect


def get_functions(obj):
    funcs = []
    for n, _ in inspect.getmembers(obj):
        if n in ['test']:
            continue
        if not callable(getattr(obj, n)):
            continue
        if isinstance(getattr(obj, n), type):
            continue
        if not n[0].islower():
            continue
        if n.startswith('__'):
            continue
        funcs.append(n)

    return set(funcs)


def import_mod(mod, cls):
    obj = importlib.import_module(mod)
    if cls:
        obj = getattr(obj, cls)
        return obj, ':meth:`{}.{}.{{}}`'.format(mod, cls)
    else:
        # ufunc is not a function
        return obj, ':obj:`{}.{{}}`'.format(mod)


def generate_totals(base_mod, ref_mods, base_type, ref_types, cls):
    base_obj, _ = import_mod(base_mod, cls)
    base_funcs = get_functions(base_obj)

    all_types = [base_type] + ref_types
    header = ', '.join('**{} Total**'.format(t) for t in all_types)
    header = '   {}'.format(header)

    totals = [len(base_funcs)]
    for ref_mod in ref_mods:
        ref_obj, _ = import_mod(ref_mod, cls)
        ref_funcs = get_functions(ref_obj)

        totals.append(len(ref_funcs & base_funcs))

    cells = ', '.join(str(t) for t in totals)
    total = '   {}'.format(cells)

    return [header, total]


def generate_comparison_rst(base_mod, ref_mods, base_type, ref_types, cls):
    base_obj, base_fmt = import_mod(base_mod, cls)
    base_funcs = get_functions(base_obj)

    header = ', '.join([base_type] + ref_types)

    rows = []
    for f in sorted(base_funcs):
        base_cell = base_fmt.format(f)

        ref_cells = []
        for ref_mod in ref_mods:
            ref_obj, ref_fmt = import_mod(ref_mod, cls)
            ref_funcs = get_functions(ref_obj)

            ref_cell = r'\-'
            if f in ref_funcs:
                ref_cell = ref_fmt.format(f)

            ref_cells.append(ref_cell)

        cells = ', '.join([base_cell] + ref_cells)
        line = '   {}'.format(cells)
        rows.append(line)

    totals = generate_totals(base_mod, ref_mods, base_type, ref_types, cls)

    return ['.. csv-table::', '   :header: {}'.format(header), ''] + rows + totals


def section(header, base_mod, ref_mods, base_type, ref_types, cls=None):
    comparison_rst = generate_comparison_rst(base_mod, ref_mods, base_type, ref_types, cls)

    return [header, '~' * len(header), ''] + comparison_rst + ['']


def generate():
    base_mod, base_type = 'numpy', 'NumPy'

    try:
        import cupy
        ref_mods, ref_types = ['cupy', 'dpnp'], ['CuPy', 'DPNP']
    except ImportError:
        ref_mods, ref_types = ['dpnp'], ['DPNP']

    header = ' / '.join([base_type] + ref_types) + ' APIs'
    buf = [header, '-' * len(header), '']

    buf += section(
        'Module-Level',
        base_mod, ref_mods, base_type, ref_types)
    buf += section(
        'Multi-Dimensional Array',
        base_mod, ref_mods, base_type, ref_types, cls='ndarray')
    buf += section(
        'Linear Algebra',
        base_mod + '.linalg', [m + '.linalg' for m in ref_mods], base_type, ref_types)
    buf += section(
        'Discrete Fourier Transform',
        base_mod + '.fft', [m + '.fft' for m in ref_mods], base_type, ref_types)
    buf += section(
        'Random Sampling',
        base_mod + '.random', [m + '.random' for m in ref_mods], base_type, ref_types)

    return '\n'.join(buf)
