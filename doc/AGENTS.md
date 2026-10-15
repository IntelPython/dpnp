# AGENTS.md - Documentation (doc/)

## Layout
- `conf.py`: Sphinx config (numpydoc, autosummary, furo). It registers the dpnp `Limitations` docstring section, documents the `DPNP*Func` wrapper objects as functions, and regenerates `reference/comparison_table.rst.inc` from `comparison_generator.py` on every build
- `reference/<category>.rst`: `.. autosummary::` lists under `.. currentmodule::` (`dpnp` or the subpackage); stubs are generated into `reference/generated/`
- `known_words.txt`: spelling allow-list
- `quick_start_guide.rst`: install and build-from-source instructions

## Rules for Changes
- A new public function goes in the autosummary block of the matching `reference/<category>.rst`; ufuncs also go in `reference/ufunc.rst`, `ndarray` members in `reference/ndarray.rst`. These pages already list many NumPy names that dpnp lacks, so check before adding a duplicate.
- A new namespace needs its own `reference/<name>.rst` and a toctree entry: in `reference/routines.rst` for NumPy namespaces (model: `reference/fft.rst`), in `reference/scipy.rst` for SciPy ones (model: `reference/scipy_linalg.rst`).
- A page kept out of every toctree on purpose, such as a placeholder, is marked `:orphan:`, or Sphinx warns about it.
- Move a deprecated function from its reference page to `reference/_deprecated.rst` (an orphan page that keeps its stub) and remove cross-links to it.
- A new wrapper base class (not a subclass of the existing `DPNP*Func` classes) must be added to `conf.py` and `_can_document_member`, or autodoc skips its instances.
- Update `quick_start_guide.rst` (and `README.md`) when build options or supported Python versions change, and `reference/array_api.rst` when the Array API version or the conforming namespaces change.
- Add new legitimate words to `known_words.txt`.
- Warnings don't fail the build (no `-W`), so check the log and the rendered page yourself.

## Verification
```bash
cd doc
make spelling
make html
```
This needs an importable dpnp build and the packages in `environments/building_docs.yml` (including CuPy, for the comparison table). CI (`build-sphinx.yml`) publishes builds of PRs from IntelPython/dpnp branches at `https://intelpython.github.io/dpnp/pull/<N>/index.html`; fork PRs get an artifact instead.
