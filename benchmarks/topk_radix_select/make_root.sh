#!/bin/bash
# usage: make_root.sh SRC SORTING_SO DEST
#
# Makes DEST/dpnp, a tree of symlinks into SRC/dpnp (a built dpnp checkout,
# e.g. the editable development tree) with a real copy of SORTING_SO in
# place of dpnp/tensor/_tensor_sorting_impl*.so. Putting DEST on PYTHONPATH
# then runs that sorting extension with everything else from SRC, the other
# extensions and the Python code are the same for every build compared here.
set -eu
src=$(cd "$1" && pwd); so=$2; dest=$3
[[ -d $src/dpnp/tensor ]] || { echo "$src/dpnp/tensor not found" >&2; exit 1; }
[[ -f $so ]] || { echo "$so not found" >&2; exit 1; }
[[ -e $dest ]] && { echo "$dest exists, remove it first" >&2; exit 1; }
mkdir -p "$dest"
cp -as "$src/dpnp" "$dest/"
# drop the byte code caches, they would point back into SRC
find "$dest/dpnp" -name __pycache__ -type d -prune -exec rm -rf {} +
for f in "$dest"/dpnp/tensor/_tensor_sorting_impl*.so; do rm -f "$f"; done
cp "$so" "$dest/dpnp/tensor/$(basename "$(ls "$src"/dpnp/tensor/_tensor_sorting_impl*.so | head -1)")"
echo "made $dest from $src with $so"
