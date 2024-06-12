_unary_doc_template = """
dpnp.%s(x, out=None, order='K', dtype=None, casting="same_kind", **kwargs)

%s

For full documentation refer to :obj:`numpy.%s`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input arrays, expected to have %s data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, Cannot be provided
    together with `out`. Default: ``"K"``.
dtype : {None, dtype}, optional
    If provided, the destination array will have this dtype. Cannot be
    provided together with `out`. Default: ``None``.
casting : {"no", "equiv", "safe", "same_kind", "unsafe"}, optional
    Controls what kind of data casting may occur. Cannot be provided
    together with `out`. Default: ``"safe"``.

Returns
-------
out : dpnp.ndarray
%s

Limitations
-----------
Keyword arguments `where` and `subok` are supported with their default values.
Other keyword arguments is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

%s
"""

_binary_doc_template = """
dpnp.%s(x1, x2, out=None, order='K', dtype=None, casting="same_kind", **kwargs)

%s

For full documentation refer to :obj:`numpy.%s`.

Parameters
----------
x1, x2 : {dpnp.ndarray, usm_ndarray}
    Input arrays, expected to have %s data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
order : {None, "C", "F", "A", "K"}, optional
    Memory layout of the newly output array, Cannot be provided
    together with `out`. Default: ``"K"``.
dtype : {None, dtype}, optional
    If provided, the destination array will have this dtype. Cannot be
    provided together with `out`. Default: ``None``.
casting : {"no", "equiv", "safe", "same_kind", "unsafe"}, optional
    Controls what kind of data casting may occur. Cannot be provided
    together with `out`. Default: ``"safe"``.

Returns
-------
out : dpnp.ndarray
%s

Limitations
-----------
Keyword arguments `where` and `subok` are supported with their default values.
Other keyword arguments is currently unsupported.
Otherwise ``NotImplementedError`` exception will be raised.

%s
"""


name = "absolute"
dtypes = "numeric"
summary = """
Calculates the absolute value for each element `x_i` of input array `x`.
"""
returns = """
    An array containing the element-wise absolute values.
    For complex input, the absolute value is its magnitude.
    If `x` has a real-valued data type, the returned array has the
    same data type as `x`. If `x` has a complex floating-point data type,
    the returned array has a real-valued floating-point data type whose
    precision matches the precision of `x`.
"""
other = """
See Also
--------
:obj:`dpnp.fabs` : Calculate the absolute value element-wise excluding complex types.

Notes
-----
``dpnp.abs`` is a shorthand for this function.

Examples
--------
>>> import dpnp as np
>>> a = np.array([-1.2, 1.2])
>>> np.absolute(a)
array([1.2, 1.2])

>>> a = np.array(1.2 + 1j)
>>> np.absolute(a)
array(1.5620499351813308)
"""
abs_docstring = _unary_doc_template % (
    name,
    summary,
    name,
    dtypes,
    returns,
    other,
)


name = "add"
dtypes = "numeric"
summary = """
Calculates the sum for each element `x1_i` of the input array `x1` with the
respective element `x2_i` of the input array `x2`.
"""
returns = """
    An array containing the element-wise sums. The data type of the returned
    array is determined by the Type Promotion Rules.
"""
other = """
Notes
-----
Equivalent to `x1` + `x2` in terms of array broadcasting.

Examples
--------
>>> import dpnp as np
>>> a = np.array([1, 2, 3])
>>> b = np.array([1, 2, 3])
>>> np.add(a, b)
array([2, 4, 6])

>>> x1 = np.arange(9.0).reshape((3, 3))
>>> x2 = np.arange(3.0)
>>> np.add(x1, x2)
array([[  0.,   2.,   4.],
       [  3.,   5.,   7.],
       [  6.,   8.,  10.]])

The ``+`` operator can be used as a shorthand for ``add`` on
:class:`dpnp.ndarray`.

>>> x1 + x2
array([[  0.,   2.,   4.],
       [  3.,   5.,   7.],
       [  6.,   8.,  10.]])
"""
add_docstring = _binary_doc_template % (
    name,
    summary,
    name,
    dtypes,
    returns,
    other,
)
