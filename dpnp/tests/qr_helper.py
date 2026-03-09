import numpy

from .helper import has_support_aspect64


def gram(x, xp):
    # Return Gram matrix: X^H @ X
    return xp.conjugate(x).swapaxes(-1, -2) @ x


def get_R_from_raw(h, m, n, xp):
    # Get reduced R from NumPy-style raw QR:
    # R = triu((tril(h))^T), shape (..., k, n)
    k = min(m, n)
    rt = xp.tril(h)
    r = xp.swapaxes(rt, -1, -2)
    r = xp.triu(r[..., :m, :n])
    return r[..., :k, :]


def check_qr(a_np, a_xp, mode, xp):
    # QR is not unique:
    # element-wise comparison with NumPy may differ by sign/phase.
    # To verify correctness use mode-dependent functional checks:
    # complete/reduced: check decomposition Q @ R = A
    # raw/r: check invariant R^H @ R = A^H @ A
    if mode in ("complete", "reduced"):
        res = xp.linalg.qr(a_xp, mode)
        assert xp.allclose(res.Q @ res.R, a_xp, atol=1e-5)

    # Since QR satisfies A = Q @ R with orthonormal Q (Q^H @ Q = I),
    # validate correctness via the invariant R^H @ R == A^H @ A
    # for raw/r modes
    elif mode == "raw":
        _, tau_np = numpy.linalg.qr(a_np, mode=mode)
        h_xp, tau_xp = xp.linalg.qr(a_xp, mode=mode)

        m, n = a_np.shape[-2], a_np.shape[-1]
        Rraw_xp = get_R_from_raw(h_xp, m, n, xp)

        # Use reduced QR as a reference:
        # reduced is validated via Q @ R == A
        exp_res = xp.linalg.qr(a_xp, mode="reduced")
        exp_r = exp_res.R
        assert xp.allclose(Rraw_xp, exp_r, atol=1e-4, rtol=1e-4)

        exp_xp = gram(a_xp, xp)

        # Compare R^H @ R == A^H @ A
        assert xp.allclose(gram(Rraw_xp, xp), exp_xp, atol=1e-4, rtol=1e-4)

        assert tau_xp.shape == tau_np.shape
        if not has_support_aspect64(tau_xp.sycl_device):
            assert tau_xp.dtype.kind == tau_np.dtype.kind
        else:
            assert tau_xp.dtype == tau_np.dtype

    else:  # mode == "r"
        r_xp = xp.linalg.qr(a_xp, mode="r")

        # Use reduced QR as a reference:
        # reduced is validated via Q @ R == A
        exp_res = xp.linalg.qr(a_xp, mode="reduced")
        exp_r = exp_res.R
        assert xp.allclose(r_xp, exp_r, atol=1e-4, rtol=1e-4)

        exp_xp = gram(a_xp, xp)

        # Compare R^H @ R == A^H @ A
        assert xp.allclose(gram(r_xp, xp), exp_xp, atol=1e-4, rtol=1e-4)
