def FEM_2D_tri6_shape_fcns_and_derivatives_CC0_H0_T0(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized evaluation of quadratic (6-node) triangular shape functions and derivatives.
    Parameters
    ----------
    xi : np.ndarray
        Natural coordinates in the reference triangle.
    Returns
    -------
    N : np.ndarray
        Shape functions evaluated at the input points. Shape: (n, 6, 1).
        Node order: [N1, N2, N3, N4, N5, N6].
    dN_dxi : np.ndarray
        Partial derivatives w.r.t (ξ, η). Shape: (n, 6, 2).
        Columns correspond to [∂()/∂ξ, ∂()/∂η] in the same node order.
    Raises
    ------
    ValueError
        If `xi` is not a NumPy array.
        If `xi` has shape other than (2,) or (n, 2).
        If `xi` contains non-finite values (NaN or Inf).
    Notes
    -----
    Uses P2 triangle with ξ_c = 1 - ξ - η:
        N1 = ξ(2ξ - 1),   N2 = η(2η - 1),   N3 = ξ_c(2ξ_c - 1),
        N4 = 4ξη,         N5 = 4ηξ_c,       N6 = 4ξξ_c.
    """
    import numpy as np
    if not isinstance(xi, np.ndarray):
        raise ValueError('`xi` must be a NumPy array.')
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError('`xi` must have shape (2,) or (n, 2).')
        X = xi[None, :].astype(np.float64, copy=False)
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('`xi` must have shape (2,) or (n, 2).')
        X = xi.astype(np.float64, copy=False)
    else:
        raise ValueError('`xi` must have shape (2,) or (n, 2).')
    if not np.all(np.isfinite(X)):
        raise ValueError('`xi` contains non-finite values (NaN or Inf).')
    s = X[:, 0]
    t = X[:, 1]
    w = 1.0 - s - t
    N1 = s * (2.0 * s - 1.0)
    N2 = t * (2.0 * t - 1.0)
    N3 = w * (2.0 * w - 1.0)
    N4 = 4.0 * s * t
    N5 = 4.0 * t * w
    N6 = 4.0 * s * w
    N = np.column_stack((N1, N2, N3, N4, N5, N6))[:, :, None]
    dN_dxi_part = np.column_stack((4.0 * s - 1.0, np.zeros_like(s), -4.0 * w + 1.0, 4.0 * t, -4.0 * t, 4.0 * (w - s)))
    dN_deta_part = np.column_stack((np.zeros_like(t), 4.0 * t - 1.0, -4.0 * w + 1.0, 4.0 * s, 4.0 * (w - t), -4.0 * s))
    dN_dxi = np.stack((dN_dxi_part, dN_deta_part), axis=2)
    return (N, dN_dxi)