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
    if not isinstance(xi, np.ndarray):
        raise ValueError('`xi` must be a NumPy array.')
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError('`xi` must have shape (2,) or (n, 2).')
        xi_2d = xi.reshape(1, 2)
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('`xi` must have shape (2,) or (n, 2).')
        xi_2d = xi
    else:
        raise ValueError('`xi` must have shape (2,) or (n, 2).')
    try:
        if not np.all(np.isfinite(xi_2d)):
            raise ValueError('`xi` contains non-finite values.')
    except Exception:
        raise ValueError('`xi` contains non-finite values.')
    xi_2d = xi_2d.astype(np.float64, copy=False)
    x = xi_2d[:, 0]
    y = xi_2d[:, 1]
    c = 1.0 - x - y
    N1 = x * (2.0 * x - 1.0)
    N2 = y * (2.0 * y - 1.0)
    N3 = c * (2.0 * c - 1.0)
    N4 = 4.0 * x * y
    N5 = 4.0 * y * c
    N6 = 4.0 * x * c
    n = xi_2d.shape[0]
    N = np.empty((n, 6, 1), dtype=np.float64)
    N[:, 0, 0] = N1
    N[:, 1, 0] = N2
    N[:, 2, 0] = N3
    N[:, 3, 0] = N4
    N[:, 4, 0] = N5
    N[:, 5, 0] = N6
    dN_dxi = np.empty((n, 6, 2), dtype=np.float64)
    dN_dxi[:, 0, 0] = 4.0 * x - 1.0
    dN_dxi[:, 0, 1] = 0.0
    dN_dxi[:, 1, 0] = 0.0
    dN_dxi[:, 1, 1] = 4.0 * y - 1.0
    common = 1.0 - 4.0 * c
    dN_dxi[:, 2, 0] = common
    dN_dxi[:, 2, 1] = common
    dN_dxi[:, 3, 0] = 4.0 * y
    dN_dxi[:, 3, 1] = 4.0 * x
    dN_dxi[:, 4, 0] = -4.0 * y
    dN_dxi[:, 4, 1] = 4.0 * (c - y)
    dN_dxi[:, 5, 0] = 4.0 * (c - x)
    dN_dxi[:, 5, 1] = -4.0 * x
    return (N, dN_dxi)