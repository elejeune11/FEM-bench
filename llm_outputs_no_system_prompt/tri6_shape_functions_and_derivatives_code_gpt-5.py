def tri6_shape_functions_and_derivatives(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('`xi` must have shape (2,) or (n, 2).')
    else:
        raise ValueError('`xi` must have shape (2,) or (n, 2).')
    if not np.all(np.isfinite(xi)):
        raise ValueError('`xi` contains non-finite values.')
    pts = xi[None, :] if xi.ndim == 1 else xi
    pts = pts.astype(float, copy=False)
    x = pts[:, 0]
    y = pts[:, 1]
    c = 1.0 - x - y
    N1 = x * (2.0 * x - 1.0)
    N2 = y * (2.0 * y - 1.0)
    N3 = c * (2.0 * c - 1.0)
    N4 = 4.0 * x * y
    N5 = 4.0 * y * c
    N6 = 4.0 * x * c
    dN1_dx = 4.0 * x - 1.0
    dN1_dy = np.zeros_like(x)
    dN2_dx = np.zeros_like(x)
    dN2_dy = 4.0 * y - 1.0
    dc_dx = -1.0
    dc_dy = -1.0
    dN3_dc = 4.0 * c - 1.0
    dN3_dx = dN3_dc * dc_dx
    dN3_dy = dN3_dc * dc_dy
    dN4_dx = 4.0 * y
    dN4_dy = 4.0 * x
    dN5_dx = -4.0 * y
    dN5_dy = 4.0 * (c - y)
    dN6_dx = 4.0 * (c - x)
    dN6_dy = -4.0 * x
    N = np.stack([N1, N2, N3, N4, N5, N6], axis=1)[:, :, None]
    dN_dxi_x = np.stack([dN1_dx, dN2_dx, dN3_dx, dN4_dx, dN5_dx, dN6_dx], axis=1)
    dN_dxi_y = np.stack([dN1_dy, dN2_dy, dN3_dy, dN4_dy, dN5_dy, dN6_dy], axis=1)
    dN_dxi = np.stack([dN_dxi_x, dN_dxi_y], axis=2)
    return (N, dN_dxi)