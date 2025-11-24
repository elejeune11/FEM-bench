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
    if xi.ndim == 1 and xi.shape == (2,):
        pts = xi.reshape(1, 2)
    elif xi.ndim == 2 and xi.shape[1] == 2:
        pts = xi
    else:
        raise ValueError('`xi` must have shape (2,) or (n, 2).')
    try:
        if not np.all(np.isfinite(pts)):
            raise ValueError('`xi` must contain only finite values.')
    except Exception:
        raise ValueError('`xi` must contain only finite numeric values.')
    xi1 = pts[:, 0]
    eta = pts[:, 1]
    xi_c = 1.0 - xi1 - eta
    N1 = xi1 * (2.0 * xi1 - 1.0)
    N2 = eta * (2.0 * eta - 1.0)
    N3 = xi_c * (2.0 * xi_c - 1.0)
    N4 = 4.0 * xi1 * eta
    N5 = 4.0 * eta * xi_c
    N6 = 4.0 * xi1 * xi_c
    N = np.stack([N1, N2, N3, N4, N5, N6], axis=1).reshape(pts.shape[0], 6, 1)
    zero = np.zeros_like(xi1)
    dN_dxi_col = np.stack([4.0 * xi1 - 1.0, zero, -4.0 * xi_c + 1.0, 4.0 * eta, -4.0 * eta, 4.0 * xi_c - 4.0 * xi1], axis=1)
    dN_deta_col = np.stack([zero, 4.0 * eta - 1.0, -4.0 * xi_c + 1.0, 4.0 * xi1, 4.0 * xi_c - 4.0 * eta, -4.0 * xi1], axis=1)
    dN_dxi = np.stack([dN_dxi_col, dN_deta_col], axis=2)
    return (N, dN_dxi)