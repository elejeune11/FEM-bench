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
        raise ValueError('xi must be a NumPy array.')
    is_single_point = xi.ndim == 1 and xi.shape == (2,)
    is_batch = xi.ndim == 2 and xi.shape[1] == 2
    if not (is_single_point or is_batch):
        raise ValueError('xi must have shape (2,) or (n, 2).')
    if not np.all(np.isfinite(xi)):
        raise ValueError('xi must contain non-finite values (NaN or Inf).')
    xi_2d = np.atleast_2d(xi)
    xi_ = xi_2d[:, 0]
    eta = xi_2d[:, 1]
    xi_c = 1.0 - xi_ - eta
    N1 = xi_ * (2 * xi_ - 1)
    N2 = eta * (2 * eta - 1)
    N3 = xi_c * (2 * xi_c - 1)
    N4 = 4 * xi_ * eta
    N5 = 4 * eta * xi_c
    N6 = 4 * xi_ * xi_c
    N = np.stack([N1, N2, N3, N4, N5, N6], axis=1)
    N = N[:, :, np.newaxis]
    dN1_dxi = 4 * xi_ - 1
    dN2_dxi = np.zeros_like(xi_)
    dN3_dxi = 1 - 4 * xi_c
    dN4_dxi = 4 * eta
    dN5_dxi = -4 * eta
    dN6_dxi = 4 * (xi_c - xi_)
    dN_dxi_col = np.stack([dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi, dN5_dxi, dN6_dxi], axis=1)
    dN1_deta = np.zeros_like(eta)
    dN2_deta = 4 * eta - 1
    dN3_deta = 1 - 4 * xi_c
    dN4_deta = 4 * xi_
    dN5_deta = 4 * (xi_c - eta)
    dN6_deta = -4 * xi_
    dN_deta_col = np.stack([dN1_deta, dN2_deta, dN3_deta, dN4_deta, dN5_deta, dN6_deta], axis=1)
    dN_dxi = np.stack([dN_dxi_col, dN_deta_col], axis=2)
    return (N, dN_dxi)