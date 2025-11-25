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
        raise ValueError('xi must be a NumPy array')
    if xi.shape == (2,):
        xi = xi.reshape(1, 2)
        n = 1
    elif len(xi.shape) == 2 and xi.shape[1] == 2:
        n = xi.shape[0]
    else:
        raise ValueError('xi must have shape (2,) or (n, 2)')
    if not np.all(np.isfinite(xi)):
        raise ValueError('xi must contain only finite values')
    xi_val = xi[:, 0]
    eta = xi[:, 1]
    xi_c = 1 - xi_val - eta
    N1 = xi_val * (2 * xi_val - 1)
    N2 = eta * (2 * eta - 1)
    N3 = xi_c * (2 * xi_c - 1)
    N4 = 4 * xi_val * eta
    N5 = 4 * eta * xi_c
    N6 = 4 * xi_val * xi_c
    N = np.stack([N1, N2, N3, N4, N5, N6], axis=1).reshape(n, 6, 1)
    dN1_dxi = 4 * xi_val - 1
    dN1_deta = np.zeros_like(xi_val)
    dN2_dxi = np.zeros_like(xi_val)
    dN2_deta = 4 * eta - 1
    dN3_dxi = -(4 * xi_c - 1)
    dN3_deta = -(4 * xi_c - 1)
    dN4_dxi = 4 * eta
    dN4_deta = 4 * xi_val
    dN5_dxi = -4 * eta
    dN5_deta = 4 * (xi_c - eta)
    dN6_dxi = 4 * (xi_c - xi_val)
    dN6_deta = -4 * xi_val
    dN_dxi = np.stack([np.stack([dN1_dxi, dN1_deta], axis=1), np.stack([dN2_dxi, dN2_deta], axis=1), np.stack([dN3_dxi, dN3_deta], axis=1), np.stack([dN4_dxi, dN4_deta], axis=1), np.stack([dN5_dxi, dN5_deta], axis=1), np.stack([dN6_dxi, dN6_deta], axis=1)], axis=1)
    return (N, dN_dxi)