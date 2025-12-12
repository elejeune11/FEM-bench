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
        raise ValueError('Input `xi` must be a NumPy array.')
    if xi.ndim == 1:
        if xi.shape[0] != 2:
            raise ValueError('Input `xi` must have shape (2,) or (n, 2).')
        xi = xi[np.newaxis, :]
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('Input `xi` must have shape (2,) or (n, 2).')
    else:
        raise ValueError('Input `xi` must have shape (2,) or (n, 2).')
    if not np.all(np.isfinite(xi)):
        raise ValueError('Input `xi` contains non-finite values.')
    n_points = xi.shape[0]
    xi_c = 1 - xi[:, 0] - xi[:, 1]
    (N1, N2, N3) = (xi[:, 0] * (2 * xi[:, 0] - 1), xi[:, 1] * (2 * xi[:, 1] - 1), xi_c * (2 * xi_c - 1))
    (N4, N5, N6) = (4 * xi[:, 0] * xi[:, 1], 4 * xi[:, 1] * xi_c, 4 * xi[:, 0] * xi_c)
    N = np.stack([N1, N2, N3, N4, N5, N6], axis=1)[:, :, np.newaxis]
    dN1_dxi = np.stack([4 * xi[:, 0] - 1, np.zeros(n_points)], axis=1)
    dN2_dxi = np.stack([np.zeros(n_points), 4 * xi[:, 1] - 1], axis=1)
    dN3_dxi = np.stack([-(4 * xi_c - 1), -(4 * xi_c - 1)], axis=1)
    dN4_dxi = np.stack([4 * xi[:, 1], 4 * xi[:, 0]], axis=1)
    dN5_dxi = np.stack([-4 * xi[:, 1], 4 * xi_c - 4 * xi[:, 1]], axis=1)
    dN6_dxi = np.stack([4 * xi_c - 4 * xi[:, 0], -4 * xi[:, 0]], axis=1)
    dN_dxi = np.stack([dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi, dN5_dxi, dN6_dxi], axis=1)
    return (N, dN_dxi)