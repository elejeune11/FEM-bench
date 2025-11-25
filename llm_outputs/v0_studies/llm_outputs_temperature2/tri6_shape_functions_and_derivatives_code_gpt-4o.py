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
    if xi.ndim == 1:
        xi = xi[np.newaxis, :]
    if xi.shape[1] != 2:
        raise ValueError('xi must have shape (2,) or (n, 2).')
    if not np.isfinite(xi).all():
        raise ValueError('xi must contain only finite values.')
    xi1 = xi[:, 0]
    xi2 = xi[:, 1]
    xi3 = 1 - xi1 - xi2
    N1 = xi1 * (2 * xi1 - 1)
    N2 = xi2 * (2 * xi2 - 1)
    N3 = xi3 * (2 * xi3 - 1)
    N4 = 4 * xi1 * xi2
    N5 = 4 * xi2 * xi3
    N6 = 4 * xi1 * xi3
    N = np.stack((N1, N2, N3, N4, N5, N6), axis=-1)[..., np.newaxis]
    dN1_dxi1 = 4 * xi1 - 1
    dN1_dxi2 = 0
    dN2_dxi1 = 0
    dN2_dxi2 = 4 * xi2 - 1
    dN3_dxi1 = -4 * xi3 + 1
    dN3_dxi2 = -4 * xi3 + 1
    dN4_dxi1 = 4 * xi2
    dN4_dxi2 = 4 * xi1
    dN5_dxi1 = -4 * xi2
    dN5_dxi2 = 4 * (1 - xi1 - 2 * xi2)
    dN6_dxi1 = 4 * (1 - 2 * xi1 - xi2)
    dN6_dxi2 = -4 * xi1
    dN_dxi = np.stack((np.stack((dN1_dxi1, dN1_dxi2), axis=-1), np.stack((dN2_dxi1, dN2_dxi2), axis=-1), np.stack((dN3_dxi1, dN3_dxi2), axis=-1), np.stack((dN4_dxi1, dN4_dxi2), axis=-1), np.stack((dN5_dxi1, dN5_dxi2), axis=-1), np.stack((dN6_dxi1, dN6_dxi2), axis=-1)), axis=1)
    return (N, dN_dxi)