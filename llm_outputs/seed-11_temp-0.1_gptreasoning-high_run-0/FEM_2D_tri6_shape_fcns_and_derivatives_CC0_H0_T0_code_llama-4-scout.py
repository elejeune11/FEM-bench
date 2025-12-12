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
        raise ValueError("Input 'xi' must be a NumPy array.")
    if xi.ndim not in [1, 2] or (xi.ndim == 1 and xi.shape[0] != 2) or (xi.ndim == 2 and xi.shape[1] != 2):
        raise ValueError("Input 'xi' must have shape (2,) or (n, 2).")
    if not np.all(np.isfinite(xi)):
        raise ValueError("Input 'xi' must contain only finite values (no NaN or Inf).")
    n = xi.shape[0] if xi.ndim == 2 else 1
    xi = xi.reshape(-1, 2)
    xi_c = 1 - np.sum(xi, axis=1).reshape(-1, 1)
    N1 = xi[:, 0:1] * (2 * xi[:, 0:1] - 1)
    N2 = xi[:, 1:2] * (2 * xi[:, 1:2] - 1)
    N3 = xi_c * (2 * xi_c - 1)
    N4 = 4 * xi[:, 0:1] * xi[:, 1:2]
    N5 = 4 * xi[:, 1:2] * xi_c
    N6 = 4 * xi[:, 0:1] * xi_c
    N = np.hstack((N1, N2, N3, N4, N5, N6)).reshape(-1, 6, 1)
    dN1_dxi = (4 * xi[:, 0:1] - 1).reshape(-1, 1, 1)
    dN1_deta = np.zeros((n, 1, 1))
    dN2_dxi = np.zeros((n, 1, 1))
    dN2_deta = (4 * xi[:, 1:2] - 1).reshape(-1, 1, 1)
    dN3_dxi = -2 * (2 * xi_c - 1).reshape(-1, 1, 1)
    dN3_deta = -2 * (2 * xi_c - 1).reshape(-1, 1, 1)
    dN4_dxi = 4 * xi[:, 1:2]
    dN4_deta = 4 * xi[:, 0:1]
    dN5_dxi = -4 * xi[:, 1:2]
    dN5_deta = 4 * xi_c - 4 * xi[:, 1:2]
    dN6_dxi = 4 * xi_c - 4 * xi[:, 0:1]
    dN6_deta = -4 * xi[:, 0:1]
    dN_dxi = np.stack((dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi, dN5_dxi, dN6_dxi), axis=1)
    dN_deta = np.stack((dN1_deta, dN2_deta, dN3_deta, dN4_deta, dN5_deta, dN6_deta), axis=1)
    dN_dxi = np.dstack((dN_dxi, dN_deta))
    return (N, dN_dxi)