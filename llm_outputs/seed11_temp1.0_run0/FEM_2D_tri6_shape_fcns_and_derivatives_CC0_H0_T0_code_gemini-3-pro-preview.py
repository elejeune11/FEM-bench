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
        raise ValueError('xi must be a NumPy array.')
    if not np.all(np.isfinite(xi)):
        raise ValueError('xi contains non-finite values (NaN or Inf).')
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError('xi must have shape (2,) or (n, 2).')
        points = xi[np.newaxis, :]
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('xi must have shape (2,) or (n, 2).')
        points = xi
    else:
        raise ValueError('xi must have shape (2,) or (n, 2).')
    xi_val = points[:, 0]
    eta_val = points[:, 1]
    xi_c = 1.0 - xi_val - eta_val
    N1 = xi_val * (2 * xi_val - 1)
    N2 = eta_val * (2 * eta_val - 1)
    N3 = xi_c * (2 * xi_c - 1)
    N4 = 4 * xi_val * eta_val
    N5 = 4 * eta_val * xi_c
    N6 = 4 * xi_val * xi_c
    N_stack = np.stack([N1, N2, N3, N4, N5, N6], axis=1)
    N = N_stack[:, :, np.newaxis]
    dN1_dxi = 4 * xi_val - 1
    dN1_deta = np.zeros_like(xi_val)
    dN2_dxi = np.zeros_like(xi_val)
    dN2_deta = 4 * eta_val - 1
    dN3_dxi = 1 - 4 * xi_c
    dN3_deta = 1 - 4 * xi_c
    dN4_dxi = 4 * eta_val
    dN4_deta = 4 * xi_val
    dN5_dxi = -4 * eta_val
    dN5_deta = 4 * (xi_c - eta_val)
    dN6_dxi = 4 * (xi_c - xi_val)
    dN6_deta = -4 * xi_val
    dN_dxi_comp = np.stack([dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi, dN5_dxi, dN6_dxi], axis=1)
    dN_deta_comp = np.stack([dN1_deta, dN2_deta, dN3_deta, dN4_deta, dN5_deta, dN6_deta], axis=1)
    dN_dxi = np.stack([dN_dxi_comp, dN_deta_comp], axis=2)
    return (N, dN_dxi)