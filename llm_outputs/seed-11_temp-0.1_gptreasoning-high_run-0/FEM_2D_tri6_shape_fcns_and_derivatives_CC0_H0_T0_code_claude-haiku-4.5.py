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
        raise ValueError('`xi` is not a NumPy array.')
    if xi.ndim == 1:
        if xi.shape[0] != 2:
            raise ValueError('`xi` has shape other than (2,) or (n, 2).')
        xi = xi.reshape(1, 2)
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('`xi` has shape other than (2,) or (n, 2).')
    else:
        raise ValueError('`xi` has shape other than (2,) or (n, 2).')
    if not np.all(np.isfinite(xi)):
        raise ValueError('`xi` contains non-finite values (NaN or Inf).')
    n = xi.shape[0]
    xi_coord = xi[:, 0]
    eta_coord = xi[:, 1]
    xi_c = 1 - xi_coord - eta_coord
    N1 = xi_coord * (2 * xi_coord - 1)
    N2 = eta_coord * (2 * eta_coord - 1)
    N3 = xi_c * (2 * xi_c - 1)
    N4 = 4 * xi_coord * eta_coord
    N5 = 4 * eta_coord * xi_c
    N6 = 4 * xi_coord * xi_c
    N = np.column_stack([N1, N2, N3, N4, N5, N6]).reshape(n, 6, 1)
    dN1_dxi = 4 * xi_coord - 1
    dN1_deta = np.zeros(n)
    dN2_dxi = np.zeros(n)
    dN2_deta = 4 * eta_coord - 1
    dN3_dxi = -1 * (2 * xi_c - 1) - xi_c * 2
    dN3_deta = -1 * (2 * xi_c - 1) - xi_c * 2
    dN4_dxi = 4 * eta_coord
    dN4_deta = 4 * xi_coord
    dN5_dxi = -4 * eta_coord
    dN5_deta = 4 * xi_c - 4 * eta_coord
    dN6_dxi = 4 * xi_c - 4 * xi_coord
    dN6_deta = -4 * xi_coord
    dN_dxi = np.zeros((n, 6, 2))
    dN_dxi[:, 0, 0] = dN1_dxi
    dN_dxi[:, 0, 1] = dN1_deta
    dN_dxi[:, 1, 0] = dN2_dxi
    dN_dxi[:, 1, 1] = dN2_deta
    dN_dxi[:, 2, 0] = dN3_dxi
    dN_dxi[:, 2, 1] = dN3_deta
    dN_dxi[:, 3, 0] = dN4_dxi
    dN_dxi[:, 3, 1] = dN4_deta
    dN_dxi[:, 4, 0] = dN5_dxi
    dN_dxi[:, 4, 1] = dN5_deta
    dN_dxi[:, 5, 0] = dN6_dxi
    dN_dxi[:, 5, 1] = dN6_deta
    return (N, dN_dxi)