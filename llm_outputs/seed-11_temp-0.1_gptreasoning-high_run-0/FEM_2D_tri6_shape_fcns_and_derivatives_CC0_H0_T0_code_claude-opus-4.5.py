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
    if xi.shape == (2,):
        xi = xi.reshape(1, 2)
    elif xi.ndim != 2 or xi.shape[1] != 2:
        raise ValueError('xi must have shape (2,) or (n, 2).')
    if not np.all(np.isfinite(xi)):
        raise ValueError('xi contains non-finite values (NaN or Inf).')
    n = xi.shape[0]
    xi_coord = xi[:, 0]
    eta = xi[:, 1]
    xi_c = 1.0 - xi_coord - eta
    N1 = xi_coord * (2.0 * xi_coord - 1.0)
    N2 = eta * (2.0 * eta - 1.0)
    N3 = xi_c * (2.0 * xi_c - 1.0)
    N4 = 4.0 * xi_coord * eta
    N5 = 4.0 * eta * xi_c
    N6 = 4.0 * xi_coord * xi_c
    N = np.zeros((n, 6, 1))
    N[:, 0, 0] = N1
    N[:, 1, 0] = N2
    N[:, 2, 0] = N3
    N[:, 3, 0] = N4
    N[:, 4, 0] = N5
    N[:, 5, 0] = N6
    dN_dxi = np.zeros((n, 6, 2))
    dN1_dxi = 4.0 * xi_coord - 1.0
    dN1_deta = np.zeros(n)
    dN2_dxi = np.zeros(n)
    dN2_deta = 4.0 * eta - 1.0
    dN3_dxi = -4.0 * xi_c + 1.0
    dN3_deta = -4.0 * xi_c + 1.0
    dN4_dxi = 4.0 * eta
    dN4_deta = 4.0 * xi_coord
    dN5_dxi = -4.0 * eta
    dN5_deta = 4.0 * xi_c - 4.0 * eta
    dN6_dxi = 4.0 * xi_c - 4.0 * xi_coord
    dN6_deta = -4.0 * xi_coord
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